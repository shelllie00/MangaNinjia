# MangaNinjia Pipeline Explanation (`inference/manganinjia_pipeline.py`)

---

## Overview

The **MangaNinjiaPipeline** class implements the end‑to‑end inference pipeline for the MangaNinjia model. It wires together:
- a **Reference U‑Net** that extracts style/semantic keys & values from a reference image,
- a **ControlNet** that conditions on the reference image (edge maps, etc.),
- a **Denoising U‑Net** that performs diffusion‑based colorization of the target line‑art,
- a **PointNet** that injects user‑specified point correspondences,
- a **CLIP image encoder** that provides the `K_ref`/`V_ref` tensors,
- a **VAE** for latent‑space encoding/decoding of RGB images.

The pipeline follows the classic DiffusionPipeline pattern (scheduler → encode → denoise → decode) while adding **dual‑branch attention control** and **multi‑classifier‑free guidance**.

---

## Imports & Utilities

```python
# Lines 2‑8
from typing import Any, Dict, Union
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
```
These imports provide typing, image handling, tensor utilities, progress bars and the core PyTorch / Diffusers components.

---

##  Output Container

```python
# Lines 27‑30
class MangaNinjiaPipelineOutput(BaseOutput):
    img_np: np.ndarray
    img_pil: Image.Image
    to_save_dict: dict
```
A thin `BaseOutput` subclass that groups the final NumPy image, a Pillow image, and an auxiliary dictionary for intermediate tensors.

---

##  Pipeline Class Definition

```python
# Lines 33‑65
class MangaNinjiaPipeline(DiffusionPipeline):
    rgb_latent_scale_factor = 0.18215
```
The constructor (`__init__`, lines 36‑65) registers all sub‑modules with `self.register_modules`. Key modules include:
- `reference_unet` – `RefUNet2DConditionModel`
- `controlnet` – `ControlNetModel`
- `denoising_unet` – `UNet2DConditionModel`
- `vae` – `AutoencoderKL`
- Tokenizers & text encoders for both reference and control branches
- CLIP vision encoders (`refnet_image_encoder`, `controlnet_image_encoder`)
- `point_net` – the PointNet model
- `scheduler` – a `DDIMScheduler`
During init we also create a `CLIPImageProcessor` for preprocessing images.

---

## Pipeline Code Flow by Sections

The pipeline executes through a sequence of specific functions and logic blocks. Below is the step-by-step code flow and the meaning of the key functions and parameters.

### 1. The Entry Point (`__call__`)
This function orchestrates the entire inference process.
**Key Parameters:**
- `is_lineart` (bool): Whether the input `raw2` is pure lineart or requires edge extraction.
- `ref1` (Image): The reference image providing the style/color.
- `raw2` (Image): The target lineart or base image to be colorized.
- `edit2` (Image): The initial target image (often the same as `raw2`).
- `denosing_steps` (int): Number of diffusion steps (default 20).
- `guidance_scale_ref` (float): CFG scale for the reference image conditioning.
- `guidance_scale_point` (float): CFG scale for the point-based conditioning.
- `point_ref`, `point_main`: Tensors representing user-clicked points on the reference and target images.

**Flow inside `__call__`:**
1. **Helper Functions Definition:** Defines `img2embeds` (processes images through CLIP) and `prompt2embeds` (creates empty text embeddings for unconditional guidance).
2. **Embedding Generation:** Extracts semantic CLIP embeddings for the reference and control branches. (See **CLIP Embeddings and Cross-Attention** below).
3. **Image Pre-processing:** Converts `ref1`, `raw2`, and `edit2` to tensors normalized between `[-1, 1]` and wraps them in a `DataLoader`.
4. **PointNet Encoding:** Passes `point_ref` and `point_main` through the `point_net` to extract spatial point embeddings.
5. **ReferenceAttentionControl Setup:** Initializes the Writer (for the Reference UNet) and Reader (for the Denoising UNet) to enable feature fusion.
6. **Diffusion Loop:** Calls `self.single_infer()` to perform the actual generation.
7. **Post-processing:** Converts the output tensor back to a NumPy array and a PIL Image, resizing it if `match_input_res` is True. Returns a `MangaNinjiaPipelineOutput`.

### 2. CLIP Embeddings and Cross-Attention
- **Reference image embedding** (`refnet_encoder_hidden_states`): Extracted using the CLIP vision encoder inside `__call__`. This semantic embedding is passed as `encoder_hidden_states` to BOTH the Reference UNet and the Denoising UNet.
- Inside both UNets, this embedding directly feeds the standard **Cross-Attention layers (`attn2`)**, acting as the Keys and Values. This allows both networks to understand the high-level semantic content of the reference image.
- **Note:** The CLIP image embedding does *not* become `K_ref` and `V_ref`. It provides global semantic conditioning, while `K_ref` and `V_ref` are spatial features extracted by the Reference UNet.

### 3. Feature-Fusion Attention (Self-Attention Hooking)
While the CLIP embedding handles semantics via `attn2`, the spatial style/color details are transferred through the **Feature-Fusion Attention**, which hooks into the **Self-Attention layers (`attn1`)**:
1. **Reference UNet (Write Mode):** As it processes the reference image, `ReferenceAttentionControl` saves its spatial features (before the self-attention projection) into a `bank`. These saved features effectively become `K_ref` and `V_ref`.
2. **Denoising UNet (Read Mode):** The standard self-attention (`attn1`) is modified. It takes the Denoising UNet's own spatial features and concatenates them with the Reference UNet's spatial features (`bank`) along the sequence dimension.
   The modified `attn1` then uses the denoising latent as the Query (`Q`), and the concatenated features as the Keys and Values. This allows the denoising process to directly attend to the spatial features of the reference image.
3. **PointNet Injection:** Spatial point embeddings are added directly to the queries and keys before this concatenation, providing explicit point-correspondence guidance.

### 4. The Core Diffusion Loop (`single_infer`)
This function executes the timestep-by-timestep denoising process.
**Key Parameters:**
- Shares most parameters with `__call__`, but operates on pre-processed tensors instead of PIL Images.
- `reference_control_writer`, `reference_control_reader`: The attention hooks passed down from `__call__`.
- `refnet_encoder_hidden_states`, `controlnet_encoder_hidden_states`: The CLIP embeddings.

**Flow inside `single_infer`:**
1. **Timestep Initialization:** Sets up the `DDIMScheduler` timesteps.
2. **Latent Encoding:** Calls `self.encode_RGB(ref1)` to project the reference image into the VAE latent space.
3. **Noise Initialization:** Creates a random Gaussian noise tensor (`noisy_latents`) matching the target image's latent shape.
4. **Iterative Denoising (for each timestep `t`):**
   - **Step 0 Only (Caching):** Runs the Reference UNet on the reference latents to extract and cache the spatial features (`K_ref`, `V_ref`) into the `reference_control_writer`'s bank. The reader is then updated with these cached features.
   - **ControlNet Forward:** If enabled, runs the ControlNet to extract down-block and mid-block residuals based on the edge map.
   - **Denoising UNet Forward:** Predicts the noise residual. The input is duplicated 3 times (unconditional, reference, point) to support Classifier-Free Guidance.
   - **Guidance Combination:** Splits the predicted noise into 3 chunks and blends them using the guidance scales:
     ```python
     guided1 = noise_uncond + guidance_scale_ref * (noise_ref - noise_uncond)
     guided2 = noise_ref + guidance_scale_point * (noise_point - noise_ref)
     noise_pred = (guided1 + guided2) / 2
     ```
   - **Scheduler Step:** Computes the previous noisy sample (t-1) using `self.scheduler.step`.
5. **Cleanup:** Clears the reader and writer memory banks to free VRAM.
6. **Latent Decoding:** Calls `self.decode_RGB(noisy_latents)` to project the final denoised latent back into pixel space. Returns the image tensor.

### 5. Encoding/Decoding Helpers
- **`encode_RGB(img, generator)`**: Takes a `[-1, 1]` normalized RGB tensor, passes it through `self.vae.encode`, and multiplies by the VAE scaling factor (`0.18215`).
- **`decode_RGB(latents)`**: Takes a latent tensor, divides by the scaling factor, and passes it through `self.vae.decode` to get the RGB tensor.
- **`get_timesteps(...)`**: A utility to slice the scheduler timesteps if the user specifies a particular `denoising_start` strength (used for image-to-image workflows).

---

## How the Paper Concepts Map to Code
| Paper Concept | Code Hook |
|---------------|-----------|
| **Dual‑Branch Attention Control** | `ReferenceAttentionControl` writer/reader (lines 200‑218) + `hacked_basic_transformer_inner_forward` in `mutual_self_attention_multi_scale.py` |
| **Point‑Guided Control** | Point embeddings added to queries/keys inside `hacked_basic_transformer_inner_forward` – PointNet encodings passed via `point_ref` / `point_main` (lines 180‑181) |
| **Classifier‑Free Guidance** | `do_classifier_free_guidance` flag (line 134) and the three‑branch guidance combination (lines 432‑439) |
| **CLIP Image Embedding for Reference** | `img2embeds` using `self.refnet_image_encoder` (lines 94‑102) |
| **ControlNet Conditioning** | `self.controlnet` call inside the denoising loop (lines 406‑418) |

---

## Quick Usage Example
```python
from inference.manganinjia_pipeline import MangaNinjiaPipeline

pipe = MangaNinjiaPipeline(
    reference_unet=ref_unet,
    controlnet=cn_model,
    denoising_unet=den_unet,
    vae=vae,
    refnet_tokenizer=ref_tok,
    refnet_text_encoder=ref_text_enc,
    refnet_image_encoder=ref_img_enc,
    controlnet_tokenizer=cn_tok,
    controlnet_text_encoder=cn_text_enc,
    controlnet_image_encoder=cn_img_enc,
    scheduler=ddim_sched,
    point_net=pointnet,
)

output = pipe(
    is_lineart=False,
    ref1=Image.open('ref.png'),
    raw2=Image.open('lineart.png'),
    edit2=Image.open('lineart.png'),  # placeholder for target
    point_ref=point_tensor_ref,
    point_main=point_tensor_main,
)
output.img_pil.save('colored.png')
```

---

## VAE 4-Channel Latent Representation

The pipeline does not operate in pixel space. Following the paper, the reference image and target image are encoded into a **4-channel latent space** by the VAE encoder before any UNet processing.

```python
# single_infer(), line 354
ref1_latents = self.encode_RGB(ref1, generator=generator)
# Shape: (1, 4, H/8, W/8)
# e.g. for a 512x512 input: (1, 4, 64, 64)

# encode_RGB implementation, lines 466-467
rgb_latent = self.vae.encode(rgb_in).latent_dist.sample(generator)
rgb_latent = rgb_latent * self.rgb_latent_scale_factor  # scale factor = 0.18215
```

The initial noisy target latent is also created in this same 4-channel latent space:

```python
# single_infer(), line 371-373
noisy_edit2_latents = torch.randn(
    ref1_latents.shape, device=device, dtype=self.dtype
)  # shape: (1, 4, H/8, W/8)
```

**Why 4 channels?** The Stable Diffusion VAE compresses an RGB image `(3, H, W)` into a 4-channel latent `(4, H/8, W/8)`. This 8x spatial downsampling dramatically reduces the compute cost for all the UNet attention operations described below. At the end of inference, `decode_RGB` maps the latent back to an RGB image:

```python
# lines 482-483
rgb_latent = rgb_latent / self.rgb_latent_scale_factor
rgb_out = self.vae.decode(rgb_latent, return_dict=False)[0]
# shape: (1, 3, H, W)
```

---

## attn1 vs attn2: What is the Difference?

Every `BasicTransformerBlock` inside the UNet has **two separate attention layers**. They serve completely different purposes.

```
Input hidden_states (B, L, C)
       |
    [norm1]
       |
    [attn1]  <-- Self-attention / Feature-Fusion Attention
       | + residual
    [norm2]
       |
    [attn2]  <-- Cross-Attention with CLIP embedding
       | + residual
    [norm3]
       |
    [ff]     <-- Feed-forward MLP
       |
    Output hidden_states (B, L, C)
```

| | attn1 | attn2 |
|---|---|---|
| Type | Self-attention (modified to Feature-Fusion in read mode) | Cross-attention |
| Q source | The latent feature itself | The latent feature itself |
| K, V source | The latent feature (+ reference bank features in read mode) | CLIP image embedding |
| Purpose | Transfer spatial style/color from reference UNet | Inject global semantic content from reference image |
| Shape of K,V | `(B, L or 2L, C)` | `(B, 1, C)` — a single CLIP token |

---

## Detailed Shape Flow: attn2 (CLIP Cross-Attention)

### Step 1: CLIP encodes the reference image

```python
# manganinjia_pipeline.py, lines 94-102
clip_image = self.clip_image_processor.preprocess(ref1, return_tensors="pt").pixel_values
# shape: (1, 3, 224, 224)  -- CLIP always resizes to 224x224

clip_image_embeds = self.refnet_image_encoder(clip_image).image_embeds
# shape: (1, 768)  -- single global image vector (CLIP ViT-L gives 768-dim)

refnet_encoder_hidden_states = clip_image_embeds.unsqueeze(1)
# shape: (1, 1, 768)  -- one token in the sequence dimension
```

### Step 2: attn2 inside every BasicTransformerBlock

```python
# mutual_self_attention_multi_scale.py, lines 256-271
if self.attn2 is not None:
    norm_hidden_states = self.norm2(hidden_states)
    # norm_hidden_states shape: (B, L, C)
    # e.g. at the lowest resolution block: (3, 64*64=4096, 320)
    # (B=3 because of the 3x CFG duplication)

    hidden_states = self.attn2(
        norm_hidden_states,                    # Q: (B, L, C)
        encoder_hidden_states=encoder_hidden_states,  # K,V: (B, 1, 768)
        attention_mask=attention_mask,
    ) + hidden_states
    # attention output shape: (B, L, C)
```

**Key point:** K and V come from only **1 token** (the CLIP embedding). The attention weights are therefore `(B, L, 1)` — every spatial token in the latent attends to this single global reference descriptor. This is a very compact conditioning: no spatial detail, only global semantic.

---

## Detailed Shape Flow: attn1 (Feature-Fusion Attention)

### Step 1: Reference UNet runs in Write Mode (step 0 only)

```python
# mutual_self_attention_multi_scale.py, line 148-156
if MODE == "write":
    self.bank.append(norm_hidden_states.clone())
    # norm_hidden_states shape at this block: (B, L, C)
    # e.g. at 64x64 spatial resolution: (B, 4096, 320)
    # These are the RAW spatial features of the reference image latent
    # BEFORE any self-attention projection.

    attn_output = self.attn1(norm_hidden_states, ...)
    # The RefUNet still does its own normal self-attention
```

The bank now holds one tensor per UNet block containing the spatial features of the reference image.

### Step 2: Reader transfers the bank to the Denoising UNet

```python
# manganinjia_pipeline.py, line 404
reference_control_reader.update(reference_control_writer,
    point_embedding_ref=point_ref,
    point_embedding_main=point_main)

# Inside update() -- mutual_self_attention_multi_scale.py, line 370
r.bank = [v.clone().to(dtype) for v in w.bank]
# For each matched pair of blocks, the reader now holds the same
# spatial feature maps that the writer cached.
```

### Step 3: Denoising UNet runs in Read Mode (every timestep)

```python
# mutual_self_attention_multi_scale.py, lines 158-190
if MODE == "read":
    bank_fea = [rearrange(d.unsqueeze(1).repeat(1,1,1,1), "b t l c -> (b t) l c")
                for d in self.bank]
    # bank_fea[0] shape: (B, L, C)  -- same as norm_hidden_states
    # e.g. (3, 4096, 320) at the 64x64 block

    # Build K (with point embeddings injected):
    modify_norm_hidden_states = torch.cat(
        [norm_hidden_states + point_bank_main],   # denoising target features + point
        +[bank_fea[0]      + point_bank_ref],     # reference features + point
        dim=1
    )
    # shape: (B, 2L, C)  e.g. (3, 8192, 320)

    # Build V (no point embeddings for values):
    modify_norm_hidden_states_v = torch.cat(
        [norm_hidden_states] + bank_fea, dim=1
    )
    # shape: (B, 2L, C)  e.g. (3, 8192, 320)

    # Run the modified self-attention:
    hidden_states_uc = self.attn1(
        norm_hidden_states + point_bank_main,     # Q: (B, L, C)
        encoder_hidden_states=modify_norm_hidden_states,    # K: (B, 2L, C)
        encoder_hidden_states_v=modify_norm_hidden_states_v, # V: (B, 2L, C)
        attention_mask=attention_mask,
    ) + hidden_states
    # attention weights shape: (B, num_heads, L, 2L)
    # each target spatial token attends over L target tokens + L reference tokens
    # output shape: (B, L, C) -- same as input
```

**Key point:** Unlike `attn2` where V comes from a single CLIP token, here V has shape `(B, 2L, C)` — it contains the full spatial feature maps of both the target latent and the reference latent. This is how fine-grained color and style details are transferred: every target pixel can pull information from **any spatial location** in the reference.

---

## Classifier-Free Guidance and the 3x Batch

Because CFG is enabled, the batch dimension is tripled before entering the denoising loop. The 3 copies correspond to 3 different conditioning modes, each producing a different noise prediction:

```
Batch index 0 → uncond:  attn1 with target self-attention only (no reference features)
Batch index 1 → ref:     attn1 with target + reference features (no point)
Batch index 2 → point:   attn1 with target + reference features + point embeddings
```

After the UNet forward pass, the noise is split and blended:
```python
noise_uncond, noise_ref, noise_point = noise_pred.chunk(3)
guided1 = noise_uncond + guidance_scale_ref   * (noise_ref   - noise_uncond)
guided2 = noise_ref    + guidance_scale_point * (noise_point - noise_ref)
noise_pred = (guided1 + guided2) / 2
```

---

##  Reference Links
- **Pipeline file:** [manganinjia_pipeline.py](file:///home/iismtl519-6895/2026summer/MangaNinjia/inference/manganinjia_pipeline.py)
- **ReferenceAttentionControl:** [mutual_self_attention_multi_scale.py#L22](file:///home/iismtl519-6895/2026summer/MangaNinjia/src/models/mutual_self_attention_multi_scale.py#L22)
- **PointNet:** [point_network.py#L6](file:///home/iismtl519-6895/2026summer/MangaNinjia/src/point_network.py#L6)
- **VAE & Diffusers:** Standard `diffusers` components imported at the top of the file.
---

*This document provides a section‑by‑section walkthrough of the MangaNinjia inference pipeline, tying the implementation back to the concepts presented in the original paper.*

---

## Internal Layers of the Reference UNet and Denoising UNet

Both the Reference UNet (`RefUNet2DConditionModel`, defined in [refunet_2d_condition.py](file:///home/iismtl519-6895/2026summer/MangaNinjia/src/models/refunet_2d_condition.py)) and the Denoising UNet (`UNet2DConditionModel`, from [unet_2d_condition.py](file:///home/iismtl519-6895/2026summer/MangaNinjia/src/models/unet_2d_condition.py)) share the **same overall U-Net architecture** — a modified Stable Diffusion UNet. The critical difference is what happens inside the `BasicTransformerBlock` attention layers due to `ReferenceAttentionControl` monkey-patching.

### Overall Architecture (Shared by Both UNets)

```
Input latent: (B, 4, H/8, W/8)         <-- 4-channel VAE latent
       |
   [conv_in]      Conv2d(4 -> 320)      --> (B, 320, H/8, W/8)
       |
   [time_proj + time_embedding]          timestep t -> sinusoidal -> MLP
       |                                 --> emb: (B, 1280)
       |
   [down_blocks x4]                      Encoder: downsample
       |
   [mid_block]                           Bottleneck (smallest spatial res)
       |
   [up_blocks x4]                        Decoder: upsample + skip connections
       |
   [conv_out]     Conv2d(320 -> 4)       --> (B, 4, H/8, W/8)   [Denoising UNet only]
```

### Layer 0: Input Projection (`conv_in`)

```python
# refunet_2d_condition.py, lines 295-300
self.conv_in = nn.Conv2d(
    in_channels=4,              # VAE latent has 4 channels
    out_channels=320,           # block_out_channels[0]
    kernel_size=3, padding=1,
)
# Input:  (B, 4,   H/8, W/8)
# Output: (B, 320, H/8, W/8)
```

Lifts the 4-channel VAE latent to the 320-channel feature space used by the first UNet block.

### Layer 1: Timestep Embedding

```python
# lines 319-333
self.time_proj = Timesteps(320, flip_sin_to_cos=True, freq_shift=0)
# Encodes scalar t into sinusoidal vector: (B,) -> (B, 320)

self.time_embedding = TimestepEmbedding(in_channels=320, time_embed_dim=1280)
# Two-layer MLP: 320 -> 1280 -> 1280
# Output: emb shape (B, 1280)
```

`emb` is injected inside every `ResnetBlock2D` via a linear projection, conditioning all convolutions on the current noise level `t`.

### Layer 2: Down Blocks (`self.down_blocks`, 4 blocks total)

```
Block 0: CrossAttnDownBlock2D   channels: 320  -> 320,  spatial: H/8  -> H/8  (no downsample in this block)
Block 1: CrossAttnDownBlock2D   channels: 320  -> 640,  spatial: H/8  -> H/16
Block 2: CrossAttnDownBlock2D   channels: 640  -> 1280, spatial: H/16 -> H/32
Block 3: DownBlock2D            channels: 1280 -> 1280, spatial: H/32 -> H/64 (no attention)
```

```python
# lines 491-526
for i, down_block_type in enumerate(down_block_types):
    down_block = get_down_block(
        down_block_type,
        num_layers=2,                     # 2 ResNet layers per block
        out_channels=block_out_channels[i],  # [320, 640, 1280, 1280]
        temb_channels=1280,
        cross_attention_dim=768,          # matches CLIP embedding dim
        num_attention_heads=8,
        add_downsample=not is_final_block,
    )
```

Each `CrossAttnDownBlock2D` internally runs:
```
for 2 layers:
    sample = ResnetBlock2D(sample, emb)
    # Conv3x3 -> GroupNorm -> SiLU, with emb injected via linear proj
    # Input/Output: (B, C, H', W')

    sample = BasicTransformerBlock(sample, encoder_hidden_states=clip_embed)
    # Contains attn1 (feature-fusion) and attn2 (CLIP cross-attention)
    # Input/Output: (B, L, C)  where L = H' * W'

save sample as skip connection (res_samples)
Downsample2D(sample)  --> halves H', W'
```

### Layer 3: Mid Block (Bottleneck)

```python
# lines 529-545
self.mid_block = UNetMidBlock2DCrossAttn(
    in_channels=1280,
    temb_channels=1280,
    cross_attention_dim=768,
    num_attention_heads=8,
)
# Spatial shape throughout: (B, 1280, H/64, W/64)
# for 512x512 input: (B, 1280, 8, 8)
```

Structure:
```
ResnetBlock2D(1280 -> 1280)
BasicTransformerBlock    <-- attn1 + attn2 (ReferenceAttentionControl active here too)
ResnetBlock2D(1280 -> 1280)
```

### Layer 4: Up Blocks (`self.up_blocks`, 4 blocks total)

The decoder mirrors the encoder. Skip connections from down blocks are concatenated (standard U-Net).

```
Block 0: UpBlock2D          (1280+1280 -> 1280, H/64 -> H/32)   [skip from down block 3]
Block 1: CrossAttnUpBlock2D (1280+1280 -> 1280, H/32 -> H/16)   [skip from down block 2]
Block 2: CrossAttnUpBlock2D (1280+640  -> 640,  H/16 -> H/8 )   [skip from down block 1]
Block 3: CrossAttnUpBlock2D (640+320   -> 320,  H/8  -> H/8 )   [skip from down block 0]
```

Each `CrossAttnUpBlock2D` runs:
```
for 3 layers (one extra compared to down blocks, to handle skip connection):
    sample = torch.cat([sample, skip_connection], dim=1)  # concat skip
    sample = ResnetBlock2D(sample, emb)
    sample = BasicTransformerBlock(sample, encoder_hidden_states=clip_embed)
        # attn1: feature-fusion with reference bank (read mode)
        # attn2: CLIP cross-attention
Upsample2D --> double H', W'
```

### Layer 5: Output Projection (`conv_out`)

```python
# refunet_2d_condition.py, lines 647-652 -- COMMENTED OUT for RefUNet
# self.conv_out = nn.Conv2d(320, out_channels=4, kernel_size=3, padding=1)
```

**This is the key structural difference between the two UNets:**

- **Denoising UNet**: `conv_out` is **enabled**. Final output: `(B, 4, H/8, W/8)` — the predicted noise residual at each timestep.
- **Reference UNet**: `conv_out` is **disabled (commented out)**. The RefUNet never produces a usable image output. Its sole purpose is to execute the forward pass so that `ReferenceAttentionControl` (write mode) intercepts the `norm_hidden_states` inside each `BasicTransformerBlock` and saves them to `self.bank`.

### Why the Reference UNet Only Runs at Timestep 0

```python
# manganinjia_pipeline.py, lines 393-404
if i == 0:                              # Only at the very first denoising step
    self.reference_unet(
        refnet_input.repeat(3, 1, 1, 1),
        torch.zeros_like(t),            # t = 0, meaning "clean image, no noise"
        encoder_hidden_states=refnet_encoder_hidden_states,
    )
    reference_control_reader.update(reference_control_writer, ...)
```

The reference image latent is clean (no noise added to it). Passing `t=0` tells the UNet's time embedding that this is a fully denoised input. The resulting spatial features stored in `bank` represent the pure content of the reference image, unaffected by any noise schedule. These cached features remain constant for all subsequent denoising timesteps.

### Complete Summary Table

| | Reference UNet | Denoising UNet |
|---|---|---|
| Class | `RefUNet2DConditionModel` | `UNet2DConditionModel` |
| Input latent | Reference image, clean `(B, 4, H/8, W/8)` | Noisy target `(B, 4, H/8, W/8)`, 3x repeated for CFG |
| Timestep | Always `t=0` | Current denoising step `t` |
| `conv_in` | 4 -> 320 channels | 4 -> 320 channels |
| `time_embedding` | Encodes `t=0` | Encodes current `t` |
| `attn1` role | Normal self-attention; hidden states saved to `bank` | Feature-fusion: Q=target, K/V=cat(target, reference bank) |
| `attn2` role | CLIP cross-attention with `refnet_encoder_hidden_states` | CLIP cross-attention with `refnet_encoder_hidden_states` |
| `conv_out` | Disabled (commented out) | Enabled, outputs predicted noise `(B, 4, H/8, W/8)` |
| Runs every step? | No, only step 0 | Yes, every denoising timestep |


### What the code does                                                                                                 
                                                                                                                         
    # mutual_self_attention_multi_scale.py, lines 320-332                                                                
                                                                                                                         
    if self.fusion_blocks == "midup":                                                                                    
        # Patch ONLY the mid block and up blocks                                                                         
        attn_modules = [                                                                                                 
            module                                                                                                       
            for module in (                                                                                              
                torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)                                          
            )                                                                                                            
            if isinstance(module, BasicTransformerBlock)                                                                 
        ]                                                                                                                
                                                                                                                         
    elif self.fusion_blocks == "full":                                                                                   
        # Patch EVERY BasicTransformerBlock in the entire UNet                                                           
        attn_modules = [                                                                                                 
            module                                                                                                       
            for module in torch_dfs(self.unet)   # traverses ALL modules                                                 
            if isinstance(module, BasicTransformerBlock)                                                                 
        ]                                                                                                                
                                                                                                                         
  The  fusion_blocks  parameter controls which transformer blocks get their  forward  method monkey-patched with         
  hacked_basic_transformer_inner_forward .                                                                               
  ──────                                                                                                                 
  ### The difference                                                                                                     
                                                                                                                         
                                  │  "midup"                                │  "full" 
  ────────────────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────────
   Blocks patched                 │  mid_block  +  up_blocks  only          │  down_blocks  +  mid_block  +  up_blocks 
   Number of patched blocks       │ ~13                                     │ ~26 (all of them)
   Reference features injected at │ Decoder and bottleneck only             │ Encoder + bottleneck + decoder
   Computational cost             │ Lower                                   │ Higher
   Style transfer strength        │ Weaker (reference features only affect  │ Stronger (reference features affect every
                                  │ the decoding stage)                     │ stage)
  ──────                                                                                                                 
  ### Why the distinction matters architecturally                                                                        
                                                                                                                         
  In a standard U-Net, down blocks act as a feature extractor — they encode the input into increasingly abstract         
  representations. Up blocks act as the generator — they decode the features back to image space, guided by skip         
  connections.                                                                                                           
                                                                                                                         
  •  "midup" : The encoder ( down_blocks ) processes the noisy target independently, without any reference influence.    
  Only when the features pass through the bottleneck and decoder do they get fused with reference features. This is the  
  more conservative approach.                                                                                            
  •  "full" : The reference features are injected from the very first encoder layer. Every encoding step of the noisy    
  target can already attend to the reference. This gives the reference image much more influence over the final output — 
  both structural and color-level — but at twice the patching cost.                                                      
  ──────
  ### What MangaNinjia uses
  
    # manganinjia_pipeline.py, lines 200-218
    reference_control_writer = ReferenceAttentionControl(
        self.reference_unet,
        mode="write",
        fusion_blocks="full",   # <-- full is used
        ...
    )
    reference_control_reader = ReferenceAttentionControl(
        self.denoising_unet,
        mode="read",
        fusion_blocks="full",   # <-- full is used
        ...
    )
  
  MangaNinjia uses  "full"  because manga colorization requires very strong style and color fidelity to the reference.   
  The model needs to transfer fine-grained color information at all scales — not just at the decoder level. Using        
  "midup"  would weaken the color transfer, which would be unacceptable for a colorization task.                         
  ──────
  ### When to use which in general
  
   Use case                                                  │ Recommended setting
  ───────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────
   Strong style/color transfer (e.g., colorization)          │  "full" 
   Subtle style influence, preserve target structure         │  "midup" 
   Faster inference / lower VRAM usage                       │  "midup" 
   The reference has very different content from the target  │  "midup"  (less risk of feature leakage)

