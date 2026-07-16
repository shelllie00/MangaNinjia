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

##  Reference Links
- **Pipeline file:** [manganinjia_pipeline.py](file:///home/iismtl519-6895/2026summer/MangaNinjia/inference/manganinjia_pipeline.py)
- **ReferenceAttentionControl:** [mutual_self_attention_multi_scale.py#L22](file:///home/iismtl519-6895/2026summer/MangaNinjia/src/models/mutual_self_attention_multi_scale.py#L22)
- **PointNet:** [point_network.py#L6](file:///home/iismtl519-6895/2026summer/MangaNinjia/src/point_network.py#L6)
- **VAE & Diffusers:** Standard `diffusers` components imported at the top of the file.
---

*This document provides a section‑by‑section walkthrough of the MangaNinjia inference pipeline, tying the implementation back to the concepts presented in the original paper.*
