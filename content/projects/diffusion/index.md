+++
title = "Diffusion"
description = "The diffusion model for image generation"
weight = 10

tags = ["Generative Model", "Diffusion"]

[params]
    math = true
    Toc = true
+++

A simplified, home-made diffusion model that can be trained on 5GB VRAM

#### [Repository (private)](https://jingyu.tplinkdns.com/gitea/Diffusion/ddpm) {.centered-text}

## Implementation

This project implements a [DDPM](https://arxiv.org/abs/2006.11239) training framework
that departs from [standard latent diffusion pipelines](https://arxiv.org/abs/2112.10752).
It integrates [a custom VAE architecture](/projects/vae/)
with a serpentine sequence-processing module,
producing a latent space that is both sharper and less leaky,
making it well-suited for iterative denoising and high-fidelity reconstructions.

### Key Features

#### VAE with Localized and Aligned Latents

The [VAE](/projects/vae/)
has been carefully designed to mitigate the blurring and leakage issues common in conventional implementations.
Its decoder avoids oversized convolution kernels, which reduces cross-pixel contamination,
and the latent-to-pixel mapping is tightly aligned so that each latent feature corresponds to localized structures within the image.
To maintain global context without compromising local independence, the encoder incorporates [Squeeze-and-Excitation layers](https://arxiv.org/abs/1709.01507).
Together, these design choices create a latent space that balances local sharpness with global awareness,
providing a clean and reliable foundation for the diffusion process.

#### Serpentine Module: Sequential Feature Processing

The serpentine module reshapes latent tensors into sequences to capture directional dependencies across the feature map.
By scanning in multiple directions — horizontal, vertical, and others — the model can model context beyond what standard grid-based convolutions allow.
Padding and reshaping ensure that latent features flow naturally into sequence-processing layers inspired by state-space models (SSMs).
This approach provides richer contextual information at lower computational cost than full self-attention mechanisms,
while remaining more flexible than conventional CNN architectures.

#### DDPM in the Custom Latent Space

Building on these components, the DDPM implementation operates directly within the refined latent space.
Forward diffusion and noise scheduling are applied to the latent codes,
and the noise-prediction network leverages a UNet backbone enhanced with serpentine layers to model sequential dependencies.
During reverse denoising,
the cleaner and more structured latent representations from the VAE enable reconstructions with reduced blur compared to typical VAE-based diffusion pipelines.

## Training

Training runs entirely in the VAE latent space:
latents are noised with a standard DDPM schedule and a UNet denoiser enhanced with serpentine sequence blocks predicts the noise.
The localized, SE-augmented VAE keeps latents sharp and well-aligned, enabling efficient batches without leakage.
On an [NVIDIA RTX 5060 Laptop GPU](https://www.nvidia.com/en-us/geforce/news/rtx-5060-laptops-available-now/),
mixed precision (AMP) fits batch size 32 in ~5 GB VRAM.
The serpentine modules supply directional context with much lower memory than full self-attention,
maintaining high throughput and stable optimization over long runs.

## Result

After over 10,000 iterations, the system demonstrates stable convergence and consistent reconstruction quality.
The reduced leakage in the latent space makes the model particularly well-suited for localized editing and inpainting,
where maintaining precise alignment between latent codes and pixel space is critical.
By combining a localized, SE-augmented VAE with a serpentine sequence module,
this project establishes a distinct diffusion framework that emphasizes efficiency, sharpness, and structural fidelity in latent space,
positioning it as a flexible foundation for future generative modeling applications.

![](/jrunkening/projects/diffusion/assets/generated.jpg)

### Noise Schedule

This project uses a simple linear noise schedule: betas increase evenly from 1e−4 to 2e−2 across N−1 steps (step 0 represents the original image). At each step in the forward pass, the latent is blended with standard Gaussian noise according to the scheduled factor, gradually fading structure into noise. In the reverse pass, the denoiser removes the predicted noise to reconstruct the previous step’s latent; a small amount of fresh noise is added only when the step index is greater than 1 to keep the process stochastic. This modest schedule is deliberately gentle—easy to optimize, memory‑friendly, and sufficient to let the model learn the shape of the signal without washing it out.

![](/jrunkening/projects/diffusion/assets/demo_diffuser.jpg)

### Generate From Random Noise

Generation starts from pure Gaussian latent noise at the last diffusion step and proceeds backward one step at a time. At each step, the denoiser predicts the noise present in the current latent and removes it, reconstructing the previous step’s latent; a small random perturbation is injected except at the final step to keep sampling diverse. The loop runs from the highest step down to 1, with step 0 representing the clean latent. After the reverse loop finishes, the latent is decoded by the VAE decoder to produce the image. This procedure mirrors the training schedule, but in reverse: from structureless noise to a coherent sample, guided by the learned noise predictor and the gentle linear schedule.

| Result | Process |
|--------|---------|
| ![](/jrunkening/projects/diffusion/assets/generated_single.jpg) | ![](/jrunkening/projects/diffusion/assets/generated_single.gif)

### Inpainting

Inpainting is executed in the VAE latent space to respect locality and alignment. The original image is encoded once; I take the posterior mean as a clean latent anchor. The binary mask from pixel space is mapped to the latent grid by nearest‑neighbor downsampling that matches the encoder’s effective stride (here, stride‑8), so a masked latent tile corresponds exactly to the intended image region with no fractional overlap.

Sampling begins at the last diffusion step: the latent is initialized as random noise everywhere except on the masked (preserve) region, where it is clamped to the encoder mean. The reverse loop then runs from the highest step down to one. At each step, the Serpentine denoiser predicts the noise for the current latent, the diffuser steps back one level, and the masked region is re‑clamped to the mean. This simple clamp‑and‑denoise rhythm prevents information from bleeding across the mask boundary while allowing the unmasked area to evolve toward a coherent fill. A small random perturbation is only injected when the step index is greater than one, preserving diversity without disturbing the final alignment. After the loop, decoding the latent yields the inpainted image; non‑masked content is preserved exactly by construction.

This behavior hinges on the carefully designed VAE and the chosen latent space: localized features and tight latent‑to‑pixel alignment allow precise mask mapping, and the decoder’s restrained kernels avoid cross‑pixel contamination. Meanwhile, the serpentine sequence module supplies directional context across rows and columns, guiding the fill without washing details over the mask edge.

In the first set of examples below, the mask covers the face while the background latents are frozen to the encoder’s mean. The result preserves background textures, lighting, and edges exactly, while the masked facial region is resynthesized to match the scene. Boundary transitions remain clean because masked tiles are re‑clamped at every step and the decoder does not spill information across the mask.

| Original | Mask | Inpainted | Process |
|----------|------|-----------|---------|
| ![](/jrunkening/projects/diffusion/assets/original_jingyu.jpg) | ![](/jrunkening/projects/diffusion/assets/mask_jingyu.jpg) | ![](/jrunkening/projects/diffusion/assets/inpainted_jingyu_1.jpg) | ![](/jrunkening/projects/diffusion/assets/inpainted_jingyu_1.gif) |
| ![](/jrunkening/projects/diffusion/assets/original_jingyu.jpg) | ![](/jrunkening/projects/diffusion/assets/mask_jingyu.jpg) | ![](/jrunkening/projects/diffusion/assets/inpainted_jingyu_2.jpg) | ![](/jrunkening/projects/diffusion/assets/inpainted_jingyu_2.gif) |

In the second set, the mask targets the background while faces are preserved. Face latents stay clamped, so identity, expression, and contours are maintained bit‑for‑bit across the sequence. The model rebuilds the background to harmonize color and illumination with the preserved subject, and the serpentine context helps align large‑scale structure without bleeding into facial details.

| Original | Mask | Inpainted | Process |
|----------|------|-----------|---------|
| ![](/jrunkening/projects/diffusion/assets/original_willa_fitzgerald.jpg) | ![](/jrunkening/projects/diffusion/assets/mask_willa_fitzgerald.jpg) | ![](/jrunkening/projects/diffusion/assets/inpainted_willa_fitzgerald_1.jpg) | ![](/jrunkening/projects/diffusion/assets/inpainted_willa_fitzgerald_1.gif) |
| ![](/jrunkening/projects/diffusion/assets/original_willa_fitzgerald.jpg) | ![](/jrunkening/projects/diffusion/assets/mask_willa_fitzgerald.jpg) | ![](/jrunkening/projects/diffusion/assets/inpainted_willa_fitzgerald_2.jpg) | ![](/jrunkening/projects/diffusion/assets/inpainted_willa_fitzgerald_2.gif) |
| ![](/jrunkening/projects/diffusion/assets/original_kanna_seto.jpg) | ![](/jrunkening/projects/diffusion/assets/mask_kanna_seto.jpg) | ![](/jrunkening/projects/diffusion/assets/inpainted_kanna_seto_1.jpg) | ![](/jrunkening/projects/diffusion/assets/inpainted_kanna_seto_1.gif) |
| ![](/jrunkening/projects/diffusion/assets/original_kanna_seto.jpg) | ![](/jrunkening/projects/diffusion/assets/mask_kanna_seto.jpg) | ![](/jrunkening/projects/diffusion/assets/inpainted_kanna_seto_2.jpg) | ![](/jrunkening/projects/diffusion/assets/inpainted_kanna_seto_2.gif) |
