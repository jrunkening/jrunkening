+++
title = " VAE"
description = "A tiny VAE model to encode images into latent space for LDM"
weight = 11

tags = ["Generative Model", "VAE", "Diffusion"]

[params]
    math = true
    Toc = true
+++

A tiny variational autoencoder (VAE) model to encode images into latent space for latent diffusion models (LDM) in crunch computing power.

#### [Repository (private)](https://jingyu.tplinkdns.com/gitea/Diffusion/vae) {.centered-text}

## Implementation

In this implementation, I designed the vae using three residual blocks for both encoder and decoder.

```mermaid
graph LR
    Images[Images]--> E0(Lift)
    E0(Lift) --> E1(Res0) --> E2(Res1) --> E3(Res2)
    E3(Res2) --> E4Mean(Linear) --> Mean[Mean] --> Sample(Sample)
    E3(Res2) --> E4Var(Linear) --> Var[Var] --> Sample(Sample)
    Sample(Sample) --> Embedding[Embedding]
```

```mermaid
graph LR
    Embedding[Embedding] --> D4(Lift)
    D4(Lift) --> D3(Res2) --> D2(Res1) --> D1(Res0)
    D1(Res0) --> D0(Project) --> Reconstructed
```

In the encoder, I also added the squeeze-and-excitation mechanism in the trunk of the residual blocks to enhance the ability of feature extraction without dramatically increasing computational resources like attention does.

## Training

The pre-trained model on FFHQ 256x256 dataset can be found [here](https://jingyu.tplinkdns.com/gitea/Diffusion/vae/releases).
I use [NVIDIA TITAN Xp](https://www.nvidia.com/zh-tw/geforce/products/10series/star-wars-galactic-empire-titan-xp-collectors-edition/) with 12GB VRAM to train the model.
And use the mixed precision techniques with batch size `28` to maximize the usage of VRAM.
The learning rate is scheduled from `1e-4` to `0` using a cosine annealing schedule.

## Result

The VAE is training on FFHQ 256x256 dataset

| Original | Reconstructed |
|----------|---------------|
| ![](/projects/vae/assets/gallery_jingyu/original.jpg) | ![](/projects/vae/assets/gallery_jingyu/reconstructed.jpg) |
| ![](/projects/vae/assets/gallery_willa_fitzgerald/original.jpg) | ![](/projects/vae/assets/gallery_willa_fitzgerald/reconstructed.jpg) |

## Mask

To enable inpainting with LDM,
the encoder and decoder in this VAE are carefully designed so that a mask in pixel space can be easily mapped into latent space using simple steps like interpolation.
The decoder is able to understand masked images in latent space and can reconstruct the image correctly for both masked and unmasked areas in pixel space.

| Original | Target | Masked In Latent Space |
|----------|--------|------------------------|
| ![](/projects/vae/assets/gallery_jingyu/original.jpg) | ![](/projects/vae/assets/gallery_jingyu/masked.jpg) | ![](/projects/vae/assets/gallery_jingyu/masked_reconstructed.jpg) |
| ![](/projects/vae/assets/gallery_willa_fitzgerald/original.jpg) | ![](/projects/vae/assets/gallery_willa_fitzgerald/masked.jpg) | ![](/projects/vae/assets/gallery_willa_fitzgerald/masked_reconstructed.jpg) |
