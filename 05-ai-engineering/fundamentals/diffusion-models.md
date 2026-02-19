> **TODO:** Expand this guide with detailed content.

# Diffusion Models

Generative models that learn to denoise data — the foundation of modern image, video, and audio generation.

## Topics to Cover

### Forward & Reverse Process
- Forward: gradually add Gaussian noise to data over T steps
- Reverse: neural network learns to denoise step by step
- DDPM (Denoising Diffusion Probabilistic Models) — the foundational paper

### Key Architectures
- **U-Net** — encoder-decoder with skip connections (original backbone)
- **DiT (Diffusion Transformer)** — replacing U-Net with transformers (Sora, SD3)
- **Latent Diffusion (Stable Diffusion)** — diffuse in latent space (VAE encoder/decoder), not pixel space
- **Classifier-Free Guidance** — trade diversity for quality with guidance scale

### Samplers & Scheduling
- DDPM vs DDIM (deterministic, fewer steps)
- Euler, DPM-Solver, UniPC — faster sampling
- Noise schedules: linear, cosine, shifted

### Conditioning
- Text-to-image: CLIP text encoder → cross-attention
- ControlNet — spatial conditioning (edges, depth, pose)
- IP-Adapter — image prompt conditioning
- Inpainting, outpainting, img2img

### Video & Audio
- Temporal attention layers for video (Sora, Runway)
- Audio diffusion (Stable Audio, MusicGen approach)

### Interview Questions
- How does classifier-free guidance work?
- Why latent diffusion vs pixel diffusion?
- How do you speed up inference (fewer steps, distillation)?
