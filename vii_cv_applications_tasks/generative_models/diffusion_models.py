"""
Diffusion Models: Denoising Diffusion Probabilistic Models (DDPM) for image generation.

Implementation uses PyTorch with a simplified U-Net for denoising.

Theory:
- Forward process: Add noise to data over T steps.
- Reverse process: Learn to denoise from noise to data.
- Loss: Simplified noise prediction loss.

Math: q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t) x_{t-1}, beta_t I)
      p(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 I)

Reference:
- Ho et al., Denoising Diffusion Probabilistic Models, NeurIPS 2020
"""

import torch
import torch.nn as nn
import math

class DiffusionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=256):
        super(DiffusionUNet, self).__init__()
        self.time_emb = nn.Embedding(1000, time_emb_dim)  # T=1000

        self.enc1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, 3, padding=1)

        self.dec3 = nn.Conv2d(256 + time_emb_dim, 128, 3, padding=1)
        self.dec2 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec1 = nn.Conv2d(64, out_channels, 3, padding=1)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        t_emb = self.time_emb(t).unsqueeze(-1).unsqueeze(-1)

        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(self.pool(e1)))
        e3 = self.relu(self.enc3(self.pool(e2)))

        d3 = self.relu(self.dec3(torch.cat([e3, t_emb], dim=1)))
        d2 = self.relu(self.dec2(self.up(d3) + e2))
        d1 = self.dec1(self.up(d2) + e1)
        return d1

class DDPM(nn.Module):
    def __init__(self, model, betas_start=1e-4, betas_end=0.02, T=1000):
        super(DDPM, self).__init__()
        self.model = model
        self.T = T
        self.betas = torch.linspace(betas_start, betas_end, T)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def forward_diffusion(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod[t])
        return sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise, noise

    def reverse_sample(self, xt, t):
        predicted_noise = self.model(xt, t)
        return predicted_noise

if __name__ == "__main__":
    unet = DiffusionUNet()
    ddpm = DDPM(unet)
    dummy_x = torch.randn(1, 3, 32, 32)
    t = torch.randint(0, 1000, (1,))
    noisy_x, noise = ddpm.forward_diffusion(dummy_x, t)
    predicted_noise = ddpm.reverse_sample(noisy_x, t)
    print(f"Original shape: {dummy_x.shape}, Noisy shape: {noisy_x.shape}, Predicted noise shape: {predicted_noise.shape}")
