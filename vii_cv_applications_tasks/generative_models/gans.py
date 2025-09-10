"""
GANs: Generative Adversarial Networks for image generation.

Implementation includes DCGAN and a simplified StyleGAN.

Theory:
- Generator: Creates fake images from noise.
- Discriminator: Distinguishes real from fake.
- Adversarial training: Min-max game.

Math: Loss_G = -log(D(G(z))), Loss_D = -log(D(x)) - log(1 - D(G(z)))

Reference:
- Goodfellow et al., GANs, 2014
- Radford et al., DCGAN, 2016
- Karras et al., StyleGAN, 2019
"""

import torch
import torch.nn as nn

class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, ngf=64):
        super(DCGANGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class DCGANDiscriminator(nn.Module):
    def __init__(self, ndf=64):
        super(DCGANDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Simplified StyleGAN-like generator (not full implementation)
class StyleGANGenerator(nn.Module):
    def __init__(self, latent_dim=512):
        super(StyleGANGenerator, self).__init__()
        self.mapping = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.synthesis = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        w = self.mapping(z)
        w = w.unsqueeze(-1).unsqueeze(-1)
        return self.synthesis(w)

if __name__ == "__main__":
    # Test DCGAN
    gen = DCGANGenerator()
    disc = DCGANDiscriminator()
    noise = torch.randn(1, 100, 1, 1)
    fake_img = gen(noise)
    disc_out = disc(fake_img)
    print(f"DCGAN Fake image shape: {fake_img.shape}, Discriminator output shape: {disc_out.shape}")

    # Test StyleGAN
    style_gen = StyleGANGenerator()
    z = torch.randn(1, 512)
    style_img = style_gen(z)
    print(f"StyleGAN Image shape: {style_img.shape}")
