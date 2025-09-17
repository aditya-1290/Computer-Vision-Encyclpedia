"""
Script to generate dragon images using a pre-trained Generator model.
"""

import torch
import torch.nn as nn
import os
import cv2
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

latent_size = 100
image_size = 256

# Generator model definition (same as in dragon_gan.py)
class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

def generate_images(num_images=400):
    # Create output directory
    os.makedirs('dragon', exist_ok=True)

    # Initialize generator and load weights
    netG = Generator(latent_size).to(device)
    if not os.path.exists('generator.pth'):
        print("Generator model weights not found. Please train the model first.")
        return
    netG.load_state_dict(torch.load('generator.pth'))
    netG.eval()

    with torch.no_grad():
        for i in range(num_images):
            noise = torch.randn(1, latent_size, 1, 1, device=device)
            fake_image = netG(noise)
            fake_image = fake_image.squeeze(0)
            fake_image = fake_image * 0.5 + 0.5  # Denormalize
            fake_image = fake_image.cpu().numpy().transpose(1, 2, 0)
            fake_image = (fake_image * 255).astype(np.uint8)
            fake_image = cv2.cvtColor(fake_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'dragon/{i+1:03d}_dragon.jpg', fake_image)

    print(f"Generated {num_images} dragon images in 'dragon/' folder.")

if __name__ == "__main__":
    generate_images()
