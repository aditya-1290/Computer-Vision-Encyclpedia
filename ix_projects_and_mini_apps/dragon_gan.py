"""
Dragon Image GAN: Generate additional dragon images using DCGAN for data augmentation.

This script trains a DCGAN on existing dragon images and generates new synthetic images.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
latent_size = 100
image_size = 256
batch_size = 32
num_epochs = 300000
learning_rate = 0.0002
beta1 = 0.5

# Generator
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

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Custom dataset for loading images
class DragonDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        supported_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.avif', '.jfif', '.jpeg']
        for file in os.listdir(root_dir):
            if os.path.splitext(file)[1].lower() in supported_extensions:
                self.image_files.append(os.path.join(root_dir, file))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(img_path)
        if image is None:
            # Skip invalid images
            return self.__getitem__((idx + 1) % len(self.image_files))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train_gan():
    # Create output directory
    os.makedirs('dragon', exist_ok=True)

    # Data loading
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = DragonDataset('../images/', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Initialize models
    netG = Generator(latent_size).to(device)
    netD = Discriminator().to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # Load existing models if available for incremental training
    if os.path.exists('generator.pth'):
        netG.load_state_dict(torch.load('generator.pth'))
        print("Loaded existing Generator model.")
    if os.path.exists('discriminator.pth'):
        netD.load_state_dict(torch.load('discriminator.pth'))
        print("Loaded existing Discriminator model.")

    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    # Training loop
    print("Starting GAN training...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            # Update Discriminator
            netD.zero_grad()
            real_images = data.to(device)
            batch_size_real = real_images.size(0)
            label = torch.full((batch_size_real,), 1, dtype=torch.float, device=device)
            output = netD(real_images).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(batch_size_real, latent_size, 1, 1, device=device)
            fake_images = netG(noise)
            label.fill_(0)
            output = netD(fake_images.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Update Generator
            netG.zero_grad()
            label.fill_(1)
            output = netD(fake_images).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            # Save intermediate generated images for monitoring
            if i % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(dataloader)}] '
                      f'Loss D: {errD.item():.4f}, Loss G: {errG.item():.4f}')
                netG.eval()
                with torch.no_grad():
                    noise = torch.randn(1, latent_size, 1, 1, device=device)
                    fake_image = netG(noise)
                    fake_image = fake_image.squeeze(0)
                    fake_image = fake_image * 0.5 + 0.5  # Denormalize
                    fake_image = fake_image.cpu().numpy().transpose(1, 2, 0)
                    fake_image = (fake_image * 255).astype(np.uint8)
                    fake_image = cv2.cvtColor(fake_image, cv2.COLOR_RGB2BGR)
                    os.makedirs('intermediate_images', exist_ok=True)
                    cv2.imwrite(f'intermediate_images/epoch{epoch+1}_batch{i}.jpg', fake_image)
                netG.train()

    print("Training completed. Saving models...")

    # Save models
    torch.save(netG.state_dict(), 'generator.pth')
    torch.save(netD.state_dict(), 'discriminator.pth')
    print("Models saved as 'generator.pth' and 'discriminator.pth'.")
    print("Use generate_dragon_images.py to generate images.")

if __name__ == "__main__":
    train_gan()
