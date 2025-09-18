"""
Enhanced Dragon Image Generation: DCGAN + Transformer Implementation

This implementation includes:
1. Improved DCGAN with better architecture, training stability, and monitoring
2. Vision Transformer GAN (ViT-GAN) for comparison
3. Comprehensive training utilities and visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import json
from typing import Tuple, List, Optional

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
config = {
    "latent_size": 256,
    "image_size": 256,
    "batch_size": 32,
    "num_epochs": 300000,
    "learning_rate": 0.0002,
    "beta1": 0.5,
    "beta2": 0.999,
    "gp_weight": 10.0,  # Gradient penalty weight for WGAN-GP
    "n_critic": 5,  # Number of discriminator updates per generator update
    "save_interval": 100,
    "sample_interval": 50,
    "checkpoint_interval": 200,
}

# Save config
with open('training_config.json', 'w') as f:
    json.dump(config, f, indent=4)

# Enhanced Generator (DCGAN)
class ImprovedGenerator(nn.Module):
    def __init__(self, latent_size, img_channels=3, feature_map_size=64, img_size=256):
        super(ImprovedGenerator, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels
        self.feature_map_size = feature_map_size
        self.img_size = img_size
        
        # Calculate the initial size after projection
        self.initial_size = img_size // 32
        self.initial_channels = feature_map_size * 16
        
        self.main = nn.Sequential(
            # Input: latent_size x 1 x 1
            nn.ConvTranspose2d(latent_size, self.initial_channels, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.initial_channels),
            nn.ReLU(True),
            
            # State: (feature_map_size*16) x 4 x 4
            nn.ConvTranspose2d(self.initial_channels, feature_map_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.ReLU(True),
            
            # State: (feature_map_size*8) x 8 x 8
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),
            
            # State: (feature_map_size*4) x 16 x 16
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),
            
            # State: (feature_map_size*2) x 32 x 32
            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),
            
            # State: (feature_map_size) x 64 x 64
            nn.ConvTranspose2d(feature_map_size, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: img_channels x 128 x 128
        )
        
        # Additional layers to get to target size
        if img_size > 128:
            scale_factor = img_size // 128
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                nn.Conv2d(img_channels, img_channels, 3, 1, 1, bias=False),
                nn.Tanh()
            )
        else:
            self.upsample = nn.Identity()

    def forward(self, input):
        x = self.main(input)
        x = self.upsample(x)  # Upsample to target size if needed
        return x

# Enhanced Discriminator (WGAN-GP)
class ImprovedDiscriminator(nn.Module):
    def __init__(self, img_channels=3, feature_map_size=64, img_size=256):
        super(ImprovedDiscriminator, self).__init__()
        self.img_channels = img_channels
        self.feature_map_size = feature_map_size
        
        # Calculate the number of downsampling steps needed
        self.downsample_layers = nn.ModuleList()
        current_size = img_size
        current_channels = img_channels
        
        # Create downsampling layers dynamically based on image size
        while current_size > 4:
            next_channels = min(current_channels * 2, feature_map_size * 16)
            self.downsample_layers.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, next_channels, 4, 2, 1, bias=False),
                    nn.InstanceNorm2d(next_channels) if current_channels > img_channels else nn.Identity(),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            current_channels = next_channels
            current_size //= 2
        
        # Final layer
        self.final_layer = nn.Conv2d(current_channels, 1, 4, 1, 0, bias=False)

    def forward(self, input):
        x = input
        for layer in self.downsample_layers:
            x = layer(x)
        x = self.final_layer(x)
        return x.view(-1)

# Vision Transformer (ViT) Generator
class ViTGenerator(nn.Module):
    def __init__(self, latent_size=256, img_size=256, patch_size=16, 
                 num_layers=12, num_heads=8, hidden_dim=512, mlp_dim=1024):
        super(ViTGenerator, self).__init__()
        self.latent_size = latent_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size
        
        # Project latent vector to patch embeddings
        self.latent_to_patches = nn.Linear(latent_size, self.num_patches * hidden_dim)
        
        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_dim))
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=mlp_dim,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Layer normalization
        self.ln = nn.LayerNorm(hidden_dim)
        
        # Project to output patches
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, self.patch_dim),
            nn.Tanh()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, z):
        batch_size = z.size(0)
        
        # Project latent vector to patch embeddings
        x = self.latent_to_patches(z)  # (batch_size, num_patches * hidden_dim)
        x = x.view(batch_size, self.num_patches, -1)  # (batch_size, num_patches, hidden_dim)
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, 1:]  # Skip class token
        
        # Transformer encoder
        x = self.transformer(x)
        x = self.ln(x)
        
        # Project to output patches
        x = self.mlp_head(x)  # (batch_size, num_patches, patch_dim)
        
        # Reshape to image
        x = x.view(batch_size, self.num_patches, 3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4)  # (batch_size, 3, num_patches, patch_size, patch_size)
        
        # Rearrange patches to form image
        patches_per_side = self.img_size // self.patch_size
        x = x.view(batch_size, 3, patches_per_side, patches_per_side, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 2, 4, 3, 5)  # (batch_size, 3, patches_per_side, patch_size, patches_per_side, patch_size)
        x = x.contiguous().view(batch_size, 3, self.img_size, self.img_size)
        
        return x

# ViT Discriminator
class ViTDiscriminator(nn.Module):
    def __init__(self, img_size=256, patch_size=16, num_layers=6, 
                 num_heads=8, hidden_dim=512, mlp_dim=1024):
        super(ViTDiscriminator, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size
        
        # Patch embedding
        self.patch_embed = nn.Linear(self.patch_dim, hidden_dim)
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_dim))
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=mlp_dim,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Layer normalization
        self.ln = nn.LayerNorm(hidden_dim)
        
        # Classification head
        self.head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Create patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, 3, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4)  # (batch_size, num_patches, 3, patch_size, patch_size)
        patches = patches.contiguous().view(batch_size, self.num_patches, self.patch_dim)
        
        # Embed patches
        x = self.patch_embed(patches)  # (batch_size, num_patches, hidden_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, hidden_dim)
        
        # Add positional embeddings
        x = x + self.pos_embedding
        
        # Transformer encoder
        x = self.transformer(x)
        x = self.ln(x)
        
        # Use class token for classification
        x = x[:, 0]  # (batch_size, hidden_dim)
        x = self.head(x)  # (batch_size, 1)
        
        return x

# Gradient Penalty for WGAN-GP
def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN-GP"""
    # Random weight term for interpolation
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Custom dataset for loading images
class DragonDataset(Dataset):
    def __init__(self, root_dir, transform=None, img_size=256):
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        self.image_files = []
        supported_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']
        
        # Recursively find all image files
        for root, _, files in os.walk(root_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in supported_extensions:
                    self.image_files.append(os.path.join(root, file))
        
        print(f"Found {len(self.image_files)} images in {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            # Try multiple ways to load the image
            try:
                image = Image.open(img_path).convert('RGB')
            except:
                image = cv2.imread(img_path)
                if image is None:
                    raise ValueError(f"Could not load image: {img_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
            
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a random image instead
            return self.__getitem__((idx + 1) % len(self.image_files))

# Utility functions
def save_images(images, path, nrow=8):
    """Save a grid of images"""
    grid = torchvision.utils.make_grid(images, nrow=nrow, normalize=True)
    ndarr = grid.permute(1, 2, 0).cpu().numpy() * 255
    ndarr = ndarr.astype(np.uint8)
    im = Image.fromarray(ndarr)
    im.save(path)

def plot_losses(g_losses, d_losses, path):
    """Plot and save training losses"""
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path)
    plt.close()

# Training function for DCGAN
def train_dcgan():
    print("Training DCGAN...")
    
    # Create output directories
    os.makedirs('checkpoints/dcgan', exist_ok=True)
    os.makedirs('samples/dcgan', exist_ok=True)
    os.makedirs('logs/dcgan', exist_ok=True)
    
    # Tensorboard writer
    writer = SummaryWriter('logs/dcgan')
    
    # Data loading
    transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.CenterCrop(config['image_size']),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = DragonDataset('../images/', transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], 
                           shuffle=True, num_workers=4, pin_memory=True)
    
    # Initialize models
    netG = ImprovedGenerator(config['latent_size'], img_size=config['image_size']).to(device)
    netD = ImprovedDiscriminator(img_size=config['image_size']).to(device)
    
    # Initialize weights
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # Load existing models if available
    start_epoch = 0
    if os.path.exists('checkpoints/dcgan/generator.pth'):
        netG.load_state_dict(torch.load('checkpoints/dcgan/generator.pth'))
        print("Loaded existing Generator model.")
    if os.path.exists('checkpoints/dcgan/discriminator.pth'):
        netD.load_state_dict(torch.load('checkpoints/dcgan/discriminator.pth'))
        print("Loaded existing Discriminator model.")
    if os.path.exists('checkpoints/dcgan/epoch.txt'):
        with open('checkpoints/dcgan/epoch.txt', 'r') as f:
            start_epoch = int(f.read()) + 1
    
    # Optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=config['learning_rate'], 
                           betas=(config['beta1'], config['beta2']))
    optimizerD = optim.Adam(netD.parameters(), lr=config['learning_rate'], 
                           betas=(config['beta1'], config['beta2']))
    
    # Loss tracking
    g_losses = []
    d_losses = []
    
    # Fixed noise for sample generation
    fixed_noise = torch.randn(64, config['latent_size'], 1, 1, device=device)
    
    # Training loop
    print("Starting DCGAN training...")
    for epoch in range(start_epoch, config['num_epochs']):
        start_time = time.time()
        
        for i, real_images in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # Train Discriminator
            netD.zero_grad()
            
            # Real images
            real_output = netD(real_images)
            real_loss = -torch.mean(real_output)
            
            # Fake images
            noise = torch.randn(batch_size, config['latent_size'], 1, 1, device=device)
            fake_images = netG(noise)
            fake_output = netD(fake_images.detach())
            fake_loss = torch.mean(fake_output)
            
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(netD, real_images.data, fake_images.data)
            
            # Total discriminator loss
            d_loss = real_loss + fake_loss + config['gp_weight'] * gradient_penalty
            d_loss.backward()
            optimizerD.step()
            
            # Train Generator every n_critic steps
            if i % config['n_critic'] == 0:
                netG.zero_grad()
                fake_output = netD(fake_images)
                g_loss = -torch.mean(fake_output)
                g_loss.backward()
                optimizerG.step()
                
                # Record losses
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
                
                # Log to tensorboard
                writer.add_scalar('Loss/Generator', g_loss.item(), epoch * len(dataloader) + i)
                writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch * len(dataloader) + i)
            
            # Print training progress
            if i % 100 == 0:
                print(f'Epoch [{epoch}/{config["num_epochs"]}] Batch [{i}/{len(dataloader)}] '
                      f'Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}, '
                      f'GP: {gradient_penalty.item():.4f}')
        
        # Save generated images
        if epoch % config['sample_interval'] == 0:
            netG.eval()
            with torch.no_grad():
                fake_images = netG(fixed_noise)
                save_images(fake_images, f'samples/dcgan/epoch_{epoch}.png')
            netG.train()
        
        # Save checkpoints
        if epoch % config['checkpoint_interval'] == 0:
            torch.save(netG.state_dict(), f'checkpoints/dcgan/generator_{epoch}.pth')
            torch.save(netD.state_dict(), f'checkpoints/dcgan/discriminator_{epoch}.pth')
            torch.save(netG.state_dict(), 'checkpoints/dcgan/generator.pth')
            torch.save(netD.state_dict(), 'checkpoints/dcgan/discriminator.pth')
            with open('checkpoints/dcgan/epoch.txt', 'w') as f:
                f.write(str(epoch))
            
            # Plot losses
            plot_losses(g_losses, d_losses, 'checkpoints/dcgan/losses.png')
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch} completed in {epoch_time:.2f} seconds')
    
    print("DCGAN training completed.")
    writer.close()

# Training function for ViT-GAN
def train_vit_gan():
    print("Training ViT-GAN...")
    
    # Create output directories
    os.makedirs('checkpoints/vitgan', exist_ok=True)
    os.makedirs('samples/vitgan', exist_ok=True)
    os.makedirs('logs/vitgan', exist_ok=True)
    
    # Tensorboard writer
    writer = SummaryWriter('logs/vitgan')
    
    # Data loading
    transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.CenterCrop(config['image_size']),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = DragonDataset('../images/', transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], 
                           shuffle=True, num_workers=4, pin_memory=True)
    
    # Initialize models
    netG = ViTGenerator(config['latent_size'], img_size=config['image_size']).to(device)
    netD = ViTDiscriminator(img_size=config['image_size']).to(device)
    
    # Load existing models if available
    start_epoch = 0
    if os.path.exists('checkpoints/vitgan/generator.pth'):
        netG.load_state_dict(torch.load('checkpoints/vitgan/generator.pth'))
        print("Loaded existing ViT Generator model.")
    if os.path.exists('checkpoints/vitgan/discriminator.pth'):
        netD.load_state_dict(torch.load('checkpoints/vitgan/discriminator.pth'))
        print("Loaded existing ViT Discriminator model.")
    if os.path.exists('checkpoints/vitgan/epoch.txt'):
        with open('checkpoints/vitgan/epoch.txt', 'r') as f:
            start_epoch = int(f.read()) + 1
    
    # Optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=config['learning_rate'], 
                           betas=(config['beta1'], config['beta2']))
    optimizerD = optim.Adam(netD.parameters(), lr=config['learning_rate'], 
                           betas=(config['beta1'], config['beta2']))
    
    # Loss tracking
    g_losses = []
    d_losses = []
    
    # Fixed noise for sample generation
    fixed_noise = torch.randn(64, config['latent_size'], device=device)
    
    # Training loop
    print("Starting ViT-GAN training...")
    for epoch in range(start_epoch, config['num_epochs']):
        start_time = time.time()
        
        for i, real_images in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # Train Discriminator
            netD.zero_grad()
            
            # Real images
            real_output = netD(real_images)
            real_loss = -torch.mean(real_output)
            
            # Fake images
            noise = torch.randn(batch_size, config['latent_size'], device=device)
            fake_images = netG(noise)
            fake_output = netD(fake_images.detach())
            fake_loss = torch.mean(fake_output)
            
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(netD, real_images.data, fake_images.data)
            
            # Total discriminator loss
            d_loss = real_loss + fake_loss + config['gp_weight'] * gradient_penalty
            d_loss.backward()
            optimizerD.step()
            
            # Train Generator every n_critic steps
            if i % config['n_critic'] == 0:
                netG.zero_grad()
                fake_output = netD(fake_images)
                g_loss = -torch.mean(fake_output)
                g_loss.backward()
                optimizerG.step()
                
                # Record losses
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
                
                # Log to tensorboard
                writer.add_scalar('Loss/Generator', g_loss.item(), epoch * len(dataloader) + i)
                writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch * len(dataloader) + i)
            
            # Print training progress
            if i % 100 == 0:
                print(f'Epoch [{epoch}/{config["num_epochs"]}] Batch [{i}/{len(dataloader)}] '
                      f'Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}, '
                      f'GP: {gradient_penalty.item():.4f}')
        
        # Save generated images
        if epoch % config['sample_interval'] == 0:
            netG.eval()
            with torch.no_grad():
                fake_images = netG(fixed_noise)
                save_images(fake_images, f'samples/vitgan/epoch_{epoch}.png')
            netG.train()
        
        # Save checkpoints
        if epoch % config['checkpoint_interval'] == 0:
            torch.save(netG.state_dict(), f'checkpoints/vitgan/generator_{epoch}.pth')
            torch.save(netD.state_dict(), f'checkpoints/vitgan/discriminator_{epoch}.pth')
            torch.save(netG.state_dict(), 'checkpoints/vitgan/generator.pth')
            torch.save(netD.state_dict(), 'checkpoints/vitgan/discriminator.pth')
            with open('checkpoints/vitgan/epoch.txt', 'w') as f:
                f.write(str(epoch))
            
            # Plot losses
            plot_losses(g_losses, d_losses, 'checkpoints/vitgan/losses.png')
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch} completed in {epoch_time:.2f} seconds')
    
    print("ViT-GAN training completed.")
    writer.close()

# Image generation function
def generate_dragon_images(model_type='dcgan', num_images=100, output_dir='generated_dragons'):
    """Generate dragon images using trained model"""
    os.makedirs(output_dir, exist_ok=True)
    
    if model_type == 'dcgan':
        model = ImprovedGenerator(config['latent_size'], img_size=config['image_size']).to(device)
        model.load_state_dict(torch.load('checkpoints/dcgan/generator.pth', map_location=device))
        model.eval()
        
        with torch.no_grad():
            for i in range(num_images):
                noise = torch.randn(1, config['latent_size'], 1, 1, device=device)
                fake_image = model(noise)
                fake_image = fake_image * 0.5 + 0.5  # Denormalize
                
                # Save image
                img_path = os.path.join(output_dir, f'dragon_dcgan_{i}.png')
                save_images(fake_image, img_path, nrow=1)
    
    elif model_type == 'vitgan':
        model = ViTGenerator(config['latent_size'], img_size=config['image_size']).to(device)
        model.load_state_dict(torch.load('checkpoints/vitgan/generator.pth', map_location=device))
        model.eval()
        
        with torch.no_grad():
            for i in range(num_images):
                noise = torch.randn(1, config['latent_size'], device=device)
                fake_image = model(noise)
                fake_image = fake_image * 0.5 + 0.5  # Denormalize
                
                # Save image
                img_path = os.path.join(output_dir, f'dragon_vitgan_{i}.png')
                save_images(fake_image, img_path, nrow=1)
    
    print(f"Generated {num_images} images in {output_dir}")

if __name__ == "__main__":
    # Train both models
    train_dcgan()
    train_vit_gan()
    
    # Generate sample images
    generate_dragon_images('dcgan', 50, 'generated_dragons/dcgan')
    generate_dragon_images('vitgan', 50, 'generated_dragons/vitgan')
    
    print("All training and generation completed!")