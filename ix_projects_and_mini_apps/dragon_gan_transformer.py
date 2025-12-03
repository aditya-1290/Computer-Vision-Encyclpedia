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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CyclicLR
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
    "num_epochs": 30000,
    "learning_rate": 0.0001,
    "beta1": 0.5,
    "beta2": 0.999,
    "gp_weight": 10.0,
    "n_critic": 5,
    "save_interval": 100,
    "sample_interval": 50,
    "checkpoint_interval": 200,
    
    # Enhanced optimizer settings
    "optimizer": "adam",  # Options: "adam", "rmsprop", "adagrad", "adamw"
    "weight_decay": 0.0001,  # L2 regularization
    "grad_clip": 1.0,  # Gradient clipping value
    "lr_scheduler": "cosine",  # Options: "cosine", "plateau", "cyclic", "step", "none"
    
    # Scheduler specific parameters
    "lr_min": 1e-6,  # Minimum learning rate
    "lr_patience": 10,  # Patience for ReduceLROnPlateau
    "lr_factor": 0.5,  # Factor for ReduceLROnPlateau
    "cycle_step_size": 2000,  # Step size for cyclic LR
    
    # Stabilization parameters
    "gradient_penalty_type": "wgan-gp",  # Options: "wgan-gp", "dragan", "none"
    "spectral_norm": True,  # Use spectral normalization for stability
}

# Save config
with open('training_config.json', 'w') as f:
    json.dump(config, f, indent=4)
    
# Spectral Normalization wrapper
def spectral_norm(module, use_spectral_norm=True):
    if use_spectral_norm:
        return nn.utils.spectral_norm(module)
    return module

# Enhanced Generator (DCGAN)
class ImprovedGenerator(nn.Module):
    def __init__(self, latent_size, img_channels=3, feature_map_size=64, img_size=256, use_spectral_norm=True):
        super(ImprovedGenerator, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels
        self.feature_map_size = feature_map_size
        self.img_size = img_size
        
        # Calculate the initial size after projection
        self.initial_size = img_size // 32
        self.initial_channels = feature_map_size * 16
        
        # Use spectral normalization for stability
        conv_transpose = spectral_norm(nn.ConvTranspose2d, use_spectral_norm)
        batch_norm = nn.BatchNorm2d
        
        self.main = nn.Sequential(
            # Input: latent_size x 1 x 1
            conv_transpose(latent_size, self.initial_channels, 4, 1, 0, bias=False),
            batch_norm(self.initial_channels),
            nn.ReLU(True),
            
            # State: (feature_map_size*16) x 4 x 4
            conv_transpose(self.initial_channels, feature_map_size * 8, 4, 2, 1, bias=False),
            batch_norm(feature_map_size * 8),
            nn.ReLU(True),
            
            # State: (feature_map_size*8) x 8 x 8
            conv_transpose(feature_map_size * 8, feature_map_size * 4, 4, 2, 1, bias=False),
            batch_norm(feature_map_size * 4),
            nn.ReLU(True),
            
            # State: (feature_map_size*4) x 16 x 16
            conv_transpose(feature_map_size * 4, feature_map_size * 2, 4, 2, 1, bias=False),
            batch_norm(feature_map_size * 2),
            nn.ReLU(True),
            
            # State: (feature_map_size*2) x 32 x 32
            conv_transpose(feature_map_size * 2, feature_map_size, 4, 2, 1, bias=False),
            batch_norm(feature_map_size),
            nn.ReLU(True),
            
            # State: (feature_map_size) x 64 x 64
            conv_transpose(feature_map_size, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
        # Additional layers to get to target size
        if img_size > 128:
            scale_factor = img_size // 128
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                spectral_norm(nn.Conv2d(img_channels, img_channels, 3, 1, 1, bias=False), use_spectral_norm),
                nn.Tanh()
            )
        else:
            self.upsample = nn.Identity()

    def forward(self, input):
        x = self.main(input)
        x = self.upsample(x)
        return x
    
# Enhanced Discriminator (WGAN-GP)
class ImprovedDiscriminator(nn.Module):
    def __init__(self, img_channels=3, feature_map_size=64, img_size=256, use_spectral_norm=True):
        super(ImprovedDiscriminator, self).__init__()
        self.img_channels = img_channels
        self.feature_map_size = feature_map_size
        
        # Use spectral normalization for stability
        conv2d = spectral_norm(nn.Conv2d, use_spectral_norm)
        
        self.downsample_layers = nn.ModuleList()
        current_size = img_size
        current_channels = img_channels
        
        # Create downsampling layers dynamically based on image size
        while current_size > 4:
            next_channels = min(current_channels * 2, feature_map_size * 16)
            self.downsample_layers.append(
                nn.Sequential(
                    conv2d(current_channels, next_channels, 4, 2, 1, bias=False),
                    nn.InstanceNorm2d(next_channels) if current_channels > img_channels else nn.Identity(),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.2)  # Add dropout for regularization
                )
            )
            current_channels = next_channels
            current_size //= 2
        
        # Final layer
        self.final_layer = conv2d(current_channels, 1, 4, 1, 0, bias=False)

    def forward(self, input):
        x = input
        for layer in self.downsample_layers:
            x = layer(x)
        x = self.final_layer(x)
        return x.view(-1)

    def forward(self, input):
        x = input
        for layer in self.downsample_layers:
            x = layer(x)
        x = self.final_layer(x)
        return x.view(-1)

# Vision Transformer (ViT) Generator
class VisionTransformerGenerator(nn.Module):
    def __init__(self, latent_size, img_channels=3, patch_size=16, dim=512, depth=6, heads=8, mlp_dim=2048, 
                 img_size=256, use_spectral_norm=True):
        super(VisionTransformerGenerator, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.dim = dim
        
        num_patches = (img_size // patch_size) ** 2
        patch_dim = img_channels * patch_size * patch_size
        
        # Linear projection of latent vector
        self.to_patch_embedding = nn.Sequential(
            spectral_norm(nn.Linear(latent_size, dim), use_spectral_norm),
            nn.LayerNorm(dim)
        )
        
        # Position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=0.0,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, depth)
        
        # Output projection
        self.to_patches = nn.Sequential(
            spectral_norm(nn.Linear(dim, patch_dim), use_spectral_norm),
            nn.Tanh()
        )
        
        # Additional conv layers for better image quality
        self.final_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(img_channels, 64, 3, 1, 1, bias=False), use_spectral_norm),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(64, img_channels, 3, 1, 1, bias=False), use_spectral_norm),
            nn.Tanh()
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Project latent vector
        x = self.to_patch_embedding(x.view(batch_size, -1))
        
        # Add class token and position embeddings
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        
        # Apply transformer
        x = self.transformer(x)
        
        # Remove class token and reshape to patches
        x = x[:, 1:]
        x = self.to_patches(x)
        
        # Reshape to image
        h = w = self.img_size // self.patch_size
        x = x.view(batch_size, h, w, self.patch_size, self.patch_size, self.img_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(batch_size, self.img_channels, self.img_size, self.img_size)
        
        # Apply final convolutions for better quality
        x = self.final_conv(x)
        return x


# ViT Discriminator
class VisionTransformerDiscriminator(nn.Module):
    def __init__(self, img_channels=3, patch_size=16, dim=512, depth=6, heads=8, mlp_dim=2048, 
                 img_size=256, use_spectral_norm=True):
        super(VisionTransformerDiscriminator, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        num_patches = (img_size // patch_size) ** 2
        patch_dim = img_channels * patch_size * patch_size
        
        # Initial conv layers for better feature extraction
        self.feature_extraction = nn.Sequential(
            spectral_norm(nn.Conv2d(img_channels, 64, 3, 1, 1, bias=False), use_spectral_norm),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(64, img_channels, 3, 1, 1, bias=False), use_spectral_norm)
        )
        
        # Patch embedding
        self.to_patch_embedding = nn.Sequential(
            spectral_norm(nn.Linear(patch_dim, dim), use_spectral_norm),
            nn.LayerNorm(dim)
        )
        
        # Position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=0.1,  # Add some dropout for regularization
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, depth)
        
        # Output projection
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            spectral_norm(nn.Linear(dim, 1), use_spectral_norm)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Apply initial feature extraction
        x = self.feature_extraction(x)
        
        # Reshape into patches
        x = x.view(batch_size, -1, self.patch_size * self.patch_size * x.shape[1])
        
        # Linear projection of patches
        x = self.to_patch_embedding(x)
        
        # Add class token and position embeddings
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        
        # Apply transformer
        x = self.transformer(x)
        
        # Use only the class token for final prediction
        x = x[:, 0]
        
        # Project to scalar output
        x = self.to_logits(x)
        return x.view(-1)
    

# Enhanced Gradient Penalty with different types
def compute_gradient_penalty(discriminator, real_samples, fake_samples, penalty_type="wgan-gp"):
    """Calculates the gradient penalty loss with different variants"""
    
    if penalty_type == "none":
        return torch.tensor(0.0, device=real_samples.device)
    
    # Random weight term for interpolation
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=real_samples.device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates, device=real_samples.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    
    if penalty_type == "wgan-gp":
        # WGAN-GP penalty: (||gradients||_2 - 1)^2
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    elif penalty_type == "dragan":
        # DRAGAN penalty: (||gradients||_2)^2
        gradient_penalty = (gradients.norm(2, dim=1) ** 2).mean()
    else:
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

# Custom dataset for loading images
class DragonDataset(Dataset):
    def __init__(self, root_dir, transform=None, img_size=256):
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        self.image_files = []
        supported_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.avif', '.jfif']
        
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
    
# Advanced Optimizer Factory
def create_optimizer(model, optimizer_type, learning_rate, beta1=0.5, beta2=0.999, weight_decay=0.0001):
    """Create optimizer with specified type and parameters"""
    
    if optimizer_type.lower() == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate, 
                         betas=(beta1, beta2), weight_decay=weight_decay)
    elif optimizer_type.lower() == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=learning_rate, 
                            weight_decay=weight_decay, momentum=0.9)
    elif optimizer_type.lower() == "adagrad":
        return optim.Adagrad(model.parameters(), lr=learning_rate, 
                            weight_decay=weight_decay)
    elif optimizer_type.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=learning_rate, 
                          betas=(beta1, beta2), weight_decay=weight_decay)
    else:
        print(f"Unknown optimizer {optimizer_type}, using Adam as default")
        return optim.Adam(model.parameters(), lr=learning_rate, 
                         betas=(beta1, beta2), weight_decay=weight_decay)

# Learning Rate Scheduler Factory
def create_scheduler(optimizer, scheduler_type, num_epochs, lr_min=1e-6, patience=10, factor=0.5, step_size=2000):
    """Create learning rate scheduler"""
    
    if scheduler_type.lower() == "cosine":
        return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr_min)
    elif scheduler_type.lower() == "plateau":
        return ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor, verbose=True)
    elif scheduler_type.lower() == "cyclic":
        return CyclicLR(optimizer, base_lr=lr_min, max_lr=config['learning_rate'], 
                       step_size_up=step_size, mode='triangular2')
    elif scheduler_type.lower() == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    else:
        # No scheduler
        return None


# Gradient Clipping
def clip_gradients(model, clip_value):
    """Clip gradients to prevent explosion"""
    if clip_value > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
# Training function for DCGAN
# Enhanced Training Function with Advanced Optimizers
def train_dcgan():
    print("Training DCGAN with Advanced Optimizers...")
    
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
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = DragonDataset('../images/', transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], 
                           shuffle=True, num_workers=4, pin_memory=True)
    
    # Initialize models with spectral normalization
    netG = ImprovedGenerator(config['latent_size'], img_size=config['image_size'], 
                           use_spectral_norm=config['spectral_norm']).to(device)
    netD = ImprovedDiscriminator(img_size=config['image_size'], 
                               use_spectral_norm=config['spectral_norm']).to(device)
    
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
    
    # Create advanced optimizers
    optimizerG = create_optimizer(netG, config['optimizer'], config['learning_rate'], 
                                 config['beta1'], config['beta2'], config['weight_decay'])
    optimizerD = create_optimizer(netD, config['optimizer'], config['learning_rate'], 
                                 config['beta1'], config['beta2'], config['weight_decay'])
    
    # Create learning rate schedulers
    schedulerG = create_scheduler(optimizerG, config['lr_scheduler'], config['num_epochs'],
                                 config['lr_min'], config['lr_patience'], config['lr_factor'], config['cycle_step_size'])
    schedulerD = create_scheduler(optimizerD, config['lr_scheduler'], config['num_epochs'],
                                 config['lr_min'], config['lr_patience'], config['lr_factor'], config['cycle_step_size'])
    
    # Loss tracking
    g_losses = []
    d_losses = []
    current_lr = config['learning_rate']
    
    # Fixed noise for sample generation
    fixed_noise = torch.randn(64, config['latent_size'], 1, 1, device=device)
    
    # Training loop
    print(f"Starting DCGAN training with {config['optimizer'].upper()} optimizer...")
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
            gradient_penalty = compute_gradient_penalty(
                netD, real_images.data, fake_images.data, config['gradient_penalty_type']
            )
            
            # Total discriminator loss
            d_loss = real_loss + fake_loss + config['gp_weight'] * gradient_penalty
            d_loss.backward()
            
            # Clip gradients
            clip_gradients(netD, config['grad_clip'])
            optimizerD.step()
            
            # Train Generator every n_critic steps
            if i % config['n_critic'] == 0:
                netG.zero_grad()
                fake_output = netD(fake_images)
                g_loss = -torch.mean(fake_output)
                g_loss.backward()
                
                # Clip gradients
                clip_gradients(netG, config['grad_clip'])
                optimizerG.step()
                
                # Record losses
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
                
                # Log to tensorboard
                writer.add_scalar('Loss/Generator', g_loss.item(), epoch * len(dataloader) + i)
                writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch * len(dataloader) + i)
                writer.add_scalar('Loss/Gradient_Penalty', gradient_penalty.item(), epoch * len(dataloader) + i)
            
            # Print training progress with more informative metrics
            if i % 100 == 0:
                # Calculate Wasserstein distance estimate
                wasserstein_dist = real_loss.item() - fake_loss.item()
                
                print(f'Epoch [{epoch}/{config["num_epochs"]}] Batch [{i}/{len(dataloader)}] '
                      f'D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}, '
                      f'GP: {gradient_penalty.item():.4f}, WD: {wasserstein_dist:.4f}, '
                      f'LR: {current_lr:.6f}')
        
        # Update learning rate
        if schedulerG is not None:
            if isinstance(schedulerG, ReduceLROnPlateau):
                schedulerG.step(g_losses[-1] if g_losses else 1.0)
            else:
                schedulerG.step()
        
        if schedulerD is not None:
            if isinstance(schedulerD, ReduceLROnPlateau):
                schedulerD.step(d_losses[-1] if d_losses else 1.0)
            else:
                schedulerD.step()
        
        # Get current learning rate
        current_lr = optimizerG.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
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
            
            # Save optimizer states
            torch.save(optimizerG.state_dict(), 'checkpoints/dcgan/optimizerG.pth')
            torch.save(optimizerD.state_dict(), 'checkpoints/dcgan/optimizerD.pth')
            
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
    print("Training ViT-GAN with Advanced Optimizers...")
    
    # Create output directories
    os.makedirs('checkpoints/vitgan', exist_ok=True)
    os.makedirs('samples/vitgan', exist_ok=True)
    os.makedirs('logs/vitgan', exist_ok=True)
    
    # Tensorboard writer
    writer = SummaryWriter('logs/vitgan')
    
    # Data loading with the same transformations as DCGAN
    transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.CenterCrop(config['image_size']),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = DragonDataset('../images/', transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], 
                          shuffle=True, num_workers=4, pin_memory=True)
    
    # Initialize ViT-GAN models with spectral normalization
    netG = VisionTransformerGenerator(config['latent_size'], img_size=config['image_size'],
                                    use_spectral_norm=config['spectral_norm']).to(device)
    netD = VisionTransformerDiscriminator(img_size=config['image_size'],
                                        use_spectral_norm=config['spectral_norm']).to(device)
    
    # Initialize weights
    def weights_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # Load existing models if available
    start_epoch = 0
    if os.path.exists('checkpoints/vitgan/generator.pth'):
        netG.load_state_dict(torch.load('checkpoints/vitgan/generator.pth'))
        print("Loaded existing Generator model.")
    if os.path.exists('checkpoints/vitgan/discriminator.pth'):
        netD.load_state_dict(torch.load('checkpoints/vitgan/discriminator.pth'))
        print("Loaded existing Discriminator model.")
    if os.path.exists('checkpoints/vitgan/epoch.txt'):
        with open('checkpoints/vitgan/epoch.txt', 'r') as f:
            start_epoch = int(f.read()) + 1
    
    # Create advanced optimizers with the same settings as DCGAN
    optimizerG = create_optimizer(netG, config['optimizer'], config['learning_rate'],
                                config['beta1'], config['beta2'], config['weight_decay'])
    optimizerD = create_optimizer(netD, config['optimizer'], config['learning_rate'],
                                config['beta1'], config['beta2'], config['weight_decay'])
    
    # Create learning rate schedulers
    schedulerG = create_scheduler(optimizerG, config['lr_scheduler'], config['num_epochs'],
                                config['lr_min'], config['lr_patience'], config['lr_factor'],
                                config['cycle_step_size'])
    schedulerD = create_scheduler(optimizerD, config['lr_scheduler'], config['num_epochs'],
                                config['lr_min'], config['lr_patience'], config['lr_factor'],
                                config['cycle_step_size'])
    
    # Loss tracking
    g_losses = []
    d_losses = []
    current_lr = config['learning_rate']
    
    # Fixed noise for sample generation
    fixed_noise = torch.randn(64, config['latent_size'], 1, 1, device=device)
    
    # Training loop
    print(f"Starting ViT-GAN training with {config['optimizer'].upper()} optimizer...")
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
            gradient_penalty = compute_gradient_penalty(
                netD, real_images.data, fake_images.data, config['gradient_penalty_type']
            )
            
            # Total discriminator loss
            d_loss = real_loss + fake_loss + config['gp_weight'] * gradient_penalty
            d_loss.backward()
            
            # Clip gradients
            clip_gradients(netD, config['grad_clip'])
            optimizerD.step()
            
            # Train Generator every n_critic steps
            if i % config['n_critic'] == 0:
                netG.zero_grad()
                fake_output = netD(fake_images)
                g_loss = -torch.mean(fake_output)
                g_loss.backward()
                
                # Clip gradients
                clip_gradients(netG, config['grad_clip'])
                optimizerG.step()
                
                # Record losses
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
                
                # Log to tensorboard
                writer.add_scalar('Loss/Generator', g_loss.item(), epoch * len(dataloader) + i)
                writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch * len(dataloader) + i)
                writer.add_scalar('Loss/Gradient_Penalty', gradient_penalty.item(), epoch * len(dataloader) + i)
            
            # Print training progress with more informative metrics
            if i % 100 == 0:
                # Calculate Wasserstein distance estimate
                wasserstein_dist = real_loss.item() - fake_loss.item()
                
                print(f'Epoch [{epoch}/{config["num_epochs"]}] Batch [{i}/{len(dataloader)}] '
                      f'D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}, '
                      f'GP: {gradient_penalty.item():.4f}, WD: {wasserstein_dist:.4f}, '
                      f'LR: {current_lr:.6f}')
        
        # Update learning rate
        if schedulerG is not None:
            if isinstance(schedulerG, ReduceLROnPlateau):
                schedulerG.step(g_losses[-1] if g_losses else 1.0)
            else:
                schedulerG.step()
        
        if schedulerD is not None:
            if isinstance(schedulerD, ReduceLROnPlateau):
                schedulerD.step(d_losses[-1] if d_losses else 1.0)
            else:
                schedulerD.step()
        
        # Get current learning rate
        current_lr = optimizerG.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
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
            
            # Save optimizer states
            torch.save(optimizerG.state_dict(), 'checkpoints/vitgan/optimizerG.pth')
            torch.save(optimizerD.state_dict(), 'checkpoints/vitgan/optimizerD.pth')
            
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
        model = VisionTransformerGenerator(config['latent_size'], img_size=config['image_size']).to(device)
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