"""
Neural Style Transfer: Apply artistic style to content image.

Implementation uses PyTorch for style transfer with VGG19.

Theory:
- Content loss: Preserve content features.
- Style loss: Match style features (gram matrix).
- Total loss: Weighted sum of content and style losses.

Math: Style loss = sum ||G^l - A^l||^2, where G is gram matrix of features.

Reference:
- Gatys et al., A Neural Algorithm of Artistic Style, arXiv 2015
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
import os
from pathlib import Path

class ImprovedStyleTransfer(StyleTransfer):
    def __init__(self, content_image, style_image, num_steps=300, style_weight=1e6, 
                 content_weight=1, output_size=512, content_layers=None, style_layers=None):
        super().__init__(content_image, style_image, num_steps, style_weight, content_weight)
        self.output_size = output_size
        
        # Customizable layers with defaults
        self.content_layers = content_layers or ['21']  # conv4_2
        self.style_layers = style_layers or ['0', '5', '10', '19', '28']
        
        # Add progress tracking
        self.progress_callback = None
        self.current_step = 0
        
    def load_image(self, image_path, size=None):
        """Improved image loading with customizable size"""
        size = size or self.output_size
        image = Image.open(image_path)
        
        # Preserve aspect ratio while resizing
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)
    
    def set_progress_callback(self, callback):
        """Allow external progress monitoring"""
        self.progress_callback = callback
        
    def transfer_style(self, output_path=None):
        """Enhanced style transfer with progress and saving"""
        content_features = self.get_features(self.content_image, self.content_layers)
        style_features = self.get_features(self.style_image, self.style_layers)
        style_grams = {layer: self.gram_matrix(style_features[layer]) for layer in self.style_layers}

        input_image = self.content_image.clone().requires_grad_(True)
        optimizer = optim.LBFGS([input_image])
        
        # Track best result
        best_loss = float('inf')
        best_image = None
        
        for step in range(self.num_steps):
            self.current_step = step
            
            def closure():
                optimizer.zero_grad()
                input_features = self.get_features(input_image, self.content_layers + self.style_layers)

                content_loss = torch.mean((input_features[self.content_layers[0]] - content_features[self.content_layers[0]]) ** 2)
                
                style_loss = 0
                for layer in self.style_layers:
                    input_gram = self.gram_matrix(input_features[layer])
                    style_gram = style_grams[layer]
                    style_loss += torch.mean((input_gram - style_gram) ** 2)

                total_loss = self.content_weight * content_loss + self.style_weight * style_loss
                total_loss.backward()
                
                # Track best result
                nonlocal best_loss, best_image
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_image = input_image.clone().detach()
                
                # Report progress
                if self.progress_callback:
                    self.progress_callback(step, self.num_steps, total_loss.item())
                    
                return total_loss

            optimizer.step(closure)
            
            # Early stopping if loss stabilizes
            if step > 50 and abs(closure().item() - best_loss) < 1e-6:
                break

        # Save result if path provided
        if output_path and best_image is not None:
            self.save_image(best_image, output_path)
            
        return best_image if best_image is not None else input_image
    
    def save_image(self, tensor, path):
        """Save tensor as image"""
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = transforms.ToPILImage()(image)
        image.save(path)
