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

class StyleTransfer:
    def __init__(self, content_image, style_image, num_steps=300, style_weight=1e6, content_weight=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = vgg19(pretrained=True).features.to(self.device).eval()
        self.content_image = self.load_image(content_image).to(self.device)
        self.style_image = self.load_image(style_image).to(self.device)
        self.num_steps = num_steps
        self.style_weight = style_weight
        self.content_weight = content_weight

    def load_image(self, image_path):
        """
        Load and preprocess image.
        """
        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)

    def get_features(self, image, layers):
        """
        Extract features from specified layers.
        """
        features = {}
        x = image
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in layers:
                features[name] = x
        return features

    def gram_matrix(self, tensor):
        """
        Compute gram matrix for style loss.
        """
        b, c, h, w = tensor.size()
        tensor = tensor.view(b * c, h * w)
        gram = torch.mm(tensor, tensor.t())
        return gram / (b * c * h * w)

    def transfer_style(self):
        """
        Perform style transfer.
        """
        content_layers = ['21']  # conv4_2
        style_layers = ['0', '5', '10', '19', '28']  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1

        content_features = self.get_features(self.content_image, content_layers)
        style_features = self.get_features(self.style_image, style_layers)

        style_grams = {layer: self.gram_matrix(style_features[layer]) for layer in style_layers}

        input_image = self.content_image.clone().requires_grad_(True)
        optimizer = optim.LBFGS([input_image])

        for step in range(self.num_steps):
            def closure():
                optimizer.zero_grad()
                input_features = self.get_features(input_image, content_layers + style_layers)

                content_loss = torch.mean((input_features['21'] - content_features['21']) ** 2)

                style_loss = 0
                for layer in style_layers:
                    input_gram = self.gram_matrix(input_features[layer])
                    style_gram = style_grams[layer]
                    style_loss += torch.mean((input_gram - style_gram) ** 2)

                total_loss = self.content_weight * content_loss + self.style_weight * style_loss
                total_loss.backward()
                return total_loss

            optimizer.step(closure)

        return input_image

if __name__ == "__main__":
    # Assume images exist
    # transfer = StyleTransfer('content.jpg', 'style.jpg')
    # result = transfer.transfer_style()
    print("Style transfer class defined.")
