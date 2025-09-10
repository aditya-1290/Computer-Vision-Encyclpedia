"""
Image Classification with CNNs

Theory:
- Image classification assigns a label to an input image.
- Convolutional Neural Networks (CNNs) are effective for image classification due to spatial feature extraction.
- CNN layers include convolution, activation, pooling, and fully connected layers.

Math:
- Convolution: (I * K)(x, y) = Σ_m Σ_n I(x+m, y+n) * K(m, n)
- Activation: ReLU(x) = max(0, x)
- Pooling: Downsamples feature maps to reduce spatial size.
- Fully connected layers map extracted features to class scores.

Implementation: PyTorch CNN for image classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B, 32, 112, 112)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 64, 56, 56)
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = SimpleCNN(num_classes=10)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be [1, 10]
