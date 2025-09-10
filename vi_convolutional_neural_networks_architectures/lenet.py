"""
LeNet-5: Implementation of LeNet-5

Theory:
- Pioneering CNN for digit recognition.
- Layers: Conv1, Pool1, Conv2, Pool2, FC1, FC2, Output.

Math:
- Convolution: Output = (Input - Kernel + 2*Padding) / Stride + 1
- Pooling: Max/Avg over kernel size.

Implementation: Using PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    model = LeNet5()
    print(model)
    # Dummy input
    x = torch.randn(1, 1, 32, 32)
    output = model(x)
    print(f"Output shape: {output.shape}")
