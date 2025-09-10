"""
EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

Theory:
- EfficientNet uses compound scaling to uniformly scale network width, depth, and resolution.
- Achieves better accuracy-efficiency trade-off than scaling individual dimensions.
- Uses MBConv (Mobile Inverted Bottleneck Convolution) blocks with squeeze-and-excitation.

Math:
- Compound Scaling: depth = α^φ, width = β^φ, resolution = γ^φ
  where φ is the compound coefficient, α, β, γ are scaling factors.
- MBConv: Expansion -> Depthwise Conv -> Squeeze-Excitation -> Projection

Implementation: PyTorch implementation of EfficientNet-B0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super(SqueezeExcitation, self).__init__()
        se_channels = max(1, int(in_channels * se_ratio))
        self.fc1 = nn.Conv2d(in_channels, se_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(se_channels, in_channels, kernel_size=1)

    def forward(self, x):
        se = F.adaptive_avg_pool2d(x, 1)
        se = self.fc1(se)
        se = F.relu(se)
        se = self.fc2(se)
        se = torch.sigmoid(se)
        return x * se

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super(MBConv, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        expanded_channels = in_channels * expand_ratio

        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False)
            self.expand_bn = nn.BatchNorm2d(expanded_channels)

        self.depthwise_conv = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel_size,
                                        stride=stride, padding=kernel_size//2, groups=expanded_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)

        self.se = SqueezeExcitation(expanded_channels, se_ratio)

        self.project_conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)

        self.use_residual = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        identity = x

        if self.expand_ratio != 1:
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            x = F.relu6(x)

        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = F.relu6(x)

        x = self.se(x)

        x = self.project_conv(x)
        x = self.project_bn(x)

        if self.use_residual:
            x += identity

        return x

class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, depth_mult=1.0):
        super(EfficientNet, self).__init__()
        self.width_mult = width_mult
        self.depth_mult = depth_mult

        # Base configuration for EfficientNet-B0
        base_channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        base_channels = [ceil(c * width_mult) for c in base_channels]

        base_layers = [1, 2, 2, 3, 3, 4, 1]
        base_layers = [ceil(l * depth_mult) for l in base_layers]

        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels[0]),
            nn.ReLU6()
        )

        self.blocks = nn.ModuleList()
        in_channels = base_channels[0]
        for i, (out_channels, repeats, kernel_size, stride) in enumerate(zip(
            base_channels[1:-1], base_layers, [3, 3, 5, 3, 5, 5, 3], [1, 2, 2, 2, 1, 2, 1]
        )):
            for j in range(repeats):
                if j == 0:
                    self.blocks.append(MBConv(in_channels, out_channels, kernel_size, stride, expand_ratio=1 if i == 0 else 6))
                else:
                    self.blocks.append(MBConv(in_channels, out_channels, kernel_size, 1, expand_ratio=6))
                in_channels = out_channels

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels[-1], kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channels[-1]),
            nn.ReLU6()
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(base_channels[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def efficientnet_b0(num_classes=1000):
    return EfficientNet(num_classes, width_mult=1.0, depth_mult=1.0)

if __name__ == "__main__":
    # Test with dummy input
    model = efficientnet_b0(num_classes=10)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be [1, 10]
