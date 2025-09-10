"""
DeepLab v3+ for Semantic Segmentation

Theory:
- DeepLab v3+ uses atrous convolution to capture multi-scale context without losing resolution.
- Atrous Spatial Pyramid Pooling (ASPP) applies atrous convolution at multiple rates.
- Decoder combines low-level features with ASPP output for precise segmentation.
- Effective for dense prediction tasks with large receptive fields.

Math:
- Atrous Convolution: y[i] = Σ_k x[i + r*k] * w[k], where r is dilation rate.
- ASPP: Parallel atrous conv with rates 6, 12, 18, and global average pooling.
- Decoder: Bilinear upsampling and concatenation for feature fusion.

Implementation: PyTorch DeepLab v3+ with ResNet-50 backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()
        self.atrous1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.atrous6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=6, padding=6)
        self.atrous12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=12, padding=12)
        self.atrous18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=18, padding=18)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        a1 = self.atrous1(x)
        a6 = self.atrous6(x)
        a12 = self.atrous12(x)
        a18 = self.atrous18(x)
        gap = self.global_avg_pool(x)
        gap = self.conv1x1(gap)
        gap = F.interpolate(gap, size=x.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([a1, a6, a12, a18, gap], dim=1)
        out = self.out_conv(out)
        return out

class DeepLabv3Plus(nn.Module):
    def __init__(self, num_classes=21):
        super(DeepLabv3Plus, self).__init__()
        # Backbone: ResNet-50
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc

        # ASPP
        self.aspp = ASPP(2048, 256)

        # Decoder
        self.low_level_conv = nn.Conv2d(256, 48, kernel_size=1)
        self.decoder_conv1 = nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1)
        self.decoder_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # Backbone
        low_level_features = self.backbone[:5](x)  # Up to layer2
        high_level_features = self.backbone[5:](low_level_features)

        # ASPP
        aspp_out = self.aspp(high_level_features)

        # Decoder
        low_level_features = self.low_level_conv(low_level_features)
        aspp_out = F.interpolate(aspp_out, size=low_level_features.shape[2:], mode='bilinear', align_corners=False)
        decoder_input = torch.cat([aspp_out, low_level_features], dim=1)
        decoder_out = self.decoder_conv1(decoder_input)
        decoder_out = self.decoder_conv2(decoder_out)
        out = self.classifier(decoder_out)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

if __name__ == "__main__":
    model = DeepLabv3Plus(num_classes=21)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be [1, 21, 224, 224]
