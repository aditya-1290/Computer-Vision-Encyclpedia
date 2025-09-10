"""
Fully Convolutional Networks (FCN) for Semantic Segmentation

Theory:
- Semantic segmentation assigns a class label to each pixel in an image.
- FCN replaces fully connected layers with convolutional layers to output a segmentation map.
- Uses skip connections from earlier layers to combine low-level and high-level features.
- Upsampling is done with transposed convolutions to match input resolution.

Math:
- Convolution: Preserves spatial information for pixel-wise prediction.
- Transposed Convolution: Upsamples feature maps: output_size = (input_size - 1) * stride - 2 * padding + kernel_size
- Skip Connection: Fuses features from different scales: y = upsampled_high + low_level_features

Implementation: PyTorch FCN-8s with VGG16 backbone.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class FCN(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN, self).__init__()
        # Load VGG16 backbone
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features

        # Score layers for different scales
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool5 = nn.Conv2d(512, num_classes, kernel_size=1)

        # Upsampling layers
        self.upsample2x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        self.upsample2x_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        self.upsample8x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4)

    def forward(self, x):
        # Extract features
        pool3 = self.features[:17](x)  # Up to pool3
        pool4 = self.features[17:24](pool3)  # pool4
        pool5 = self.features[24:](pool4)  # pool5

        # Score layers
        score_pool3 = self.score_pool3(pool3)
        score_pool4 = self.score_pool4(pool4)
        score_pool5 = self.score_pool5(pool5)

        # Upsample and fuse
        up_pool5 = self.upsample2x(score_pool5)  # 2x upsample
        fuse_pool4 = up_pool5 + score_pool4  # Fuse with pool4
        up_pool4 = self.upsample2x_pool4(fuse_pool4)  # 2x upsample
        fuse_pool3 = up_pool4 + score_pool3  # Fuse with pool3
        out = self.upsample8x(fuse_pool3)  # 8x upsample to input size

        return out

if __name__ == "__main__":
    model = FCN(num_classes=21)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be [1, 21, 224, 224]
