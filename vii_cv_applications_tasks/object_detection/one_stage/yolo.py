"""
YOLO (You Only Look Once) v1 for Object Detection

Theory:
- YOLO divides the image into an SxS grid.
- Each grid cell predicts bounding boxes and class probabilities.
- Single network predicts all bounding boxes and class probabilities simultaneously.
- Fast and efficient for real-time detection.

Math:
- Loss function combines classification, localization, and confidence losses.
- Bounding box parameterization includes center coordinates, width, height.

Implementation: Simplified PyTorch YOLO v1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 192, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(192, 128, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            # Additional conv layers omitted for brevity
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * self.S * self.S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * (self.C + self.B * 5))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(-1, self.S, self.S, self.C + self.B * 5)
        return x

if __name__ == "__main__":
    model = YOLOv1()
    x = torch.randn(1, 3, 448, 448)
    output = model(x)
    print(f"Output shape: {output.shape}")  # [1, 7, 7, 30]
