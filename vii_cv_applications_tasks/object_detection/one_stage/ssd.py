"""
SSD (Single Shot MultiBox Detector) for Object Detection

Theory:
- SSD discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location.
- Predicts both the bounding box offsets and the confidences for object categories in a single forward pass.
- Uses multiple feature maps at different scales for detection.

Math:
- MultiBox Loss: Combines localization loss (Smooth L1) and confidence loss (Softmax).
- Default boxes: Predefined anchor boxes for each feature map cell.

Implementation: Simplified PyTorch SSD with VGG backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SSD(nn.Module):
    def __init__(self, num_classes=21):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features

        # Additional SSD layers omitted for brevity
        self.loc = nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1)  # 4 boxes per location
        self.conf = nn.Conv2d(512, 4 * num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.features(x)
        loc = self.loc(x)
        conf = self.conf(x)

        loc = loc.permute(0, 2, 3, 1).contiguous()
        conf = conf.permute(0, 2, 3, 1).contiguous()

        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        return loc, conf

if __name__ == "__main__":
    model = SSD(num_classes=21)
    x = torch.randn(1, 3, 300, 300)
    loc, conf = model(x)
    print(f"Localization shape: {loc.shape}")  # [1, num_boxes, 4]
    print(f"Confidence shape: {conf.shape}")  # [1, num_boxes, num_classes]
