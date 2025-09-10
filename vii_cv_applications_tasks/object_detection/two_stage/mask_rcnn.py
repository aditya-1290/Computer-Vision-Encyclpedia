"""
Mask R-CNN for Instance Segmentation

Theory:
- Mask R-CNN extends Faster R-CNN with a mask prediction branch for pixel-level segmentation.
- Uses RoIAlign for better alignment, and a small FCN for mask prediction.
- Multi-task: Classification, bbox regression, and mask prediction.

Math:
- RoIAlign: Bilinear interpolation for precise RoI feature extraction.
- Mask Loss: Binary cross-entropy for each class mask.
- Total Loss: L = L_cls + L_bbox + L_mask

Implementation: Simplified PyTorch Mask R-CNN with ResNet backbone and mask head.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import RoIAlign

class MaskHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MaskHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.mask_pred = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = self.deconv(x)
        masks = self.mask_pred(x)
        return masks

class MaskRCNN(nn.Module):
    def __init__(self, num_classes=21):
        super(MaskRCNN, self).__init__()
        # Backbone: ResNet-50
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.roi_align = RoIAlign((7, 7), spatial_scale=1.0 / 32, sampling_ratio=2)

        # Classification and bbox heads (simplified)
        self.classifier = nn.Linear(2048 * 7 * 7, num_classes)
        self.bbox_regressor = nn.Linear(2048 * 7 * 7, num_classes * 4)

        # Mask head
        self.mask_head = MaskHead(2048, num_classes)

    def forward(self, x, rois):
        features = self.backbone(x)
        roi_features = self.roi_align(features, rois)
        roi_features_flat = roi_features.view(roi_features.size(0), -1)

        class_scores = self.classifier(roi_features_flat)
        bbox_offsets = self.bbox_regressor(roi_features_flat)
        masks = self.mask_head(roi_features)

        return class_scores, bbox_offsets, masks

if __name__ == "__main__":
    model = MaskRCNN(num_classes=21)
    x = torch.randn(1, 3, 224, 224)
    rois = torch.tensor([[0, 10, 10, 100, 100]], dtype=torch.float)
    class_scores, bbox_offsets, masks = model(x, rois)
    print(f"Class scores shape: {class_scores.shape}")  # [1, 21]
    print(f"Bbox offsets shape: {bbox_offsets.shape}")  # [1, 84]
    print(f"Masks shape: {masks.shape}")  # [1, 21, 14, 14]
