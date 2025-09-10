"""
Faster R-CNN for Object Detection

Theory:
- Faster R-CNN integrates Region Proposal Network (RPN) into the network for end-to-end training.
- RPN generates proposals from feature maps using anchor boxes and predicts objectness scores.
- Shares features between RPN and detection head for efficiency.

Math:
- Anchor Boxes: Pre-defined boxes of various scales and aspect ratios.
- RPN Loss: L = L_cls + λ L_reg, for classification and regression of anchors.
- RoI Pooling: As in Fast R-CNN.

Implementation: Simplified PyTorch Faster R-CNN with VGG backbone and RPN.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import RoIPool

class RPN(nn.Module):
    def __init__(self, in_channels, num_anchors=9):
        super(RPN, self).__init__()
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(512, num_anchors * 2, kernel_size=1)  # Objectness
        self.bbox_preds = nn.Conv2d(512, num_anchors * 4, kernel_size=1)  # Bbox offsets

    def forward(self, x):
        x = self.conv(x)
        cls_logits = self.cls_logits(x)
        bbox_preds = self.bbox_preds(x)
        return cls_logits, bbox_preds

class FasterRCNN(nn.Module):
    def __init__(self, num_classes=21, num_anchors=9):
        super(FasterRCNN, self).__init__()
        # Backbone: VGG16
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        self.rpn = RPN(512, num_anchors)
        self.roi_pool = RoIPool((7, 7), spatial_scale=1.0 / 16)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        self.bbox_regressor = nn.Linear(4096, num_classes * 4)

    def forward(self, x):
        features = self.features(x)
        rpn_cls, rpn_bbox = self.rpn(features)
        # For simplicity, assume proposals are generated externally
        # In full implementation, RPN would generate proposals
        # Here, we skip to RoI pooling
        # Assume rois are provided
        # roi_features = self.roi_pool(features, rois)
        # Then classify
        # For demo, return RPN outputs
        return rpn_cls, rpn_bbox

if __name__ == "__main__":
    model = FasterRCNN(num_classes=21)
    x = torch.randn(1, 3, 224, 224)
    rpn_cls, rpn_bbox = model(x)
    print(f"RPN cls shape: {rpn_cls.shape}")  # [1, 18, H, W] for 9 anchors * 2
    print(f"RPN bbox shape: {rpn_bbox.shape}")  # [1, 36, H, W] for 9 anchors * 4
