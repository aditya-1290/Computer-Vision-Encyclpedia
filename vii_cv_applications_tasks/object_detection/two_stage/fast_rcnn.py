"""
Fast R-CNN for Object Detection

Theory:
- Fast R-CNN improves R-CNN by sharing computation: extract features from the whole image once.
- Uses Region of Interest (RoI) pooling to extract fixed-size features from proposals.
- End-to-end training with multi-task loss for classification and bounding box regression.

Math:
- RoI Pooling: Divides RoI into HxW grid, max-pools each bin to fixed size.
- Multi-task Loss: L = L_cls + λ L_bbox, where L_cls is cross-entropy, L_bbox is smooth L1.
- Smooth L1: L1 for |x| > 1, L2 for |x| < 1.

Implementation: PyTorch Fast R-CNN with VGG backbone and RoI pooling.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import RoIPool

class FastRCNN(nn.Module):
    def __init__(self, num_classes=21):
        super(FastRCNN, self).__init__()
        # Backbone: VGG16
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        self.roi_pool = RoIPool((7, 7), spatial_scale=1.0 / 16)  # Assuming stride 16

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

    def forward(self, x, rois):
        # x: image, rois: list of RoIs [batch_idx, x1, y1, x2, y2]
        features = self.features(x)
        roi_features = self.roi_pool(features, rois)
        roi_features = roi_features.view(roi_features.size(0), -1)
        class_scores = self.classifier(roi_features)
        bbox_offsets = self.bbox_regressor(roi_features)
        return class_scores, bbox_offsets

if __name__ == "__main__":
    model = FastRCNN(num_classes=21)
    x = torch.randn(1, 3, 224, 224)
    rois = torch.tensor([[0, 10, 10, 100, 100]], dtype=torch.float)  # Example RoI
    class_scores, bbox_offsets = model(x, rois)
    print(f"Class scores shape: {class_scores.shape}")  # [1, 21]
    print(f"Bbox offsets shape: {bbox_offsets.shape}")  # [1, 84]
