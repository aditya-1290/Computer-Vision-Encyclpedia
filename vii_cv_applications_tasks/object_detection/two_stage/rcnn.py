"""
R-CNN (Region-based Convolutional Neural Networks) for Object Detection

Theory:
- R-CNN generates region proposals using selective search, then classifies each with a CNN.
- Each proposal is warped to fixed size, features extracted, then SVM for classification and regression for bounding boxes.
- Two-stage: Proposal generation + classification/regression.

Math:
- Intersection over Union (IoU): IoU = (A ∩ B) / (A ∪ B)
- Non-Maximum Suppression (NMS): Suppress overlapping boxes with low confidence.
- Bounding Box Regression: Predict offsets to refine proposals.

Implementation: Simplified PyTorch R-CNN with VGG backbone.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class RCNN(nn.Module):
    def __init__(self, num_classes=21):
        super(RCNN, self).__init__()
        # Backbone: VGG16
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        self.bbox_regressor = nn.Linear(4096, num_classes * 4)  # 4 for bbox offsets

    def forward(self, x):
        # Assume x is a batch of region proposals (e.g., 224x224)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        class_scores = self.classifier(x)
        bbox_offsets = self.bbox_regressor(x)
        return class_scores, bbox_offsets

if __name__ == "__main__":
    model = RCNN(num_classes=21)
    x = torch.randn(1, 3, 224, 224)  # Single proposal
    class_scores, bbox_offsets = model(x)
    print(f"Class scores shape: {class_scores.shape}")  # [1, 21]
    print(f"Bbox offsets shape: {bbox_offsets.shape}")  # [1, 84]
