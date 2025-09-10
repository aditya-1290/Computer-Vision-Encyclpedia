"""
RetinaNet: One-stage object detector with Focal Loss to address class imbalance.

Implementation uses PyTorch and torchvision models.

Reference:
- Lin et al., Focal Loss for Dense Object Detection, ICCV 2017
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead, RetinaNetClassificationHead, RetinaNetRegressionHead
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class RetinaNetModel(nn.Module):
    def __init__(self, num_classes=91, pretrained=True):
        super(RetinaNetModel, self).__init__()
        # Use ResNet50 backbone with FPN
        backbone = resnet_fpn_backbone('resnet50', pretrained=pretrained)
        # Create RetinaNet model
        self.model = RetinaNet(backbone, num_classes=num_classes)

    def forward(self, images, targets=None):
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict]): ground-truth boxes and labels (optional, for training)
        Returns:
            result (list[Dict] or dict[Tensor]): during training, returns losses dict; during eval, returns detections
        """
        return self.model(images, targets)

if __name__ == "__main__":
    # Minimal test
    model = RetinaNetModel(num_classes=10, pretrained=False)
    model.eval()
    # Create dummy input: list of 3 images, each 3x224x224
    dummy_images = [torch.randn(3, 224, 224) for _ in range(3)]
    with torch.no_grad():
        outputs = model(dummy_images)
    print("RetinaNet output keys:", outputs.keys() if isinstance(outputs, dict) else "List of outputs")
