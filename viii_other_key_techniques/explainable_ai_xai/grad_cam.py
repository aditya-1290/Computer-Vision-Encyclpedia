"""
Grad-CAM: Gradient-weighted Class Activation Mapping for model interpretability.

Implementation uses PyTorch for Grad-CAM visualization.

Theory:
- Grad-CAM: Compute gradients of target class w.r.t. feature maps.
- Generate heatmap by weighting feature maps with gradients.

Math: Grad-CAM = ReLU(sum(alpha_k * A_k)), where alpha_k = mean(gradients_k)

Reference:
- Selvaraju et al., Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, ICCV 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.feature_maps = None

        self.target_layer.register_forward_hook(self.save_feature_maps)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, target_class].backward()

        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        return cam.squeeze().detach().numpy()

if __name__ == "__main__":
    model = resnet18(pretrained=True)
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)
    dummy_input = torch.randn(1, 3, 224, 224)
    heatmap = grad_cam.generate_heatmap(dummy_input, target_class=0)
    print(f"Heatmap shape: {heatmap.shape}")
