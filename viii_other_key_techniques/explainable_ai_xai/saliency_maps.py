"""
Saliency Maps: Visualize input regions that most influence the model's prediction.

Implementation uses PyTorch for gradient-based saliency.

Theory:
- Saliency: Compute gradients of output w.r.t. input.
- Highlight regions with high gradient magnitude.

Math: Saliency = |grad_output / grad_input|

Reference:
- Simonyan et al., Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps, 2014
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18
import numpy as np

class SaliencyMap:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_saliency(self, input_tensor, target_class):
        input_tensor.requires_grad_()
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, target_class].backward()
        saliency = input_tensor.grad.abs().sum(dim=1, keepdim=True)
        saliency = saliency - saliency.min()
        saliency = saliency / saliency.max()
        return saliency.squeeze().detach().numpy()

if __name__ == "__main__":
    model = resnet18(pretrained=True)
    saliency_map = SaliencyMap(model)
    dummy_input = torch.randn(1, 3, 224, 224)
    saliency = saliency_map.generate_saliency(dummy_input, target_class=0)
    print(f"Saliency shape: {saliency.shape}")
