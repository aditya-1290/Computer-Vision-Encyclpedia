"""
Pruning: Removing redundant weights to reduce model size and improve efficiency.

Implementation uses PyTorch pruning for unstructured pruning.

Theory:
- Pruning: Set small weights to zero, remove connections.
- Benefits: Sparsity, faster inference, smaller models.

Math: Prune by magnitude: Remove weights |w| < threshold.

Reference:
- PyTorch Pruning Docs
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision.models import resnet18

def prune_model(model, amount=0.3):
    """
    Unstructured pruning.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
    return model

def remove_pruning(model):
    """
    Remove pruning reparameterization.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.remove(module, 'weight')
    return model

if __name__ == "__main__":
    model = resnet18(pretrained=True)
    pruned_model = prune_model(model, amount=0.2)
    print("Model pruned successfully.")
    pruned_model = remove_pruning(pruned_model)
    print("Pruning reparameterization removed.")
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = pruned_model(dummy_input)
    print(f"Pruned output shape: {output.shape}")
