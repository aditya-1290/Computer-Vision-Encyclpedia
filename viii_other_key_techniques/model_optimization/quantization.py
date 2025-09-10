"""
Quantization: Reducing model size and inference time by using lower precision.

Implementation uses PyTorch quantization for post-training quantization.

Theory:
- Quantization: Map floating-point weights to integers.
- Benefits: Smaller models, faster inference, lower power consumption.

Math: Affine quantization: r = S(q - Z), where S is scale, Z is zero-point.

Reference:
- PyTorch Quantization Docs
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18

def quantize_model(model, dummy_input):
    """
    Post-training quantization.
    """
    model.eval()
    # Fuse layers
    model.fuse_model()
    # Specify quantization config
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    # Prepare for quantization
    torch.quantization.prepare(model, inplace=True)
    # Calibrate with dummy data
    with torch.no_grad():
        model(dummy_input)
    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    return model

if __name__ == "__main__":
    model = resnet18(pretrained=True)
    dummy_input = torch.randn(1, 3, 224, 224)
    quantized_model = quantize_model(model, dummy_input)
    print("Model quantized successfully.")
    with torch.no_grad():
        output = quantized_model(dummy_input)
    print(f"Quantized output shape: {output.shape}")
