"""
Image Representation: Pixels, Channels, and Color Spaces

Theory:
- An image is a 2D array of pixels, where each pixel represents a point in the image.
- Pixels have intensity values, typically 0-255 for 8-bit images.
- Channels: Grayscale images have 1 channel; RGB images have 3 (Red, Green, Blue).
- Color Spaces:
  - RGB: Additive color model, used for display.
  - HSV: Hue (0-360), Saturation (0-1), Value (0-1). Useful for color-based segmentation.
  - LAB: L (lightness), A (green-red), B (blue-yellow). Perceptually uniform, good for color correction.

Math (Key Conversions):
- RGB to HSV: Complex formulas involving max/min of channels.
- RGB to LAB: Involves linear transformations and non-linear adjustments.

Implementation: Using OpenCV for conversions and NumPy for array handling.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_display_image(image_path):
    """
    Load an image and display its properties.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    print(f"Image shape: {image.shape}")  # (height, width, channels)
    print(f"Image dtype: {image.dtype}")

    # Display original image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image (RGB)")
    plt.show()

    return image

def convert_color_spaces(image):
    """
    Convert image to different color spaces.
    """
    # RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    print(f"HSV shape: {hsv.shape}")

    # RGB to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    print(f"LAB shape: {lab.shape}")

    # Display conversions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("RGB")
    axes[1].imshow(hsv)
    axes[1].set_title("HSV")
    axes[2].imshow(lab)
    axes[2].set_title("LAB")
    plt.show()

    return hsv, lab

if __name__ == "__main__":
    # Example usage (replace with actual image path)
    image_path = "path/to/your/image.jpg"  # Update this path
    image = load_and_display_image(image_path)
    if image is not None:
        convert_color_spaces(image)
