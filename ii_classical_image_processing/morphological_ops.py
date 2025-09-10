"""
Morphological Operations: Erosion, Dilation, Opening, Closing

Theory:
- Morphological operations process binary images based on shape.
- Erosion: Removes small noise, shrinks objects.
- Dilation: Expands objects, fills small holes.
- Opening: Erosion followed by dilation, removes noise.
- Closing: Dilation followed by erosion, fills gaps.

Math:
- Erosion: A pixel is 1 if all pixels in the structuring element are 1.
- Dilation: A pixel is 1 if any pixel in the structuring element is 1.

Implementation: Using OpenCV.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_morphological_ops(image_path, kernel_size=5):
    """
    Load binary image, apply morphological operations, display and save.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        return

    # Threshold to binary
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Structuring element
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply operations
    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Display and save
    ops = [binary, erosion, dilation, opening, closing]
    titles = ["Original Binary", "Erosion", "Dilation", "Opening", "Closing"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, (op, title) in enumerate(zip(ops, titles)):
        axes[i].imshow(op, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
        plt.imsave(f"{title.replace(' ', '_').lower()}.png", op, cmap='gray')

    plt.tight_layout()
    plt.savefig("morphological_operations.png")
    plt.show()

if __name__ == "__main__":
    image_path = "../images/ganpati_bappa.jpg"  # Update to a suitable binary image
    apply_morphological_ops(image_path)
