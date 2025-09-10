"""
Image Gradients and Laplacian for Edge Detection

Theory:
- Gradients: First derivatives, indicate intensity changes (edges).
- Laplacian: Second derivative, detects edges by zero crossings.
- Used in edge detection algorithms like Canny.

Math:
- Gradient: Gx = dI/dx, Gy = dI/dy, Magnitude = sqrt(Gx^2 + Gy^2)
- Laplacian: L = d2I/dx2 + d2I/dy2

Implementation: Using OpenCV for Sobel and Laplacian.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_gradients(image):
    """
    Compute image gradients using Sobel.
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return cv2.convertScaleAbs(magnitude)

def compute_laplacian(image):
    """
    Compute Laplacian for edge detection.
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)

def apply_edge_detection(image_path):
    """
    Load image, compute gradients and Laplacian, display and save.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        return

    # Compute
    gradients = compute_gradients(image)
    laplacian = compute_laplacian(image)

    # Display and save
    edges = [gradients, laplacian]
    titles = ["Gradients (Sobel)", "Laplacian"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i, (edge, title) in enumerate(zip(edges, titles)):
        axes[i].imshow(edge, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
        plt.imsave(f"{title.replace(' ', '_').lower()}.png", edge, cmap='gray')

    plt.tight_layout()
    plt.savefig("gradients_laplacian.png")
    plt.show()

if __name__ == "__main__":
    image_path = "../images/ganpati_bappa.jpg"  # Update path
    apply_edge_detection(image_path)
