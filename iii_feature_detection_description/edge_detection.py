"""
Edge Detection: Canny Edge Detector (from scratch & with OpenCV)

Theory:
- Edges are areas of significant intensity change.
- Canny is a multi-stage algorithm: smoothing, gradient computation, non-max suppression, double thresholding, hysteresis.

Math:
- Gradient: Magnitude = sqrt(Gx^2 + Gy^2), Direction = atan2(Gy, Gx)
- Non-max suppression: Keep local maxima in gradient direction.
- Hysteresis: Connect edges using high/low thresholds.

Implementation: From scratch using NumPy, and with OpenCV.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny_from_scratch(image, low_threshold=50, high_threshold=150):
    """
    Implement Canny edge detection from scratch.
    """
    # Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 1.4)

    # Sobel gradients
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x) * 180 / np.pi

    # Non-max suppression
    suppressed = np.zeros_like(magnitude)
    for i in range(1, magnitude.shape[0]-1):
        for j in range(1, magnitude.shape[1]-1):
            angle = direction[i, j]
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
            elif 22.5 <= angle < 67.5:
                neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
            elif 67.5 <= angle < 112.5:
                neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
            else:
                neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
            if magnitude[i, j] >= max(neighbors):
                suppressed[i, j] = magnitude[i, j]

    # Double thresholding and hysteresis (simplified)
    edges = np.zeros_like(suppressed)
    strong = suppressed > high_threshold
    weak = (suppressed >= low_threshold) & (suppressed <= high_threshold)
    edges[strong] = 255
    # Hysteresis: connect weak to strong
    for i in range(1, edges.shape[0]-1):
        for j in range(1, edges.shape[1]-1):
            if weak[i, j]:
                if np.any(edges[i-1:i+2, j-1:j+2] == 255):
                    edges[i, j] = 255

    return edges.astype(np.uint8)

def apply_canny(image_path):
    """
    Apply Canny from scratch and with OpenCV, display and save.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        return

    # From scratch
    edges_scratch = canny_from_scratch(image)

    # With OpenCV
    edges_opencv = cv2.Canny(image, 50, 150)

    # Display and save
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(edges_scratch, cmap='gray')
    axes[0].set_title("Canny from Scratch")
    axes[0].axis('off')
    plt.imsave("canny_scratch.png", edges_scratch, cmap='gray')

    axes[1].imshow(edges_opencv, cmap='gray')
    axes[1].set_title("Canny OpenCV")
    axes[1].axis('off')
    plt.imsave("canny_opencv.png", edges_opencv, cmap='gray')

    plt.tight_layout()
    plt.savefig("canny_comparison.png")
    plt.show()

if __name__ == "__main__":
    image_path = "../images/dragon.webp"  # Update path
    apply_canny(image_path)
