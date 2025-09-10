"""
Thresholding: Global, Otsu's, Adaptive Thresholding

Theory:
- Thresholding converts grayscale images to binary by comparing pixels to a threshold.
- Global: Single threshold for entire image.
- Otsu's: Automatically finds optimal threshold by minimizing intra-class variance.
- Adaptive: Computes local thresholds for each pixel based on neighborhood.

Math:
- Otsu's: Maximizes between-class variance: σ²_b = w1*w2*(μ1 - μ2)^2
- Adaptive: Threshold = mean or Gaussian-weighted mean of neighborhood minus constant.

Implementation: Using OpenCV.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def global_thresholding(image, threshold=127):
    """
    Apply global thresholding.
    """
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary

def otsu_thresholding(image):
    """
    Apply Otsu's thresholding.
    """
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def adaptive_thresholding(image, method='mean', block_size=11, C=2):
    """
    Apply adaptive thresholding.
    """
    if method == 'mean':
        binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
    elif method == 'gaussian':
        binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
    else:
        raise ValueError("Method must be 'mean' or 'gaussian'")
    return binary

def apply_thresholding(image_path):
    """
    Apply thresholding methods, display and save.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        return

    # Global
    global_thresh = global_thresholding(image)

    # Otsu
    otsu_thresh = otsu_thresholding(image)

    # Adaptive Mean
    adaptive_mean = adaptive_thresholding(image, 'mean')

    # Adaptive Gaussian
    adaptive_gauss = adaptive_thresholding(image, 'gaussian')

    # Display and save
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0,0].imshow(global_thresh, cmap='gray')
    axes[0,0].set_title("Global Thresholding")
    axes[0,0].axis('off')
    plt.imsave("global_threshold.png", global_thresh, cmap='gray')

    axes[0,1].imshow(otsu_thresh, cmap='gray')
    axes[0,1].set_title("Otsu's Thresholding")
    axes[0,1].axis('off')
    plt.imsave("otsu_threshold.png", otsu_thresh, cmap='gray')

    axes[1,0].imshow(adaptive_mean, cmap='gray')
    axes[1,0].set_title("Adaptive Mean")
    axes[1,0].axis('off')
    plt.imsave("adaptive_mean.png", adaptive_mean, cmap='gray')

    axes[1,1].imshow(adaptive_gauss, cmap='gray')
    axes[1,1].set_title("Adaptive Gaussian")
    axes[1,1].axis('off')
    plt.imsave("adaptive_gaussian.png", adaptive_gauss, cmap='gray')

    plt.tight_layout()
    plt.savefig("thresholding_methods.png")
    plt.show()

if __name__ == "__main__":
    image_path = "../images/dragon.webp"  # Update path
    apply_thresholding(image_path)
