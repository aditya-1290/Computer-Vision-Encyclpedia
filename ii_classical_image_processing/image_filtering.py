"""
Image Filtering: Gaussian, Median, Sobel, Prewitt Filters

Theory:
- Filters modify images by convolving with kernels.
- Gaussian: Blurs image, reduces noise, used in preprocessing.
- Median: Removes salt-and-pepper noise, preserves edges.
- Sobel/Prewitt: Detect edges by computing gradients.

Math:
- Gaussian kernel: G(x,y) = (1/(2*pi*sigma^2)) * exp(- (x^2 + y^2) / (2*sigma^2))
- Sobel: Horizontal: [-1,0,1; -2,0,2; -1,0,1], Vertical: [-1,-2,-1; 0,0,0; 1,2,1]
- Prewitt: Similar but with 1s instead of 2s.

Implementation: Using OpenCV for efficiency, with some from scratch.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_blur(image, kernel_size=5, sigma=1.0):
    """
    Apply Gaussian blur.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def median_blur(image, kernel_size=5):
    """
    Apply median blur.
    """
    return cv2.medianBlur(image, kernel_size)

def sobel_filter(image):
    """
    Apply Sobel edge detection.
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    return cv2.convertScaleAbs(grad)

def prewitt_filter(image):
    """
    Apply Prewitt edge detection from scratch.
    """
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]])

    grad_x = cv2.filter2D(image, -1, kernel_x)
    grad_y = cv2.filter2D(image, -1, kernel_y)
    grad = np.sqrt(grad_x.astype(float)**2 + grad_y.astype(float)**2)
    return cv2.convertScaleAbs(grad)

def apply_filters(image_path):
    """
    Load image, apply filters, display and save.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        return

    # Apply filters
    gaussian = gaussian_blur(image)
    median = median_blur(image)
    sobel = sobel_filter(image)
    prewitt = prewitt_filter(image)

    # Display and save
    filters = [gaussian, median, sobel, prewitt]
    titles = ["Gaussian Blur", "Median Blur", "Sobel Edges", "Prewitt Edges"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, (filt, title) in enumerate(zip(filters, titles)):
        ax = axes[i//2, i%2]
        ax.imshow(filt, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
        plt.imsave(f"{title.replace(' ', '_').lower()}.png", filt, cmap='gray')

    plt.tight_layout()
    plt.savefig("all_filters.png")
    plt.show()

if __name__ == "__main__":
    image_path = "../images/ganpati_bappa.jpg"  # Update path
    apply_filters(image_path)
