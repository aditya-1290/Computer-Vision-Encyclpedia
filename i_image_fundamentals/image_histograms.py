"""
Image Histograms: Compute & Plot Histograms, Histogram Equalization

Theory:
- Histogram: Distribution of pixel intensities.
- Useful for understanding image contrast and brightness.
- Histogram Equalization: Enhances contrast by spreading out intensity values.

Math:
- Histogram: Count of pixels per intensity level.
- Equalization: CDF (Cumulative Distribution Function) used to map old intensities to new ones.

Implementation: Using OpenCV and matplotlib.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_histogram(image, channel=0):
    """
    Compute histogram for a single channel.
    """
    hist = cv2.calcHist([image], [channel], None, [256], [0,256])
    return hist

def plot_histogram(hist, title="Histogram"):
    """
    Plot histogram using matplotlib.
    """
    plt.plot(hist)
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])
    plt.show()

def histogram_equalization(image):
    """
    Apply histogram equalization to enhance contrast.
    Works on grayscale images.
    """
    if len(image.shape) == 3:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    equalized = cv2.equalizeHist(gray)
    return equalized

if __name__ == "__main__":
    # Example usage
    image_path = "path/to/your/image.jpg"  # Update this
    image = cv2.imread(image_path)
    if image is not None:
        hist = compute_histogram(image, channel=0)
        plot_histogram(hist, "Original Image Histogram")

        equalized = histogram_equalization(image)
        hist_eq = compute_histogram(equalized, channel=0)
        plot_histogram(hist_eq, "Equalized Image Histogram")

        # Show images
        cv2.imshow("Original", image)
        cv2.imshow("Equalized", equalized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
