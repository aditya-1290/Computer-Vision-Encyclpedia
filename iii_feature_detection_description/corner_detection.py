"""
Corner Detection: Harris Corner Detector, Shi-Tomasi

Theory:
- Corners are interest points with high intensity variation in all directions.
- Harris: Computes corner response based on eigenvalues of gradient covariance matrix.
- Shi-Tomasi: Uses minimum eigenvalue for robustness.

Math:
- Harris: R = det(M) - k * (trace(M))^2, where M = [Ix^2, IxIy; IxIy, Iy^2]
- Threshold R to find corners.

Implementation: Using OpenCV.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def harris_corner_detection(image, block_size=2, ksize=3, k=0.04, threshold=0.01):
    """
    Apply Harris corner detection.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    harris = cv2.cornerHarris(gray, block_size, ksize, k)
    harris_norm = cv2.normalize(harris, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    corners = np.where(harris > threshold * harris.max())
    return corners, harris_norm

def shi_tomasi_corner_detection(image, max_corners=100, quality_level=0.01, min_distance=10):
    """
    Apply Shi-Tomasi corner detection.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)
    return corners

def apply_corner_detection(image_path):
    """
    Apply Harris and Shi-Tomasi, display and save.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    # Harris
    corners_harris, harris_response = harris_corner_detection(image)
    image_harris = image.copy()
    for corner in zip(corners_harris[1], corners_harris[0]):
        cv2.circle(image_harris, corner, 5, (0, 255, 0), -1)

    # Shi-Tomasi
    corners_shi = shi_tomasi_corner_detection(image)
    image_shi = image.copy()
    if corners_shi is not None:
        for corner in corners_shi:
            x, y = corner.ravel()
            cv2.circle(image_shi, (int(x), int(y)), 5, (255, 0, 0), -1)

    # Display and save
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(image_harris, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Harris Corners")
    axes[0].axis('off')
    plt.imsave("harris_corners.png", cv2.cvtColor(image_harris, cv2.COLOR_BGR2RGB))

    axes[1].imshow(cv2.cvtColor(image_shi, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Shi-Tomasi Corners")
    axes[1].axis('off')
    plt.imsave("shi_tomasi_corners.png", cv2.cvtColor(image_shi, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.savefig("corner_detection.png")
    plt.show()

if __name__ == "__main__":
    image_path = "../images/dragon.webp"  # Update path
    apply_corner_detection(image_path)
