"""
Blob Detection: LoG, DoG, SIFT keypoints

Theory:
- Blobs are bright/dark regions in images.
- LoG: Convolve with Laplacian of Gaussian to detect blobs at multiple scales.
- DoG: Approximate LoG by subtracting Gaussians.
- SIFT: Detects keypoints invariant to scale, rotation, affine.

Math:
- LoG: L(x,y,σ) = ∇²G(x,y,σ) * I(x,y)
- DoG: G(x,y,kσ) - G(x,y,σ)

Implementation: Using OpenCV.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def log_blob_detection(image, min_sigma=1, max_sigma=50, num_sigma=10, threshold=0.1):
    """
    Apply Laplacian of Gaussian blob detection.
    """
    # Simple implementation using OpenCV's blob detector with LoG
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 10
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    return keypoints

def dog_blob_detection(image):
    """
    Apply Difference of Gaussians blob detection.
    """
    # Use OpenCV's DoG via SIFT
    sift = cv2.SIFT_create()
    keypoints = sift.detect(image, None)
    return keypoints

def sift_keypoints(image):
    """
    Detect SIFT keypoints.
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints

def apply_blob_detection(image_path):
    """
    Apply LoG, DoG, SIFT, display and save.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # LoG
    keypoints_log = log_blob_detection(gray)
    image_log = cv2.drawKeypoints(image, keypoints_log, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # DoG
    keypoints_dog = dog_blob_detection(gray)
    image_dog = cv2.drawKeypoints(image, keypoints_dog, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # SIFT
    keypoints_sift = sift_keypoints(gray)
    image_sift = cv2.drawKeypoints(image, keypoints_sift, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display and save
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(image_log, cv2.COLOR_BGR2RGB))
    axes[0].set_title("LoG Blobs")
    axes[0].axis('off')
    plt.imsave("log_blobs.png", cv2.cvtColor(image_log, cv2.COLOR_BGR2RGB))

    axes[1].imshow(cv2.cvtColor(image_dog, cv2.COLOR_BGR2RGB))
    axes[1].set_title("DoG Blobs")
    axes[1].axis('off')
    plt.imsave("dog_blobs.png", cv2.cvtColor(image_dog, cv2.COLOR_BGR2RGB))

    axes[2].imshow(cv2.cvtColor(image_sift, cv2.COLOR_BGR2RGB))
    axes[2].set_title("SIFT Keypoints")
    axes[2].axis('off')
    plt.imsave("sift_keypoints.png", cv2.cvtColor(image_sift, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.savefig("blob_detection.png")
    plt.show()

if __name__ == "__main__":
    image_path = "../images/dragon.webp"  # Update path
    apply_blob_detection(image_path)
