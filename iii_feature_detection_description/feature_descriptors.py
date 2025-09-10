"""
Feature Descriptors: SIFT, SURF, ORB, HOG descriptors

Theory:
- Descriptors encode local image information around keypoints.
- SIFT: Computes histograms of gradients in 4x4 cells, 8 orientations.
- SURF: Uses Haar wavelets for speed.
- ORB: Binary descriptor, fast and rotation-invariant.
- HOG: Divides region into cells, computes gradient histograms.

Math:
- SIFT: Descriptor = concatenation of histograms, normalized.
- HOG: Similar to SIFT but for larger regions.

Implementation: Using OpenCV.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def sift_descriptor(image):
    """
    Compute SIFT descriptors.
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def surf_descriptor(image):
    """
    Compute SURF descriptors.
    """
    try:
        surf = cv2.xfeatures2d.SURF_create()
        keypoints, descriptors = surf.detectAndCompute(image, None)
        return keypoints, descriptors
    except AttributeError:
        print("SURF not available, using SIFT instead.")
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return keypoints, descriptors

def orb_descriptor(image):
    """
    Compute ORB descriptors.
    """
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def hog_descriptor(image):
    """
    Compute HOG descriptor.
    """
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    # Resize image to win_size
    resized = cv2.resize(image, win_size)
    descriptor = hog.compute(resized)
    return descriptor

def apply_descriptors(image_path):
    """
    Compute descriptors, display keypoints and save.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # SIFT
    kp_sift, desc_sift = sift_descriptor(gray)
    image_sift = cv2.drawKeypoints(image, kp_sift, None, (0, 255, 0))

    # SURF
    kp_surf, desc_surf = surf_descriptor(gray)
    image_surf = cv2.drawKeypoints(image, kp_surf, None, (255, 0, 0))

    # ORB
    kp_orb, desc_orb = orb_descriptor(gray)
    image_orb = cv2.drawKeypoints(image, kp_orb, None, (0, 0, 255))

    # HOG (no keypoints, just descriptor)
    hog_desc = hog_descriptor(gray)
    print(f"HOG descriptor shape: {hog_desc.shape}")

    # Display and save
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(image_sift, cv2.COLOR_BGR2RGB))
    axes[0].set_title("SIFT Keypoints")
    axes[0].axis('off')
    plt.imsave("sift_descriptors.png", cv2.cvtColor(image_sift, cv2.COLOR_BGR2RGB))

    axes[1].imshow(cv2.cvtColor(image_surf, cv2.COLOR_BGR2RGB))
    axes[1].set_title("SURF Keypoints")
    axes[1].axis('off')
    plt.imsave("surf_descriptors.png", cv2.cvtColor(image_surf, cv2.COLOR_BGR2RGB))

    axes[2].imshow(cv2.cvtColor(image_orb, cv2.COLOR_BGR2RGB))
    axes[2].set_title("ORB Keypoints")
    axes[2].axis('off')
    plt.imsave("orb_descriptors.png", cv2.cvtColor(image_orb, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.savefig("feature_descriptors.png")
    plt.show()

if __name__ == "__main__":
    image_path = "../images/dragon.webp"  # Update path
    apply_descriptors(image_path)
