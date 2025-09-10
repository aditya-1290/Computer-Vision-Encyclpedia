"""
Clustering Segmentation: K-Means, Mean-Shift for Color Segmentation

Theory:
- Clustering groups similar pixels into segments.
- K-Means: Partitions data into K clusters by minimizing within-cluster variance.
- Mean-Shift: Finds modes in feature space, no need to specify K.

Math:
- K-Means: Objective: min ∑ ||x_i - μ_k||² for x_i in cluster k
- Mean-Shift: Iteratively shift points towards density maxima.

Implementation: Using scikit-learn for K-Means, scikit-image for Mean-Shift.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import segmentation

def kmeans_segmentation(image, K=3):
    """
    Apply K-Means clustering for color segmentation.
    """
    # Reshape image to 2D array of pixels
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # K-Means
    kmeans = KMeans(n_clusters=K, random_state=0)
    labels = kmeans.fit_predict(pixel_values)

    # Reshape back to image
    centers = np.uint8(kmeans.cluster_centers_)
    segmented = centers[labels.flatten()]
    segmented = segmented.reshape(image.shape)

    return segmented

def meanshift_segmentation(image, bandwidth=20):
    """
    Apply Mean-Shift clustering for segmentation.
    """
    # Convert to LAB for better color segmentation
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    segmented = segmentation.mean_shift(lab, bandwidth=bandwidth)
    segmented = cv2.cvtColor(segmented.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return segmented

def apply_clustering(image_path):
    """
    Apply clustering methods, display and save.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    # K-Means
    kmeans_seg = kmeans_segmentation(image, K=3)

    # Mean-Shift
    meanshift_seg = meanshift_segmentation(image)

    # Display and save
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis('off')
    plt.imsave("original_image.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    axes[1].imshow(cv2.cvtColor(kmeans_seg, cv2.COLOR_BGR2RGB))
    axes[1].set_title("K-Means Segmentation")
    axes[1].axis('off')
    plt.imsave("kmeans_segmentation.png", cv2.cvtColor(kmeans_seg, cv2.COLOR_BGR2RGB))

    axes[2].imshow(cv2.cvtColor(meanshift_seg, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Mean-Shift Segmentation")
    axes[2].axis('off')
    plt.imsave("meanshift_segmentation.png", cv2.cvtColor(meanshift_seg, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.savefig("clustering_segmentation.png")
    plt.show()

if __name__ == "__main__":
    image_path = "../../images/dragon.webp"  # Update path
    apply_clustering(image_path)
