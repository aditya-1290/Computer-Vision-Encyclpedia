"""
Watershed Algorithm for Separating Overlapping Objects

Theory:
- Watershed treats grayscale image as a topographic surface.
- Flooding from markers fills basins, boundaries form watershed lines.
- Useful for separating touching objects.

Math:
- Gradient magnitude used as topography.
- Markers define initial basins.

Implementation: Using OpenCV.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_watershed(image_path):
    """
    Apply watershed segmentation, display and save.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Mark the unknown region with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(image,markers)
    image[markers == -1] = [255,0,0]  # Mark boundaries in red

    # Display and save
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Watershed Segmentation")
    plt.axis('off')
    plt.savefig("watershed_segmentation.png")
    plt.show()

if __name__ == "__main__":
    image_path = "../images/dragon.webp"  # Update path
    apply_watershed(image_path)
