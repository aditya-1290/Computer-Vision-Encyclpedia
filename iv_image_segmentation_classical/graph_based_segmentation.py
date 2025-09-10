"""
Graph-Based Segmentation: GrabCut Algorithm

Theory:
- GrabCut segments foreground from background using graph cuts.
- User provides initial bounding box or mask.
- Algorithm iteratively refines segmentation using color and edge information.

Math:
- Energy minimization on graph with nodes as pixels.
- Min-cut/max-flow algorithm finds optimal segmentation.

Implementation: Using OpenCV.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_grabcut(image_path, rect):
    """
    Apply GrabCut segmentation, display and save.
    rect: tuple (x, y, w, h) defining ROI
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Create mask where sure and likely foreground are 1, others 0
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    segmented = image * mask2[:, :, np.newaxis]

    # Display and save
    plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    plt.title("GrabCut Segmentation")
    plt.axis('off')
    plt.savefig("grabcut_segmentation.png")
    plt.show()

if __name__ == "__main__":
    image_path = "../../images/dragon.webp"  # Update path
    rect = (50, 50, 200, 200)  # Update ROI as needed
    apply_grabcut(image_path, rect)
