"""
Feature Matching: Brute-Force & FLANN Matchers, Homography Estimation

Theory:
- Feature matching finds correspondences between keypoints in two images.
- Brute-Force: Compares descriptors exhaustively.
- FLANN: Fast Approximate Nearest Neighbor for large datasets.
- Homography: Transformation matrix to align matched points.

Math:
- Distance metrics (e.g., Euclidean) for matching.
- RANSAC for robust homography estimation.

Implementation: Using OpenCV.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def brute_force_matcher(desc1, desc2):
    """
    Brute-Force matcher with cross-check.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def flann_matcher(desc1, desc2):
    """
    FLANN based matcher.
    """
    index_params = dict(algorithm=1, trees=5)  # KDTree
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)
    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    return good

def find_homography(kp1, kp2, matches):
    """
    Estimate homography matrix using matched keypoints.
    """
    if len(matches) < 4:
        return None, None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H, mask

def apply_feature_matching(image_path1, image_path2):
    """
    Detect features, match, estimate homography, display and save.
    """
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print("Error: One or both images not found.")
        return

    # SIFT detector
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    # Brute-Force matching
    bf_matches = brute_force_matcher(desc1, desc2)
    img_bf = cv2.drawMatches(img1, kp1, img2, kp2, bf_matches[:20], None, flags=2)

    # FLANN matching
    flann_matches = flann_matcher(desc1, desc2)
    img_flann = cv2.drawMatches(img1, kp1, img2, kp2, flann_matches[:20], None, flags=2)

    # Homography estimation
    H, mask = find_homography(kp1, kp2, flann_matches)
    if H is not None:
        h, w = img1.shape
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, H)
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        img2_color = cv2.polylines(img2_color, [np.int32(dst)], True, (0,255,0), 3, cv2.LINE_AA)
    else:
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Display and save
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    axes[0].imshow(img_bf)
    axes[0].set_title("Brute-Force Matches")
    axes[0].axis('off')
    plt.imsave("bf_matches.png", img_bf)

    axes[1].imshow(img_flann)
    axes[1].set_title("FLANN Matches")
    axes[1].axis('off')
    plt.imsave("flann_matches.png", img_flann)

    axes[2].imshow(img2_color)
    axes[2].set_title("Homography Projection")
    axes[2].axis('off')
    plt.imsave("homography_projection.png", img2_color)

    plt.tight_layout()
    plt.savefig("feature_matching.png")
    plt.show()

if __name__ == "__main__":
    image_path1 = "../../images/ganpati_bappa.jpg"  # Update paths
    image_path2 = "../../images/ganpati_bappa2.jpg"
    apply_feature_matching(image_path1, image_path2)
