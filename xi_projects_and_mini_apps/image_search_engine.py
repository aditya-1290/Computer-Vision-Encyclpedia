"""
Image Search Engine: Retrieve similar images using feature descriptors.

Implementation uses SIFT features and FLANN matcher for similarity search.

Theory:
- Extract features from query image.
- Compare with database features using nearest neighbors.
- Rank images by similarity score.

Math: Similarity = 1 / (1 + distance), where distance is L2 norm of feature differences.

Reference:
- Lowe, Distinctive Image Features from Scale-Invariant Keypoints, IJCV 2004
"""

import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

class ImageSearchEngine:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.flann = cv2.FlannBasedMatcher()
        self.database_features = []
        self.database_images = []

    def add_image(self, image, image_id):
        """
        Add image to database.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is not None:
            self.database_features.append(descriptors)
            self.database_images.append(image_id)

    def search(self, query_image, k=5):
        """
        Search for similar images.
        """
        gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None:
            return []

        matches = []
        for i, db_descriptors in enumerate(self.database_features):
            if db_descriptors is not None:
                match = self.flann.knnMatch(descriptors, db_descriptors, k=2)
                good_matches = [m for m, n in match if m.distance < 0.7 * n.distance]
                matches.append((len(good_matches), self.database_images[i]))

        matches.sort(reverse=True)
        return matches[:k]

if __name__ == "__main__":
    engine = ImageSearchEngine()
    # Assume images are loaded
    # engine.add_image(image1, 'img1')
    # results = engine.search(query_image)
    print("Image search engine initialized.")
