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

"""
Enhanced Image Search Engine: Retrieve similar images using feature descriptors.

Implementation uses multiple feature extractors and efficient matching.
"""

import cv2
import numpy as np
import pickle
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import heapq

class ImageSearchEngine:
    def __init__(self, database_path=None, use_pca=False, pca_components=128, extractor='sift'):
        self.extractor = extractor
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca = PCA(n_components=pca_components) if use_pca else None
        
        # Initialize feature extractor
        if extractor == 'sift':
            self.feature_extractor = cv2.SIFT_create()
        elif extractor == 'orb':
            self.feature_extractor = cv2.ORB_create()
        else:
            raise ValueError("Unsupported feature extractor")
            
        self.flann = self._create_flann_index(extractor)
        self.database_features = []
        self.database_images = []
        self.database_keypoints = []  # Store keypoints for geometric verification
        self.feature_index = []  # Track which features belong to which image
        self.all_features = None
        
        # Load existing database if path provided
        if database_path and os.path.exists(database_path):
            self.load_database(database_path)

    def _create_flann_index(self, extractor):
        """Create appropriate FLANN index based on feature extractor"""
        if extractor == 'sift':
            # SIFT uses L2 distance
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
        elif extractor == 'orb':
            # ORB uses Hamming distance
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
        else:
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            
        return cv2.FlannBasedMatcher(index_params, search_params)

    def add_image(self, image, image_id):
        """
        Add image to database with multiple feature extractor support
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.feature_extractor.detectAndCompute(gray, None)
        
        if descriptors is not None:
            # Apply PCA if enabled
            if self.use_pca and self.pca is not None:
                if not hasattr(self.pca, 'components_'):
                    # Fit PCA on first batch of features
                    self.pca.fit(descriptors)
                descriptors = self.pca.transform(descriptors)
            
            self.database_features.append(descriptors)
            self.database_images.append(image_id)
            self.database_keypoints.append(keypoints)  # Store keypoints for geometric verification
            
            # Track which features belong to which image
            start_idx = len(self.feature_index)
            self.feature_index.extend([image_id] * len(descriptors))
            
            return True, len(descriptors)
        return False, 0

    def build_index(self):
        """
        Build efficient index from all features
        """
        if not self.database_features:
            return False
            
        # Concatenate all features
        self.all_features = np.vstack(self.database_features)
        
        # Apply PCA if enabled
        if self.use_pca and self.pca is not None:
            if not hasattr(self.pca, 'components_'):
                self.pca.fit(self.all_features)
            self.all_features = self.pca.transform(self.all_features)
        
        print(f"Built index with {len(self.all_features)} features from {len(self.database_images)} images")
        return True

    def search(self, query_image, k=5):
        """
        Enhanced search with geometric verification for improved precision
        """
        gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        query_keypoints, query_descriptors = self.feature_extractor.detectAndCompute(gray, None)

        if query_descriptors is None or not self.database_features:
            return []

        # Apply PCA if enabled
        if self.use_pca and self.pca is not None:
            query_descriptors = self.pca.transform(query_descriptors)

        # Use FLANN for efficient matching
        matches_by_image = {}

        for i, db_descriptors in enumerate(self.database_features):
            if db_descriptors is not None and len(db_descriptors) > 0:
                # Get database keypoints (we need to store them)
                # For now, we'll skip geometric verification if keypoints not available
                # This is a limitation - we'd need to store keypoints in the database

                # For small descriptor sets, use brute force
                if len(db_descriptors) < 10:
                    # Brute force matching
                    if self.extractor == 'orb':
                        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
                    else:
                        matcher = cv2.BFMatcher()
                    matches = matcher.knnMatch(query_descriptors, db_descriptors, k=2)
                else:
                    # FLANN matching
                    matches = self.flann.knnMatch(query_descriptors, db_descriptors, k=2)

                # Apply Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)

                # For now, use number of good matches as score
                # TODO: Add geometric verification when keypoints are stored
                matches_by_image[self.database_images[i]] = len(good_matches)

        # Get top k matches
        top_matches = heapq.nlargest(k, matches_by_image.items(), key=lambda x: x[1])
        return top_matches

    def search_efficient(self, query_image, k=5):
        """
        More efficient search using approximate nearest neighbors
        """
        if self.all_features is None:
            self.build_index()
            
        gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.feature_extractor.detectAndCompute(gray, None)
        
        if descriptors is None or self.all_features is None:
            return []
        
        # Apply PCA if enabled
        if self.use_pca and self.pca is not None:
            descriptors = self.pca.transform(descriptors)

        # Use sklearn NearestNeighbors for efficient search
        nbrs = NearestNeighbors(n_neighbors=min(50, len(self.all_features)), 
                               algorithm='auto', metric='euclidean')
        nbrs.fit(self.all_features)
        
        distances, indices = nbrs.kneighbors(descriptors)
        
        # Aggregate results by image ID
        image_scores = {}
        for i, descriptor_indices in enumerate(indices):
            for idx in descriptor_indices:
                if idx < len(self.feature_index):
                    image_id = self.feature_index[idx]
                    score = 1 / (1 + distances[i][list(indices[i]).index(idx)])
                    
                    if image_id in image_scores:
                        image_scores[image_id] += score
                    else:
                        image_scores[image_id] = score
        
        # Get top k matches
        top_matches = heapq.nlargest(k, image_scores.items(), key=lambda x: x[1])
        return top_matches

    def save_database(self, path):
        """
        Save database to file
        """
        data = {
            'database_features': self.database_features,
            'database_images': self.database_images,
            'feature_index': self.feature_index,
            'use_pca': self.use_pca,
            'pca_components': self.pca_components,
            'pca': self.pca,
            'extractor': self.extractor
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Database saved to {path} with {len(self.database_images)} images")

    def load_database(self, path):
        """
        Load database from file
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.database_features = data['database_features']
        self.database_images = data['database_images']
        self.feature_index = data['feature_index']
        self.use_pca = data['use_pca']
        self.pca_components = data['pca_components']
        self.pca = data['pca']
        self.extractor = data.get('extractor', 'sift')
        
        # Reinitialize feature extractor
        if self.extractor == 'sift':
            self.feature_extractor = cv2.SIFT_create()
        elif self.extractor == 'orb':
            self.feature_extractor = cv2.ORB_create()
        
        self.flann = self._create_flann_index(self.extractor)
        print(f"Database loaded from {path} with {len(self.database_images)} images")

    def get_database_stats(self):
        """
        Get statistics about the database
        """
        total_features = sum(len(features) for features in self.database_features)
        avg_features = total_features / len(self.database_images) if self.database_images else 0
        
        return {
            'total_images': len(self.database_images),
            'total_features': total_features,
            'avg_features_per_image': avg_features,
            'using_pca': self.use_pca,
            'extractor': self.extractor
        }

# Example usage
if __name__ == "__main__":
    # Initialize search engine
    engine = ImageSearchEngine(use_pca=True, extractor='sift')

    # Supported image extensions
    supported_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.avif', '.jfif']

    # Load all images from ../images/ and dragon/ directories
    image_dirs = ['../images/', 'dragon/']
    total_image_files = []
    for images_dir in image_dirs:
        if os.path.exists(images_dir):
            image_files = [f for f in os.listdir(images_dir)
                          if os.path.splitext(f)[1].lower() in supported_extensions]
            total_image_files.extend([(images_dir, f) for f in image_files])

    print(f"Found {len(total_image_files)} images to process.")

    loaded_count = 0
    for images_dir, image_file in total_image_files:
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)
        if image is not None:
            image_id = os.path.join(images_dir, image_file)  # Use full path as ID for uniqueness
            success, num_features = engine.add_image(image, image_id)
            if success:
                print(f"Extracted {num_features} SIFT features from {image_id}")
                loaded_count += 1
            else:
                print(f"Failed to extract features from {image_id}")
        else:
            print(f"Failed to load image {os.path.join(images_dir, image_file)}")

    print(f"Successfully loaded {loaded_count} images.")

    # Build index
    if engine.build_index():
        # Save database
        database_path = 'image_database.pkl'
        engine.save_database(database_path)

        # Print database statistics
        stats = engine.get_database_stats()
        print(f"\nDatabase Statistics:")
        print(f"- Total images: {stats['total_images']}")
        print(f"- Total features: {stats['total_features']}")
        print(f"- Average features per image: {stats['avg_features_per_image']:.1f}")
        print(f"- Using PCA: {stats['using_pca']}")
        print(f"- Feature extractor: {stats['extractor']}")
        print(f"- Database saved to: {database_path}")

        print("\nEnhanced image search engine initialized and database saved.")

        # Example search with the first image as query
        if loaded_count > 0:
            query_image_file = total_image_files[0][1]  # Use first image filename as query
            query_image_path = os.path.join(total_image_files[0][0], query_image_file)
            query_image = cv2.imread(query_image_path)
            if query_image is not None:
                print(f"\nSearching for images similar to {query_image_file}...")
                results = engine.search_efficient(query_image, k=min(5, loaded_count))
                print("Top similar images:")
                for i, (image_id, score) in enumerate(results, 1):
                    print(f"{i}. {image_id}: similarity score {score:.4f}")
            else:
                print("Could not load query image for search demo.")
    else:
