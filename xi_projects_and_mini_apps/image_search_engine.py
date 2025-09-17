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
import pickle
import faiss
from sklearn.decomposition import PCA

class EnhancedImageSearchEngine(ImageSearchEngine):
    def __init__(self, database_path=None, use_pca=False, pca_components=128):
        super().__init__()
        self.database_path = database_path
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca = PCA(n_components=pca_components) if use_pca else None
        self.index = None
        self.feature_dim = None
        
        # Load existing database if path provided
        if database_path and os.path.exists(database_path):
            self.load_database(database_path)
    
    def add_image(self, image, image_id, extractor='sift'):
        """
        Enhanced image adding with multiple feature extractors
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if extractor == 'sift':
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        elif extractor == 'orb':
            orb = cv2.ORB_create()
            keypoints, descriptors = orb.detectAndCompute(gray, None)
        else:
            raise ValueError("Unsupported feature extractor")
        
        if descriptors is not None:
            # Apply PCA if enabled
            if self.use_pca and self.pca is not None:
                if not hasattr(self.pca, 'components_'):
                    # Fit PCA on first batch of features
                    self.pca.fit(descriptors)
                descriptors = self.pca.transform(descriptors)
            
            self.database_features.append(descriptors)
            self.database_images.append(image_id)
            
            # Update FAISS index
            self._update_index(descriptors, image_id)
            
            return True
        return False
    
    def _update_index(self, descriptors, image_id):
        """
        Update FAISS index with new descriptors
        """
        if self.index is None:
            # Initialize index
            self.feature_dim = descriptors.shape[1]
            self.index = faiss.IndexFlatL2(self.feature_dim)
            
            # Create ID mapping
            self.id_map = []
        
        # Add descriptors to index
        self.index.add(descriptors.astype(np.float32))
        
        # Add corresponding image IDs
        self.id_map.extend([image_id] * descriptors.shape[0])
    
    def build_index(self):
        """
        Build FAISS index from all features
        """
        if not self.database_features:
            return
        
        # Concatenate all features
        all_features = np.vstack(self.database_features)
        
        # Apply PCA if enabled
        if self.use_pca and self.pca is not None:
            if not hasattr(self.pca, 'components_'):
                self.pca.fit(all_features)
            all_features = self.pca.transform(all_features)
        
        # Create index
        self.feature_dim = all_features.shape[1]
        self.index = faiss.IndexFlatL2(self.feature_dim)
        self.index.add(all_features.astype(np.float32))
        
        # Create ID mapping
        self.id_map = []
        for i, features in enumerate(self.database_features):
            self.id_map.extend([self.database_images[i]] * features.shape[0])
    
    def search(self, query_image, k=5, extractor='sift'):
        """
        Enhanced search with FAISS for faster retrieval
        """
        gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        
        if extractor == 'sift':
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        elif extractor == 'orb':
            orb = cv2.ORB_create()
            keypoints, descriptors = orb.detectAndCompute(gray, None)
        else:
            raise ValueError("Unsupported feature extractor")
        
        if descriptors is None or self.index is None:
            return []
        
        # Apply PCA if enabled
        if self.use_pca and self.pca is not None:
            descriptors = self.pca.transform(descriptors)
        
        # Search using FAISS
        distances, indices = self.index.search(descriptors.astype(np.float32), k)
        
        # Aggregate results by image ID
        results = {}
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                image_id = self.id_map[indices[i, j]]
                score = 1 / (1 + distances[i, j])  # Convert distance to similarity
                
                if image_id in results:
                    results[image_id] += score
                else:
                    results[image_id] = score
        
        # Sort by total similarity score
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]
    
    def save_database(self, path):
        """
        Save database to file
        """
        data = {
            'database_features': self.database_features,
            'database_images': self.database_images,
            'use_pca': self.use_pca,
            'pca_components': self.pca_components,
            'pca': self.pca
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        # FAISS index needs to be saved separately
        if self.index is not None:
            faiss.write_index(self.index, path + '.index')
        
        # Save ID mapping
        if hasattr(self, 'id_map'):
            with open(path + '.idmap', 'wb') as f:
                pickle.dump(self.id_map, f)
    
    def load_database(self, path):
        """
        Load database from file
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.database_features = data['database_features']
        self.database_images = data['database_images']
        self.use_pca = data['use_pca']
        self.pca_components = data['pca_components']
        self.pca = data['pca']
        
        # Load FAISS index if exists
        index_path = path + '.index'
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.feature_dim = self.index.d
            
            # Load ID mapping
            idmap_path = path + '.idmap'
            if os.path.exists(idmap_path):
                with open(idmap_path, 'rb') as f:
                    self.id_map = pickle.load(f)
