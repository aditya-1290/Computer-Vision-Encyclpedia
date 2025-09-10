"""
Custom Datasets and DataLoaders: Writing Dataset classes, transforms, DataLoader

Theory:
- Dataset: Encapsulates data and labels, provides __getitem__ and __len__.
- Transforms: Apply augmentations/preprocessing to data.
- DataLoader: Batches data, shuffles, parallel loading.

Math: No specific math, but normalization: mean/std for standardization.

Implementation: Using PyTorch.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.webp', '.avif', '.jfif'))]
        # Assume labels from filename or dummy
        self.labels = [int(f.split('_')[0]) for f in self.images]  # Example: 0_cat.jpg -> 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def create_dataloader(image_dir, batch_size=32, shuffle=True):
    """
    Create DataLoader with transforms.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CustomImageDataset(image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader

if __name__ == "__main__":
    # Example usage
    image_dir = "../images"  # Update path
    if os.path.exists(image_dir):
        dataloader = create_dataloader(image_dir)
        for images, labels in dataloader:
            print(f"Batch shape: {images.shape}, Labels: {labels}")
            break
    else:
        print("Image directory not found.")
