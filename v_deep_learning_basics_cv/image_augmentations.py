"""
Image Augmentations: Geometric & photometric transforms with Albumentations

Theory:
- Augmentations increase dataset diversity, prevent overfitting.
- Geometric: Rotation, scaling, flipping.
- Photometric: Brightness, contrast, color changes.

Math: Affine transformations for geometric, pixel-wise operations for photometric.

Implementation: Using Albumentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt

def create_augmentation_pipeline():
    """
    Create augmentation pipeline.
    """
    transform = A.Compose([
        A.Rotate(limit=30, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return transform

def apply_augmentations(image_path, num_augmentations=5):
    """
    Apply augmentations and display/save.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = create_augmentation_pipeline()

    augmented_images = []
    for i in range(num_augmentations):
        augmented = transform(image=image)['image']
        augmented_images.append(augmented.permute(1, 2, 0).numpy())  # To HWC

    # Display
    fig, axes = plt.subplots(1, num_augmentations, figsize=(15, 5))
    for i, aug_img in enumerate(augmented_images):
        axes[i].imshow(aug_img)
        axes[i].set_title(f"Aug {i+1}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig("augmentations.png")
    plt.show()

if __name__ == "__main__":
    image_path = "../images/17_dragon.jpg"  # Update path
    apply_augmentations(image_path)
