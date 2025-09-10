"""
Transfer Learning and Fine-tuning: Fine-tuning a pre-trained model on a custom dataset

Theory:
- Transfer learning: Use pre-trained weights, adapt to new task.
- Fine-tuning: Update some layers, freeze others.

Math: Backpropagation updates weights based on loss gradients.

Implementation: Using PyTorch and torchvision.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
# Assume CustomImageDataset from previous file
from custom_datasets_dataloaders import CustomImageDataset
import os

def load_pretrained_model(num_classes=10):
    """
    Load ResNet18, modify for custom classes.
    """
    model = models.resnet18(pretrained=True)
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    # Modify classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def fine_tune_model(model, dataloader, num_epochs=5, lr=0.001):
    """
    Fine-tune the model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)  # Only train fc layer

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

    return model

if __name__ == "__main__":
    # Example
    image_dir = "../images"  # Update
    if os.path.exists(image_dir):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = CustomImageDataset(image_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        model = load_pretrained_model(num_classes=10)
        fine_tune_model(model, dataloader)
        torch.save(model.state_dict(), "fine_tuned_model.pth")
    else:
        print("Image directory not found.")
