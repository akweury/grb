# Created by jing at 26.02.25

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm
from pathlib import Path
from torch.utils.data import DataLoader

from scripts import config

# Configuration
BATCH_SIZE = 32
IMAGE_SIZE = 224  # ViT default input size
NUM_CLASSES = 2  # Positive and Negative
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloader(data_dir, batch_size=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Load Pretrained ViT Model
class ViTClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(ViTClassifier, self).__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


# Evaluation Function
def evaluate_vit(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Model Accuracy: {accuracy:.2f}%")
    return accuracy


def run_vit(data_path):
    model = ViTClassifier().to(DEVICE)

    print("Evaluating ViT Model on Gestalt Patterns...")

    for principle in ["proximity", "similarity", "closure", "symmetry", "continuity"]:
        print(f"\nEvaluating {principle} patterns:")
        principle_path = data_path / principle

        for split in ["train", "test"]:
            split_path = principle_path / split
            if not split_path.exists():
                print(f"Skipping {principle} {split} - No data found.")
                continue

            for pattern_folder in split_path.iterdir():
                if pattern_folder.is_dir():
                    print(f"\nEvaluating pattern: {pattern_folder.stem}")
                    loader = get_dataloader(pattern_folder)
                    evaluate_vit(model, loader)

    print("Evaluation complete.")

if __name__ == "__main__":
    run_vit(config.raw_patterns)
