# Created by jing at 26.02.25
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm
import argparse
import json
import wandb
from pathlib import Path
from torch.utils.data import DataLoader
from scripts import config

# Configuration
BATCH_SIZE = 32  # Reduce batch size dynamically
IMAGE_SIZE = 224  # ViT default input size
NUM_CLASSES = 2  # Positive and Negative
EPOCHS = 10
ACCUMULATION_STEPS = 2  # Gradient accumulation steps

# Initialize Weights & Biases (WandB)
wandb.init(project="ViT-Gestalt-Patterns", config={
    "batch_size": BATCH_SIZE,
    "image_size": IMAGE_SIZE,
    "num_classes": NUM_CLASSES,
    "epochs": EPOCHS
})


def get_dataloader(data_dir, batch_size=BATCH_SIZE, num_workers=2, pin_memory=False):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=pin_memory), len(dataset)


# Load Pretrained ViT Model
class ViTClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(ViTClassifier, self).__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


# Training Function
def train_vit(model, train_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    model.train()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels) / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()


# Evaluation Function
def evaluate_vit(model, test_loader, device, principle, pattern_name):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    wandb.log({f"{principle}/test_accuracy": accuracy})
    print(f"Test Accuracy for {pattern_name}: {accuracy:.2f}%")
    return accuracy


def run_vit(data_path, device):
    device = torch.device(device)
    model = ViTClassifier().to(device)

    print("Training and Evaluating ViT Model on Gestalt Patterns...")
    results = {}
    total_accuracy = []

    for principle in ["proximity", "similarity", "closure", "symmetry", "continuity"]:
        principle_path = Path(data_path) / principle
        results[principle] = {}

        pattern_folders = sorted([p for p in (principle_path / "train").iterdir() if p.is_dir()], key=lambda x: x.stem)

        for pattern_folder in pattern_folders:
            train_loader, num_train_images = get_dataloader(pattern_folder)
            wandb.log({f"{principle}/num_train_images": num_train_images})
            train_vit(model, train_loader, device)

            torch.cuda.empty_cache()

            test_folder = Path(data_path) / principle / "test" / pattern_folder.stem
            if test_folder.exists():
                test_loader, _ = get_dataloader(test_folder)
                accuracy = evaluate_vit(model, test_loader, device, principle, pattern_folder.stem)
                results[principle][pattern_folder.stem] = accuracy
                total_accuracy.append(accuracy)

                torch.cuda.empty_cache()

    avg_accuracy = sum(total_accuracy) / len(total_accuracy) if total_accuracy else 0
    wandb.log({"average_test_accuracy": avg_accuracy})
    print(f"Average Test Accuracy: {avg_accuracy:.2f}%")

    # Save results to JSON file
    results_path = Path(data_path) / "evaluation_results.json"
    with open(results_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print("Training and evaluation complete. Results saved to evaluation_results.json.")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ViT model with CUDA support.")
    parser.add_argument("--device_id", type=int, help="Specify GPU device ID. If not provided, CPU will be used.")
    args = parser.parse_args()

    device = f"cuda:{args.device_id}" if args.device_id is not None and torch.cuda.is_available() else "cpu"
    run_vit(config.raw_patterns, device)

