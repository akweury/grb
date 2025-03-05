# Created by jing at 26.02.25
import random
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm
import argparse
import json
import wandb
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score
from scripts import config
from sklearn.metrics import precision_score, recall_score

# Configuration
# BATCH_SIZE = 8  # Increase batch size for better GPU utilization  # Reduce batch size dynamically
IMAGE_SIZE = 224  # ViT default input size
NUM_CLASSES = 2  # Positive and Negative
EPOCHS = 10
ACCUMULATION_STEPS = 1  # Reduce accumulation steps for faster updates  # Gradient accumulation steps


def init_wandb(batch_size):
    # Initialize Weights & Biases (WandB)
    wandb.init(project="ViT-Gestalt-Patterns", config={
        "batch_size": batch_size,
        "image_size": IMAGE_SIZE,
        "num_classes": NUM_CLASSES,
        "epochs": EPOCHS
    })


def get_dataloader(data_dir, batch_size, num_workers=2, pin_memory=True, prefetch_factor=None):
    transform = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    total_images = len(dataset)

    # Randomly select 5 unique indices from the dataset
    selected_indices = random.sample(range(total_images), min(5, total_images))
    subset_dataset = Subset(dataset, selected_indices)

    return DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=pin_memory, prefetch_factor=prefetch_factor,
                      persistent_workers=(num_workers > 0)), len(subset_dataset)


# Load Pretrained ViT Model
class ViTClassifier(nn.Module):
    def save_checkpoint(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_checkpoint(self, filepath):
        if Path(filepath).exists():
            self.load_state_dict(torch.load(filepath))
            print(f"Checkpoint loaded from {filepath}")
        else:
            print("No checkpoint found, starting from scratch.")

    def __init__(self, model_name, num_classes=NUM_CLASSES):
        super(ViTClassifier, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.model.set_grad_checkpointing(True)  # Enable gradient checkpointing

    def forward(self, x):
        return self.model(x)


# Training Function
def train_vit(model, train_loader, device, checkpoint_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-5, betas=(0.9, 0.999))  # Faster convergence
    scaler = torch.cuda.amp.GradScaler()  # Ensure AMP is enabled
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
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_predictions, average='macro')
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    print(f"all_labels: {all_labels}")
    print(f"all_predictions: {all_predictions}")
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)

    wandb.log({
        f"{principle}/test_accuracy": accuracy,
        f"{principle}/f1_score": f1,
        f"{principle}/precision": precision,
        f"{principle}/recall": recall
    })

    print(
        f"({principle}) Test Accuracy for {pattern_name}: {accuracy:.2f}% | F1 Score: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    return accuracy, f1, precision, recall


def run_vit(data_path, principle, batch_size, device):
    init_wandb(batch_size)
    model_name = "vit_base_patch16_224"
    checkpoint_path = config.results / principle / f"{model_name}_checkpoint.pth"
    device = torch.device(device)
    model = ViTClassifier(model_name).to(device, memory_format=torch.channels_last)
    model.load_checkpoint(checkpoint_path)

    print(f"Training and Evaluating ViT Model on Gestalt ({principle}) Patterns...")
    results = {}
    total_accuracy = []
    total_f1_scores = []
    total_precision_scores = []
    total_recall_scores = []

    principle_path = Path(data_path)
    results[principle] = {}

    pattern_folders = sorted([p for p in (principle_path / "train").iterdir() if p.is_dir()], key=lambda x: x.stem)

    for pattern_folder in pattern_folders:
        train_loader, num_train_images = get_dataloader(pattern_folder, batch_size)
        wandb.log({f"{principle}/num_train_images": num_train_images})
        train_vit(model, train_loader, device, checkpoint_path)

        torch.cuda.empty_cache()

        test_folder = Path(data_path) / "test" / pattern_folder.stem
        if test_folder.exists():
            test_loader, _ = get_dataloader(test_folder, batch_size)
            accuracy, f1, precision, recall = evaluate_vit(model, test_loader, device, principle, pattern_folder.stem)
            results[principle][pattern_folder.stem] = {
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            }
            total_accuracy.append(accuracy)
            total_f1_scores.append(f1)
            total_precision_scores.append(precision)
            total_recall_scores.append(recall)

            torch.cuda.empty_cache()

    # Compute average scores per principle
    avg_f1_scores = sum(total_f1_scores) / len(total_f1_scores) if total_f1_scores else 0
    avg_accuracy = sum(total_accuracy) / len(total_accuracy) if total_accuracy else 0
    avg_precision = sum(total_precision_scores) / len(total_precision_scores) if total_precision_scores else 0
    avg_recall = sum(total_recall_scores) / len(total_recall_scores) if total_recall_scores else 0

    wandb.log({
        f"average_f1_scores_{principle}": avg_f1_scores,
        f"average_test_accuracy_{principle}": avg_accuracy,
        f"average_precision_{principle}": avg_precision,
        f"average_recall_{principle}": avg_recall
    })

    print(
        f"Average Metrics for {principle}:\n  - Accuracy: {avg_accuracy:.2f}%\n  - F1 Score: {avg_f1_scores:.4f}\n  - Precision: {avg_precision:.4f}\n  - Recall: {avg_recall:.4f}")

    # Save results to JSON file
    os.makedirs(config.results / principle, exist_ok=True)
    results_path = config.results / principle / f"{model_name}_evaluation_results.json"
    with open(results_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print("Training and evaluation complete. Results saved to evaluation_results.json.")
    model.save_checkpoint(checkpoint_path)
    wandb.finish()


torch.set_num_threads(torch.get_num_threads())  # Utilize all available threads efficiently
os.environ['OMP_NUM_THREADS'] = str(torch.get_num_threads())  # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = str(torch.get_num_threads())  # Limit MKL threads

torch.backends.cudnn.benchmark = True  # Optimize cuDNN for fixed image size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ViT model with CUDA support.")
    parser.add_argument("--device_id", type=int, help="Specify GPU device ID. If not provided, CPU will be used.")
    args = parser.parse_args()

    device = f"cuda:{args.device_id}" if args.device_id is not None and torch.cuda.is_available() else "cpu"
    run_vit(config.raw_patterns, "proximity", 2, device)
