# Created by jing at 03.03.25
import torch
import argparse
import json
import wandb
from pathlib import Path
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from scripts import config
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import f1_score


def init_wandb(batch_size):
    wandb.init(project="LLM-Gestalt-Patterns", config={"batch_size": batch_size})


# Load LLaVA model
def load_llava_model(device):
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    model.save_pretrained(config.cache_model_path)
    processor.save_pretrained(config.cache_model_path)

    return model.to(device), processor


def get_dataloader(data_dir, batch_size, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers), dataset


def generate_reasoning_prompt(positive_examples, negative_examples, principle):
    prompt = f"""You are an AI reasoning about visual patterns based on Gestalt principles.
    You are given positive and negative examples and must deduce the common logic that differentiates them.

    Principle: {principle}

    Positive examples:
    {', '.join(positive_examples)}

    Negative examples:
    {', '.join(negative_examples)}

    What logical rule differentiates the positive from the negative examples?"""
    return prompt


def evaluate_llm(model, processor, test_loader, device, principle):
    model.eval()
    correct, total = 0, 0
    all_labels, all_predictions = [], []

    for images, labels in test_loader:
        images = images.to(device)

        for i in range(images.shape[0]):
            image = images[i].unsqueeze(0)
            inputs = processor(image, return_tensors="pt").to(device)

            prompt = f"Based on the given reasoning rule, classify this image as Positive or Negative."
            inputs["input_ids"] = processor.tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

            outputs = model.generate(**inputs, max_length=50)
            prediction = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

            predicted_label = 1 if "Positive" in prediction else 0
            all_labels.append(labels[i].item())
            all_predictions.append(predicted_label)

            total += 1
            correct += (predicted_label == labels[i].item())

    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_predictions, average='macro')
    wandb.log({f"{principle}/test_accuracy": accuracy, f"{principle}/f1_score": f1})
    print(f"({principle}) Test Accuracy: {accuracy:.2f}% | F1 Score: {f1:.4f}")
    return accuracy, f1


def run_llava(data_path, principle, batch_size, device):
    init_wandb(batch_size)

    model, processor = load_llava_model(device)
    principle_path = Path(data_path) / "train" / principle
    test_path = Path(data_path) / "test" / principle

    if not principle_path.exists() or not test_path.exists():
        print("Training or test data missing for principle:", principle)
        return

    # Load training samples
    train_loader, train_dataset = get_dataloader(principle_path, batch_size)
    positive_samples = [Path(img[0]).name for img in train_dataset.imgs if img[1] == 1][:5]
    negative_samples = [Path(img[0]).name for img in train_dataset.imgs if img[1] == 0][:5]

    reasoning_prompt = generate_reasoning_prompt(positive_samples, negative_samples, principle)
    print("Generated reasoning prompt:\n", reasoning_prompt)

    # Evaluate on test set
    test_loader, test_dataset = get_dataloader(test_path, batch_size)
    test_samples = test_dataset.imgs[:5]  # Select first 5 test samples
    test_loader = DataLoader(test_samples, batch_size=batch_size, shuffle=False)

    accuracy, f1 = evaluate_llm(model, processor, test_loader, device, principle)

    results = {"principle": principle, "accuracy": accuracy, "f1_score": f1}
    results_path = Path(data_path) / "evaluation_results.json"
    with open(results_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print("Evaluation complete. Results saved to evaluation_results.json.")
    wandb.finish()
    return accuracy, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLaVA on Gestalt Reasoning Benchmark.")
    parser.add_argument("--device_id", type=int, help="Specify GPU device ID. If not provided, CPU will be used.")
    args = parser.parse_args()

    device = f"cuda:{args.device_id}" if args.device_id is not None and torch.cuda.is_available() else "cpu"
    run_llava(config.raw_patterns, "proximity", 2, device)
