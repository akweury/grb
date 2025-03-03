# Created by jing at 03.03.25
import torch
import argparse
import json
import wandb
from pathlib import Path
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from scripts import config
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
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


def load_images(image_dir, num_samples=5):
    image_paths = sorted(Path(image_dir).glob("*.png"))[:num_samples]
    return [Image.open(img_path).convert("RGB").resize((336, 336)) for img_path in image_paths]


def generate_reasoning_prompt(principle):
    prompt = f"""You are an AI reasoning about visual patterns based on Gestalt principles.
    You are given positive and negative examples and must deduce the common logic that differentiates them.

    Principle: {principle}

    Positive examples:
    first half images.

    Negative examples:
    second half images.

    What logical rule differentiates the positive from the negative examples?"""
    return prompt


def infer_logic_rules(model, processor, train_positive, train_negative, device, principle):
    prompt = generate_reasoning_prompt(principle)
    print(f"prompt: {prompt}")
    print(f"train_positive: {train_positive}, train_negative: {train_negative}")
    inputs = processor(text=prompt, images=train_positive + train_negative, return_tensors="pt").to(device)
    print(f"logic input:{inputs}")
    print(inputs["pixel_values"].shape)  # Should be (batch_size, 3, H, W)
    outputs = model.generate(**inputs, max_new_tokens=1000)
    logic_rules = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Inferred rules for {principle}: {logic_rules}")
    return logic_rules


def evaluate_llm(model, processor, test_images, logic_rules, device, principle):
    model.eval()
    correct, total = 0, 0
    all_labels, all_predictions = [], []

    for image, label in test_images:
        prompt = (f"Using the following reasoning rules: {logic_rules}. Classify this image as Positive or Negative."
                  f"Only answer with positive and negative.")
        print(f"image type")
        print(type(image))
        inputs = processor(images=[image], text=prompt, return_tensors="pt").to(device)
        # text_inputs = processor(text=prompt, return_tensors="pt").to(device)

        # inputs = {"pixel_values": image_inputs["pixel_values"], "input_ids": text_inputs["input_ids"]}

        # if "input_ids" not in inputs:
        #     print("Warning: input_ids not generated correctly for image.")
        #     continue
        print(f"eval inputs:{inputs}")
        outputs = model.generate(**inputs)
        prediction = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

        predicted_label = 1 if "positive" in prediction else 0
        all_labels.append(label)
        all_predictions.append(predicted_label)

        total += 1
        correct += (predicted_label == label)

    accuracy = 100 * correct / total if total > 0 else 0
    f1 = f1_score(all_labels, all_predictions, average='macro') if total > 0 else 0

    wandb.log({f"{principle}/test_accuracy": accuracy, f"{principle}/f1_score": f1})
    print(f"({principle}) Test Accuracy: {accuracy:.2f}% | F1 Score: {f1:.4f}")
    return accuracy, f1


def run_llava(data_path, principle, batch_size, device):
    init_wandb(batch_size)

    model, processor = load_llava_model(device)
    principle_path = Path(data_path)

    pattern_folders = sorted((principle_path / "train").iterdir())
    if not pattern_folders:
        print("No pattern folders found in", principle_path)
        return

    total_accuracy, total_f1 = [], []
    results = {}

    for pattern_folder in pattern_folders:
        train_positive = load_images(pattern_folder / "positive")
        train_negative = load_images(pattern_folder / "negative")
        test_positive = load_images((principle_path / "test" / pattern_folder.name) / "positive")
        test_negative = load_images((principle_path / "test" / pattern_folder.name) / "negative")

        logic_rules = infer_logic_rules(model, processor, train_positive, train_negative, device, principle)

        test_images = [(img, 1) for img in test_positive] + [(img, 0) for img in test_negative]
        accuracy, f1 = evaluate_llm(model, processor, test_images, logic_rules, device, principle)

        results[pattern_folder.name] = {"accuracy": accuracy, "f1_score": f1, "logic_rules": logic_rules}
        total_accuracy.append(accuracy)
        total_f1.append(f1)

    avg_accuracy = sum(total_accuracy) / len(total_accuracy) if total_accuracy else 0
    avg_f1 = sum(total_f1) / len(total_f1) if total_f1 else 0

    results["average"] = {"accuracy": avg_accuracy, "f1_score": avg_f1}
    results_path = Path(data_path) / "evaluation_results.json"
    with open(results_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print("Evaluation complete. Results saved to evaluation_results.json.")
    print(f"Overall Average Accuracy: {avg_accuracy:.2f}% | Average F1 Score: {avg_f1:.4f}")
    wandb.finish()
    return avg_accuracy, avg_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLaVA on Gestalt Reasoning Benchmark.")
    parser.add_argument("--device_id", type=int, help="Specify GPU device ID. If not provided, CPU will be used.")
    args = parser.parse_args()

    device = f"cuda:{args.device_id}" if args.device_id is not None and torch.cuda.is_available() else "cpu"
    run_llava(config.raw_patterns, "proximity", 2, device)
