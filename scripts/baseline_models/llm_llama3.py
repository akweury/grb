# Created by jing at 28.02.25


import os
import torch
from PIL import Image
import argparse
from transformers import LlavaForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import wandb
from scripts import config

# Configuration
DATASET_PATH = config.raw_patterns
MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
# Argument parsing
parser = argparse.ArgumentParser(description='Evaluate LLaVA on Gestalt Principles')
parser.add_argument('--device_id', type=int, default=None,
                    help='GPU device ID to use (e.g., 0, 1). Omit for CPU usage')
args = parser.parse_args()

# Device configuration
DEVICE = f"cuda:{args.device_id}" if args.device_id is not None else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Initialize wandb
wandb.init(
    project="Gestalt-Benchmark",
    config={
        "model": "llava-1.5-7b-hf",
        "device": DEVICE,
        "dataset_path": config.raw_patterns
    }
)

# Load model and processor from config path
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    cache_dir=config.llm_path,  # Add this line to specify model storage location
    torch_dtype=TORCH_DTYPE,
    low_cpu_mem_usage=True,
    load_in_4bit=True if 'cuda' in DEVICE else False,
    device_map="auto" if args.device_id is not None else None
).eval()

processor = AutoProcessor.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    cache_dir=config.llm_path  # Store processor in the same location
)


def is_png_file(filename):
    """Check if file is a PNG image"""
    return filename.lower().endswith('.png')


def get_image_descriptions(folder_path):
    """Get descriptions for all PNG images in a folder"""
    descriptions = []
    if not os.path.exists(folder_path):
        return descriptions

    png_files = [f for f in sorted(os.listdir(folder_path)) if is_png_file(f)]

    for img_file in tqdm(png_files, desc=f"Processing {os.path.basename(folder_path)}"):
        image_path = os.path.join(folder_path, img_file)
        try:
            image = Image.open(image_path)
            prompt = "USER: <image>\nAnalyze the spatial relationships and grouping principles in this image.\nASSISTANT:"

            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(DEVICE, TORCH_DTYPE)

            output = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False
            )

            description = processor.decode(output[0][2:], skip_special_tokens=True)
            clean_desc = description.split("ASSISTANT: ")[-1].strip()
            descriptions.append(clean_desc)
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            descriptions.append("")
    return descriptions

def process_principle_pattern(principle_path, pattern):
    """Process a single pattern within a principle"""
    results = []

    # Path setup
    train_pos = os.path.join(principle_path, "train", pattern, "positive")
    train_neg = os.path.join(principle_path, "train", pattern, "negative")
    test_pos = os.path.join(principle_path, "test", pattern, "positive")
    test_neg = os.path.join(principle_path, "test", pattern, "negative")

    # Get training descriptions
    pos_descriptions = [d for d in get_image_descriptions(train_pos) if d]
    neg_descriptions = [d for d in get_image_descriptions(train_neg) if d]

    if not pos_descriptions or not neg_descriptions:
        print(f"Skipping pattern {pattern} due to missing training data")
        return results

    # Create context prompt
    context_prompt = (
            "From the training examples:\n"
            "POSITIVE characteristics:\n- " + "\n- ".join(pos_descriptions) + "\n\n"
                                                                              "NEGATIVE characteristics:\n- " + "\n- ".join(
        neg_descriptions) + "\n\n"
                            "For this new image, does it follow the POSITIVE pattern or NEGATIVE deviation? "
                            "Answer strictly with 'positive' or 'negative'."
    )

    # Process test images
    for label, test_path in [("positive", test_pos), ("negative", test_neg)]:
        if not os.path.exists(test_path):
            continue

        for img_file in tqdm(sorted(os.listdir(test_path)), desc=f"Testing {pattern} {label}"):
            image_path = os.path.join(test_path, img_file)
            result = process_test_image(image_path, context_prompt, expected_label=label)
            results.append(result)

    return results


def process_test_image(image_path, context_prompt, expected_label):
    """Process a single test image"""
    try:
        if not is_png_file(image_path):
            return {
                "principle": "",
                "pattern": "",
                "expected": expected_label,
                "predicted": "skip",
                "correct": False,
                "image_path": image_path
            }

        image = Image.open(image_path)
        full_prompt = f"USER: <image>\n{context_prompt}\nASSISTANT:"

        inputs = processor(
            text=full_prompt,
            images=image,
            return_tensors="pt"
        ).to(DEVICE, TORCH_DTYPE)

        output = model.generate(
            **inputs,
            max_new_tokens=15,
            do_sample=False
        )

        response = processor.decode(output[0][2:], skip_special_tokens=True)
        prediction = response.split("ASSISTANT: ")[-1].strip().lower()
        prediction = "positive" if "positive" in prediction else "negative" if "negative" in prediction else "unknown"

        return {
            "principle": os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(image_path)))),
            "pattern": os.path.basename(os.path.dirname(os.path.dirname(image_path))),
            "expected": expected_label,
            "predicted": prediction,
            "correct": prediction == expected_label,
            "image_path": image_path
        }
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return {
            "principle": "",
            "pattern": "",
            "expected": expected_label,
            "predicted": "error",
            "correct": False,
            "image_path": image_path
        }


def main():
    all_results = []

    # Iterate through gestalt principles
    for principle in sorted(os.listdir(config.raw_patterns)):
        principle_path = os.path.join(config.raw_patterns, principle)
        if not os.path.isdir(principle_path):
            continue

        print(f"\n{'=' * 40}\nEvaluating principle: {principle}\n{'=' * 40}")

        # Get all patterns from train directory
        train_dir = os.path.join(principle_path, "train")
        if not os.path.exists(train_dir):
            print(f"No train directory found for {principle}")
            continue

        patterns = [p for p in sorted(os.listdir(train_dir)) if os.path.isdir(os.path.join(train_dir, p))]

        for pattern in patterns:
            print(f"\nProcessing pattern: {pattern}")
            pattern_results = process_principle_pattern(principle_path, pattern)
            all_results.extend(pattern_results)

    # Calculate and display statistics
    if not all_results:
        print("No results collected. Check dataset paths and structure.")
        return

    # Detailed statistics
    stats = {}
    for res in all_results:
        key = (res["principle"], res["pattern"])
        if key not in stats:
            stats[key] = {"total": 0, "correct": 0}

        stats[key]["total"] += 1
        stats[key]["correct"] += int(res["correct"])

    # Log results to wandb
    accuracy_data = []
    for (principle, pattern), data in stats.items():
        acc = data["correct"] / data["total"]
        accuracy_data.append([f"{principle}/{pattern}", acc])

        # Log individual pattern accuracy
        wandb.log({
            "accuracy": acc,
            "principle": principle,
            "pattern": pattern
        })

    # Create and log line chart
    table = wandb.Table(data=accuracy_data, columns=["pattern", "accuracy"])
    wandb.log({
        "accuracy_trend": wandb.plot.line(
            table,
            "pattern",
            "accuracy",
            title="Pattern Accuracy Trend"
        )
    })

    # Calculate overall accuracy
    total_correct = sum([v["correct"] for v in stats.values()])
    total_samples = sum([v["total"] for v in stats.values()])
    overall_acc = total_correct / total_samples if total_samples > 0 else 0

    # Log final metrics
    wandb.log({
        "overall_accuracy": overall_acc,
        "total_samples": total_samples
    })

    print(f"\nFinal Accuracy: {overall_acc:.2%}")
    wandb.finish()


if __name__ == "__main__":
    main()
