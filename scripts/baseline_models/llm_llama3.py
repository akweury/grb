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
        return descriptions,0

    png_files = [f for f in sorted(os.listdir(folder_path)) if is_png_file(f)]
    actual_count = len(png_files)

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
    return descriptions, actual_count  # Return both descriptions and actual count

def process_principle_pattern(principle_path, pattern):
    """Process a single pattern within a principle with dynamic counts and logging"""
    results = []

    # Construct paths for all relevant directories
    train_pos_path = os.path.join(principle_path, "train", pattern, "positive")
    train_neg_path = os.path.join(principle_path, "train", pattern, "negative")
    test_pos_path = os.path.join(principle_path, "test", pattern, "positive")
    test_neg_path = os.path.join(principle_path, "test", pattern, "negative")

    # Get training descriptions and actual counts
    pos_descriptions, num_train_pos = get_image_descriptions(train_pos_path)
    neg_descriptions, num_train_neg = get_image_descriptions(train_neg_path)

    # Count test images without processing them
    num_test_pos = len([f for f in os.listdir(test_pos_path) if is_png_file(f)]) if os.path.exists(test_pos_path) else 0
    num_test_neg = len([f for f in os.listdir(test_neg_path) if is_png_file(f)]) if os.path.exists(test_neg_path) else 0
    total_test = num_test_pos + num_test_neg

    # Skip patterns with insufficient training data
    if not pos_descriptions or not neg_descriptions:
        print(f"⚠️ Skipping pattern {pattern} - missing training data (pos: {num_train_pos}, neg: {num_train_neg})")
        return results

    # Create context prompt using actual training data
    context_prompt = (
            "From these training examples:\n"
            f"POSITIVE characteristics ({num_train_pos} examples):\n- " + "\n- ".join(pos_descriptions) + "\n\n"
                                                                                                          f"NEGATIVE characteristics ({num_train_neg} examples):\n- " + "\n- ".join(
        neg_descriptions) + "\n\n"
                            "Analyze this new image. Does it follow the POSITIVE pattern or NEGATIVE deviation? "
                            "Answer strictly with 'positive' or 'negative'."
    )

    # Initialize pattern statistics
    pattern_stats = {
        'correct': 0,
        'total': 0,
        'num_train_pos': num_train_pos,
        'num_train_neg': num_train_neg,
        'num_test_pos': num_test_pos,
        'num_test_neg': num_test_neg,
        'true_positives': 0,
        'true_negatives': 0
    }

    # Process test images
    for label, test_path in [('positive', test_pos_path), ('negative', test_neg_path)]:
        if not os.path.exists(test_path):
            continue

        for img_file in tqdm(sorted(os.listdir(test_path)),
                             desc=f"Testing {pattern} {label}",
                             leave=False):
            if not is_png_file(img_file):
                continue

            image_path = os.path.join(test_path, img_file)
            result = process_test_image(image_path, context_prompt, expected_label=label)
            results.append(result)

            # Update statistics
            if result['correct']:
                pattern_stats['correct'] += 1
                if label == 'positive':
                    pattern_stats['true_positives'] += 1
                else:
                    pattern_stats['true_negatives'] += 1
            pattern_stats['total'] += 1

    # Skip logging if no test samples processed
    if pattern_stats['total'] == 0:
        print(f"⚠️ Skipping pattern {pattern} - no test samples found")
        return results

    # Calculate metrics
    acc = pattern_stats['correct'] / pattern_stats['total']
    principle_name = os.path.basename(principle_path)
    tpr = pattern_stats['true_positives'] / num_test_pos if num_test_pos > 0 else 0
    tnr = pattern_stats['true_negatives'] / num_test_neg if num_test_neg > 0 else 0

    # Terminal logging
    print(f"\n📊 Pattern: {principle_name}/{pattern}")
    print(f"  Training: {num_train_pos}+ / {num_train_neg}-")
    print(f"  Testing:  {num_test_pos}+ / {num_test_neg}-")
    print(f"  Correct:  {pattern_stats['correct']}/{pattern_stats['total']}")
    print(f"  Accuracy: {acc:.2%}")
    print(f"  TPR:      {tpr:.2%} | TNR: {tnr:.2%}")
    print("-" * 50)

    # WandB logging
    wandb.log({
        # Base metrics
        f"{principle_name}/{pattern}/accuracy": acc,
        f"{principle_name}/{pattern}/tpr": tpr,
        f"{principle_name}/{pattern}/tnr": tnr,

        # Count metrics
        f"{principle_name}/{pattern}/train_pos": num_train_pos,
        f"{principle_name}/{pattern}/train_neg": num_train_neg,
        f"{principle_name}/{pattern}/test_pos": num_test_pos,
        f"{principle_name}/{pattern}/test_neg": num_test_neg,
        f"{principle_name}/{pattern}/test_total": pattern_stats['total'],
        f"{principle_name}/{pattern}/correct": pattern_stats['correct'],

        # Context for filtering
        "principle": principle_name,
        "pattern": pattern
    })

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
            pattern_results = process_principle_pattern(principle_path, pattern)
            all_results.extend(pattern_results)

    # Final summary logging
    if all_results:
        # Calculate overall statistics
        total_correct = sum(1 for res in all_results if res["correct"])
        total_samples = len(all_results)
        overall_acc = total_correct / total_samples if total_samples > 0 else 0

        # Terminal output
        print(f"\n{'#' * 40}")
        print(f"Final Overall Accuracy: {overall_acc:.2%}")
        print(f"Total Test Samples: {total_samples}")
        print(f"{'#' * 40}")

        # W&B final log
        wandb.log({
            "overall_accuracy": overall_acc,
            "total_samples": total_samples
        })

    wandb.finish()

if __name__ == "__main__":
    main()
