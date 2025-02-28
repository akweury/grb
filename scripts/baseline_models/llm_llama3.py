# Created by jing at 28.02.25


import os
import torch
from PIL import Image
import argparse
from transformers import LlavaForConditionalGeneration, AutoProcessor
from tqdm import tqdm

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

# Load model and processor
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=TORCH_DTYPE,
    low_cpu_mem_usage=True,
    load_in_4bit=True
).to(DEVICE)

processor = AutoProcessor.from_pretrained(MODEL_NAME)


# Rest of the code remains the same as previous version
def get_image_descriptions(folder_path):
    """Get descriptions for all images in a folder"""
    descriptions = []
    if not os.path.exists(folder_path):
        return descriptions

    for img_file in tqdm(sorted(os.listdir(folder_path)), desc=f"Processing {os.path.basename(folder_path)}"):
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
    for principle in sorted(os.listdir(DATASET_PATH)):
        principle_path = os.path.join(DATASET_PATH, principle)
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
            stats[key] = {"total": 0, "correct": 0, "positive_total": 0, "negative_total": 0}

        stats[key]["total"] += 1
        stats[key]["correct"] += int(res["correct"])
        if res["expected"] == "positive":
            stats[key]["positive_total"] += 1
        else:
            stats[key]["negative_total"] += 1

    # Print summary
    print("\n\nEvaluation Summary:")
    total_correct = sum([v["correct"] for v in stats.values()])
    total_samples = sum([v["total"] for v in stats.values()])
    print(f"\nOverall Accuracy: {total_correct}/{total_samples} ({total_correct / total_samples:.2%})")

    # Print per-pattern results
    print("\nDetailed Results:")
    for (principle, pattern), data in stats.items():
        acc = data["correct"] / data["total"]
        pos_acc = data.get("positive_accuracy", "N/A")
        neg_acc = data.get("negative_accuracy", "N/A")
        print(f"{principle} - {pattern}:")
        print(f"  Accuracy: {data['correct']}/{data['total']} ({acc:.2%})")
        print(f"  Positive samples: {data['positive_total']}")
        print(f"  Negative samples: {data['negative_total']}\n")


if __name__ == "__main__":
    main()
