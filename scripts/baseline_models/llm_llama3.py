# Created by jing at 28.02.25
import os
import torch
import argparse
import wandb
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, AutoProcessor

from scripts import config
from scripts.utils import file_utils
from scripts.utils.image_processing import get_image_descriptions, process_test_image

# Configuration
MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
DATASET_PATH = config.raw_patterns


def setup_device():
    """Setup device configuration."""
    parser = argparse.ArgumentParser(description='Evaluate LLaVA on Gestalt Principles')
    parser.add_argument('--device_id', type=int, default=None, help='GPU device ID to use')
    args = parser.parse_args()

    device = f"cuda:{args.device_id}" if args.device_id is not None else "cpu"
    torch_dtype = torch.float16 if "cuda" in device else torch.float32

    return device, torch_dtype, args


def setup_model(device, args):
    """Load model and processor."""
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        cache_dir=config.llm_path,
        device_map={"": args.device_id} if args.device_id is not None else None
    ).to("cuda" if torch.cuda.is_available() else "cpu").eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME, cache_dir=config.llm_path
    )

    return model, processor


def process_principle_pattern(principle_path, pattern, model, processor, device, torch_dtype):
    """Process a single pattern within a principle."""
    results = []

    # Construct paths for all relevant directories
    paths = {
        "train_pos": os.path.join(principle_path, "train", pattern, "positive"),
        "train_neg": os.path.join(principle_path, "train", pattern, "negative"),
        "test_pos": os.path.join(principle_path, "test", pattern, "positive"),
        "test_neg": os.path.join(principle_path, "test", pattern, "negative")
    }

    # Get training descriptions
    pos_descriptions, num_train_pos = get_image_descriptions(paths["train_pos"], model, processor, device, torch_dtype)
    neg_descriptions, num_train_neg = get_image_descriptions(paths["train_neg"], model, processor, device, torch_dtype)

    # Count test images
    num_test_pos = file_utils.count_images(paths["test_pos"])
    num_test_neg = file_utils.count_images(paths["test_neg"])

    # Skip if insufficient training data
    if not pos_descriptions or not neg_descriptions:
        print(f"⚠️ Skipping pattern {pattern} - missing training data")
        return results

    # Create context prompt
    context_prompt = (
            f"POSITIVE characteristics ({num_train_pos} examples):\n- " + "\n- ".join(pos_descriptions) +
            f"\n\nNEGATIVE characteristics ({num_train_neg} examples):\n- " + "\n- ".join(neg_descriptions) +
            "\n\nAnalyze this new image. Answer strictly with 'positive' or 'negative'."
    )

    # Pattern statistics
    pattern_stats = {
        'correct': 0, 'total': 0,
        'true_positives': 0, 'true_negatives': 0
    }

    # Process test images
    for label, test_path in [("positive", paths["test_pos"]), ("negative", paths["test_neg"])]:
        if not os.path.exists(test_path):
            continue

        for img_file in tqdm(sorted(os.listdir(test_path)), desc=f"Testing {pattern} {label}", leave=False):
            if not file_utils.is_png_file(img_file):
                continue

            image_path = os.path.join(test_path, img_file)
            result = process_test_image(image_path, context_prompt, label, model, processor, device, torch_dtype)
            results.append(result)

            if result['correct']:
                pattern_stats['correct'] += 1
                pattern_stats[f'true_{label}s'] += 1
            pattern_stats['total'] += 1

    # Skip logging if no test samples
    if pattern_stats['total'] == 0:
        print(f"⚠️ Skipping pattern {pattern} - no test samples found")
        return results

    # Compute metrics
    acc = pattern_stats['correct'] / pattern_stats['total']
    tpr = pattern_stats['true_positives'] / num_test_pos if num_test_pos > 0 else 0
    tnr = pattern_stats['true_negatives'] / num_test_neg if num_test_neg > 0 else 0

    # Log results
    principle_name = os.path.basename(principle_path)
    wandb.log({
        f"{principle_name}/{pattern}/accuracy": acc,
        f"{principle_name}/{pattern}/tpr": tpr,
        f"{principle_name}/{pattern}/tnr": tnr,
        "principle": principle_name,
        "pattern": pattern
    })

    return results


def main():
    device, torch_dtype, args = setup_device()
    model, processor = setup_model(device, args)

    # Initialize WandB
    wandb.init(
        project="Gestalt-Benchmark",
        config={"model": MODEL_NAME, "device": device, "dataset_path": DATASET_PATH}
    )

    all_results = []
    for principle in sorted(os.listdir(DATASET_PATH)):
        principle_path = os.path.join(DATASET_PATH, principle)
        if not os.path.isdir(principle_path):
            continue

        print(f"\nEvaluating principle: {principle}")

        train_dir = os.path.join(principle_path, "train")
        if not os.path.exists(train_dir):
            continue

        patterns = [p for p in sorted(os.listdir(train_dir)) if os.path.isdir(os.path.join(train_dir, p))]
        for pattern in patterns:
            pattern_results = process_principle_pattern(principle_path, pattern, model, processor, device, torch_dtype)
            all_results.extend(pattern_results)

    # Final logging
    total_correct = sum(1 for res in all_results if res["correct"])
    total_samples = len(all_results)
    overall_acc = total_correct / total_samples if total_samples > 0 else 0

    print(f"Final Overall Accuracy: {overall_acc:.2%}")
    wandb.log({"overall_accuracy": overall_acc, "total_samples": total_samples})
    wandb.finish()


if __name__ == "__main__":
    main()
