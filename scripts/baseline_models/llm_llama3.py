# Created by jing at 28.02.25
import os
import torch
import argparse
import wandb
from PIL import Image
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

    device = torch.device(
        f"cuda:{args.device_id}" if args.device_id is not None and torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if "cuda" in str(device) else torch.float32

    return device, torch_dtype, args


def setup_model(device, args):
    """Load model and processor."""
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        cache_dir=config.llm_path,
        device_map={"": args.device_id} if args.device_id is not None else None
    ).to(device).eval()

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

    # Skip if insufficient training data
    if not pos_descriptions or not neg_descriptions:
        return results

    # Define logical pattern based on positive training images
    logic_pattern = f"Common characteristics of positive examples ({num_train_pos} samples):\n- " + "\n- ".join(
        pos_descriptions)

    # Process test images
    pattern_stats = {'correct': 0, 'total': 0}
    for label, test_path in [("positive", paths["test_pos"]), ("negative", paths["test_neg"])]:
        if not os.path.exists(test_path):
            continue

        for img_file in tqdm(sorted(os.listdir(test_path)), desc=f"Testing {pattern} {label}", leave=False):
            if not file_utils.is_png_file(img_file):
                continue

            image_path = os.path.join(test_path, img_file)

            # Provide logic pattern as context for model decision
            test_prompt = f"{logic_pattern}\n\nAnalyze the given image and determine if it follows the above pattern. Answer strictly with 'positive' or 'negative'."

            result = process_test_image(image_path, test_prompt, label, model, processor, device, torch_dtype)
            results.append(result)

            if result['correct']:
                pattern_stats['correct'] += 1
            pattern_stats['total'] += 1

    if pattern_stats['total'] == 0:
        return results

    # Compute accuracy
    acc = pattern_stats['correct'] / pattern_stats['total']

    # Log results
    print(f"Pattern: {pattern}, Accuracy: {acc:.2%}")
    wandb.log({f"{pattern}/accuracy": acc})

    return acc


def main():
    device, torch_dtype, args = setup_device()
    model, processor = setup_model(device, args)

    # Initialize WandB
    wandb.init(
        project="Gestalt-Benchmark",
        config={"model": MODEL_NAME, "device": str(device), "dataset_path": DATASET_PATH}
    )

    total_patterns = 0
    total_accuracy = 0

    for principle in sorted(os.listdir(DATASET_PATH)):
        principle_path = os.path.join(DATASET_PATH, principle)
        if not os.path.isdir(principle_path):
            continue

        train_dir = os.path.join(principle_path, "train")
        if not os.path.exists(train_dir):
            continue

        patterns = [p for p in sorted(os.listdir(train_dir)) if os.path.isdir(os.path.join(train_dir, p))]
        for pattern in patterns:
            pattern_accuracy = process_principle_pattern(principle_path, pattern, model, processor, device, torch_dtype)
            if pattern_accuracy is not None:
                total_patterns += 1
                total_accuracy += pattern_accuracy

    # Final logging
    overall_acc = total_accuracy / total_patterns if total_patterns > 0 else 0
    print(f"Final Overall Accuracy: {overall_acc:.2%}")
    wandb.log({"overall_accuracy": overall_acc})
    wandb.finish()


if __name__ == "__main__":
    main()
