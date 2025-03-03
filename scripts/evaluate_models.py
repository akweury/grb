# Created by jing at 26.02.25
import argparse
from scripts import config
from scripts.baseline_models import vit
import torch
import os

# List of baseline models
baseline_models = [
    {"name": "ViT", "module": vit.run_vit}
]


def evaluate_model(model_entry, principle, data_path, device):
    model_name = model_entry["name"]
    model_module = model_entry["module"]

    print(f"{principle} Evaluating {model_name} on {device}...")
    model_module(data_path, principle, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline models with CUDA support.")
    parser.add_argument("--device_id", type=int, help="Specify GPU device ID. If not provided, CPU will be used.")
    parser.add_argument("--principle", type=str, required=True, help="Specify the principle to filter data.")
    args = parser.parse_args()

    # Determine device based on device_id flag
    if args.device_id is not None and torch.cuda.is_available():
        device = f"cuda:{args.device_id}"
    else:
        device = "cpu"

    # Construct the data path based on the principle argument
    data_path = os.path.join(config.raw_patterns, args.principle)

    print(f"Starting model evaluations with data from {data_path}...")

    for model in baseline_models:
        evaluate_model(model, args.principle, data_path, device)

    print("All model evaluations completed.")
