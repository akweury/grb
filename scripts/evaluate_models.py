# Created by jing at 26.02.25

from scripts import config
from scripts.baseline_models import vit

# List of baseline models
baseline_models = [
    {"name": "ViT", "module": vit.run_vit}
]


def evaluate_model(model_entry, data_path):
    model_name = model_entry["name"]
    model_module = model_entry["module"]

    print(f"Evaluating {model_name}...")
    model_module(data_path)



if __name__ == "__main__":
    print("Starting model evaluations...")
    data_path = config.raw_patterns

    for model in baseline_models:
        evaluate_model(model, data_path)

    print("All model evaluations completed.")