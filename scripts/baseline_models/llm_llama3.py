# Created by jing at 28.02.25

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import os
from tqdm import tqdm

from scripts import config
def load_model():
    """Load Llama 3.2 Vision model and processor"""
    model_name = "mistral-7B-vision"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model


def preprocess_image(image_path, processor):
    """Preprocess an image for Llama 3.2 Vision"""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    return inputs


def classify_image(image_path, processor, model):
    """Classify an image as positive or negative based on Gestalt reasoning benchmark"""
    inputs = preprocess_image(image_path, processor)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    prediction = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return prediction


def evaluate_pipeline(dataset_folder, processor, model):
    """Evaluate the pipeline on a test dataset"""
    correct = 0
    total = 0

    for category in os.listdir(dataset_folder):  # Proximity, Similarity, Closure, etc.
        category_path = os.path.join(dataset_folder, category, "test")

        for pattern in os.listdir(category_path):  # e.g., 0001_red_triangle
            pattern_path = os.path.join(category_path, pattern)

            for label in ["positive", "negative"]:
                image_folder = os.path.join(pattern_path, label)
                if not os.path.exists(image_folder):
                    continue

                for image_file in tqdm(os.listdir(image_folder), desc=f"Evaluating {category}/{pattern}/{label}"):
                    image_path = os.path.join(image_folder, image_file)
                    prediction = classify_image(image_path, processor, model)

                    # Assuming Llama 3.2 outputs "positive" or "negative" as text
                    if label in prediction.lower():
                        correct += 1
                    total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return accuracy


# Load Model
processor, model = load_model()

# Run Evaluation
dataset_folder = config.raw_patterns  # Updated to match dataset structure
evaluate_pipeline(dataset_folder, processor, model)
