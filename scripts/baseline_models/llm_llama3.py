# Created by jing at 28.02.25
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import os
from tqdm import tqdm

from scripts import config
def load_model():
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir=config.llm_path)
    model = AutoModelForVision2Seq.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir=config.llm_path).to(
        "cuda" if torch.cuda.is_available() else "cpu")
    return processor, model


def preprocess_image(image_path, processor):
    """Preprocess an image for model input"""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    return inputs


def extract_common_rules(train_folder, processor, model):
    """Ask the model to reason the common rules in positive images"""
    positive_images = []

    for category in os.listdir(train_folder):
        category_path = os.path.join(train_folder, category, "train")

        for pattern in os.listdir(category_path):
            pattern_path = os.path.join(category_path, pattern, "positive")
            if os.path.exists(pattern_path):
                for image_file in os.listdir(pattern_path):
                    image_path = os.path.join(pattern_path, image_file)
                    positive_images.append(preprocess_image(image_path, processor))

    if not positive_images:
        return "No positive images found."

    prompt = "Analyze the following images and describe the common patterns and rules in the positive images."
    inputs = processor(text=prompt, images=[img["pixel_values"] for img in positive_images], return_tensors="pt",
                       padding=True).to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        outputs = model.generate(**inputs)
    reasoning = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return reasoning


def classify_test_images(test_folder, processor, model, reasoning):
    """Classify test images based on learned reasoning"""
    results = {}

    for category in os.listdir(test_folder):
        category_path = os.path.join(test_folder, category, "test")

        for pattern in os.listdir(category_path):
            pattern_path = os.path.join(category_path, pattern)

            for image_file in os.listdir(pattern_path):
                image_path = os.path.join(pattern_path, image_file)
                inputs = preprocess_image(image_path, processor)

                prompt = f"Given the common reasoning: {reasoning}, classify this image as positive or negative."
                inputs = processor(text=prompt, images=inputs["pixel_values"], return_tensors="pt").to(
                    "cuda" if torch.cuda.is_available() else "cpu")

                with torch.no_grad():
                    outputs = model.generate(**inputs)
                prediction = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                results[image_path] = prediction

    return results


# Load Model
processor, model = load_model()

# Train and extract reasoning
dataset_folder = config.raw_patterns  # Updated to match dataset structure
reasoning = extract_common_rules(dataset_folder, processor, model)
print("Learned reasoning:", reasoning)

# Classify test images
classification_results = classify_test_images(dataset_folder, processor, model, reasoning)
print("Classification Results:", classification_results)
