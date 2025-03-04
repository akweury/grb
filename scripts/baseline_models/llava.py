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


# # Load LLaVA model
# def load_llava_model(device):
#     model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
#     processor = LlavaProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
#
#     model.save_pretrained(config.cache_model_path)
#     processor.save_pretrained(config.cache_model_path)
#
#     return model.to(device), processor

def load_llava_model(device):
    model_name = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"  # or whatever new checkpoint
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = LlavaProcessor.from_pretrained(model_name)

    # Ensure we have a valid patch_size
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "patch_size"):
        if processor.image_processor.patch_size is None:
            processor.image_processor.patch_size = 14  # typical for CLIP ViT-L/14

    # Optionally ensure vision tower is set
    # if model.config.vision_tower is None:
    #     model.config.vision_tower = "openai/clip-vit-large-patch14"
    if not hasattr(model.config, "vision_config") or model.config.vision_config is None:
        from transformers import CLIPVisionConfig
        model.config.vision_config = CLIPVisionConfig.from_pretrained("openai/clip-vit-large-patch14")
    if not hasattr(model.config.vision_config, "patch_size") or model.config.vision_config.patch_size is None:
        model.config.vision_config.patch_size = 14

    return model.to(device), processor


def load_images(image_dir, num_samples=5):
    image_paths = sorted(Path(image_dir).glob("*.png"))[:num_samples]
    return [Image.open(img_path).convert("RGB").resize((224, 224)) for img_path in image_paths]


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
    """
    Multi-turn approach: We feed each image individually, accumulating
    conversation context so the model "remembers" what it has seen so far.
    """

    # 1) Start with a system-style message explaining the overall goal
    conversation_context = (
        f"You are an AI reasoning about visual patterns based on Gestalt principles.\n"
        f"Principle: {principle}\n\n"
        f"We have a set of images labeled Positive and a set labeled Negative.\n"
        f"You will see each image one by one.\n"
        f"Describe each image, note any pattern features, and keep track of insights.\n"
        f"After seeing all images, we will derive the logic that differentiates Positive from Negative."
    )

    # 2) Function to do a single turn (pass one image + conversation so far)
    def single_turn(model_input_text, image):
        # Combine context + new user prompt
        full_prompt = conversation_context + "\n\nUser: " + model_input_text
        print(f"prompt: {full_prompt}")
        print(f"image: {image}")
        # Process with LLaVA
        inputs = processor(
            text=full_prompt,
            images=[image],  # pass as list
            return_tensors="pt"
        ).to(device)

        outputs = model.generate(**inputs)
        answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    # 3) Loop over all Positive images
    for i, img in enumerate(train_positive, start=1):
        user_prompt = (
            f"This is Positive image #{i}. "
            "Describe what you see that might relate to the Gestalt principle."
        )
        answer = single_turn(user_prompt, img)

        # Append the model's response to conversation (Assistant role)
        conversation_context += f"\nUser: {user_prompt}\nAssistant: {answer}"

    # 4) Loop over all Negative images
    for i, img in enumerate(train_negative, start=1):
        user_prompt = (
            f"This is Negative image #{i}. "
            "Describe what you see that might differ from the Positive examples."
        )
        answer = single_turn(user_prompt, img)

        conversation_context += f"\nUser: {user_prompt}\nAssistant: {answer}"

    # 5) Final question: "What is the core logic that differentiates Positive vs. Negative?"
    final_prompt = (
        "Now we have seen all the Positive and Negative examples. "
        "Please state the logic/rule that distinguishes them. "
        "Focus on the Gestalt principle of "
        f"{principle}."
    )
    final_answer = single_turn(final_prompt, train_negative[-1])  # or pass a dummy image if needed
    conversation_context += f"\nUser: {final_prompt}\nAssistant: {final_answer}"

    print(f"\n=== Final Inferred Rule for {principle} ===\n{final_answer}\n")
    return final_answer


# def infer_logic_rules(model, processor, train_positive, train_negative, device, principle):
#     prompt = generate_reasoning_prompt(principle)
#     print(f"prompt: {prompt}")
#     print(f"train_positive: {train_positive}, train_negative: {train_negative}")
#     inputs = processor(text=prompt, images=train_positive + train_negative, return_tensors="pt").to(device)
#     print(f"logic input:{inputs}")
#     print(inputs["pixel_values"].shape)  # Should be (batch_size, 3, H, W)
#     outputs = model.generate(**inputs)
#     logic_rules = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print(f"Inferred rules for {principle}: {logic_rules}")
#     return logic_rules


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
