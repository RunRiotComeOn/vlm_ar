#!/usr/bin/env python3
"""
Script to prepare Vision-SR1-Cold-9K dataset for LLaMA-Factory training.
Converts the dataset format to match vlm_adaptive_reasoning structure.
"""

import json
import re
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def clean_prompt(prompt):
    """
    Remove the instruction text from the prompt.
    Keep only the actual question and <image> token.
    """
    # The instruction text to remove
    instruction_pattern = r"You are tasked with analyzing an image/video to generate a detailed description.*?<image>"

    # Try to find and remove the instruction
    cleaned = re.sub(instruction_pattern, "<image>", prompt, flags=re.DOTALL)

    # If the pattern didn't match, try a simpler approach
    if cleaned == prompt:
        # Look for the last occurrence of <image> and keep everything after it
        image_token = "<image>"
        if image_token in prompt:
            # Find the last <image> token
            last_image_idx = prompt.rfind(image_token)
            # Extract the question that comes after
            after_image = prompt[last_image_idx + len(image_token):].strip()
            cleaned = image_token + after_image
        else:
            cleaned = prompt

    return cleaned.strip()

def convert_response(response):
    """
    Convert response format:
    - <description> -> <perception>
    - <think> -> <reasoning>
    - \\boxed{answer} -> <answer>answer</answer>
    """
    # Replace <description> with <perception>
    response = response.replace("<description>", "<perception>")
    response = response.replace("</description>", "</perception>")

    # Replace <think> with <reasoning>
    response = response.replace("<think>", "<reasoning>")
    response = response.replace("</think>", "</reasoning>")

    # Replace \boxed{answer} with <answer>answer</answer>
    # Handle various boxed formats: \boxed{}, \\boxed{}, $\boxed{}$, etc.
    boxed_pattern = r'\$?\\+boxed\{([^}]+)\}\$?'
    response = re.sub(boxed_pattern, r'<answer>\1</answer>', response)

    return response.strip()

def categorize_response(response):
    """
    Categorize the response based on its content:
    - type1: Only <answer>
    - type2: <perception> + <answer>
    - type3: <perception> + <reasoning> + <answer>
    """
    has_perception = '<perception>' in response
    has_reasoning = '<reasoning>' in response

    if has_perception and has_reasoning:
        return 'type3'
    elif has_perception:
        return 'type2'
    else:
        return 'type1'

def main():
    print("Loading Vision-SR1-Cold-9K dataset from HuggingFace...")

    try:
        # Load the dataset
        dataset = load_dataset("LMMs-Lab-Turtle/Vision-SR1-Cold-9K")

        # Assuming the dataset has a 'train' split
        if 'train' in dataset:
            data = dataset['train']
        else:
            # Use the first available split
            split_name = list(dataset.keys())[0]
            data = dataset[split_name]
            print(f"Using split: {split_name}")

        print(f"Loaded {len(data)} examples")

        # Create output directory
        output_dir = Path("/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/data/cold_start_9k")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create images directory
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Prepare data structures for different types
        type1_data = []
        type2_data = []
        type3_data = []
        all_data = []

        print("Processing dataset...")
        for idx, example in enumerate(tqdm(data)):
            try:
                # Get the fields (adjust based on actual dataset structure)
                # Common field names: 'conversations', 'messages', 'prompt', 'response', etc.

                # Try to detect the field names
                if 'conversations' in example:
                    conversations = example['conversations']
                    # Parse conversations format
                    user_content = None
                    assistant_content = None
                    for turn in conversations:
                        if turn.get('from') == 'human' or turn.get('role') == 'user':
                            user_content = turn.get('value') or turn.get('content')
                        elif turn.get('from') == 'gpt' or turn.get('role') == 'assistant':
                            assistant_content = turn.get('value') or turn.get('content')
                elif 'messages' in example:
                    messages = example['messages']
                    user_content = None
                    assistant_content = None
                    for msg in messages:
                        if msg['role'] == 'user':
                            user_content = msg['content']
                        elif msg['role'] == 'assistant':
                            assistant_content = msg['content']
                elif 'prompt' in example and 'response' in example:
                    user_content = example['prompt']
                    assistant_content = example['response']
                else:
                    print(f"Warning: Unknown format for example {idx}, skipping...")
                    continue

                if not user_content or not assistant_content:
                    print(f"Warning: Missing content for example {idx}, skipping...")
                    continue

                # Clean prompt and convert response
                cleaned_prompt = clean_prompt(user_content)
                converted_response = convert_response(assistant_content)

                # Get image info
                image_field = None
                if 'image' in example:
                    image_field = 'image'
                elif 'images' in example:
                    image_field = 'images'

                # Handle image
                image_path = None
                if image_field and example[image_field]:
                    # Save image
                    image_data = example[image_field]
                    if isinstance(image_data, list):
                        image_data = image_data[0]

                    # Save image to disk
                    image_filename = f"vision_sr1_{idx}.jpg"
                    image_save_path = images_dir / image_filename

                    # If image_data is a PIL Image
                    if hasattr(image_data, 'save'):
                        image_data.save(image_save_path)
                        image_path = f"cold_start_9k/images/{image_filename}"

                # Categorize and create entry
                category = categorize_response(converted_response)

                entry = {
                    "messages": [
                        {
                            "content": cleaned_prompt,
                            "role": "user"
                        },
                        {
                            "content": converted_response,
                            "role": "assistant"
                        }
                    ]
                }

                if image_path:
                    entry["images"] = [image_path]

                # Add to appropriate category
                all_data.append(entry)
                if category == 'type1':
                    type1_data.append(entry)
                elif category == 'type2':
                    type2_data.append(entry)
                elif category == 'type3':
                    type3_data.append(entry)

            except Exception as e:
                print(f"Error processing example {idx}: {e}")
                continue

        # Save the processed data
        print(f"\nSaving processed data...")
        print(f"Type 1 (answer only): {len(type1_data)} examples")
        print(f"Type 2 (perception + answer): {len(type2_data)} examples")
        print(f"Type 3 (perception + reasoning + answer): {len(type3_data)} examples")
        print(f"Total: {len(all_data)} examples")

        with open(output_dir / "train_type1.json", "w", encoding="utf-8") as f:
            json.dump(type1_data, f, indent=2, ensure_ascii=False)

        with open(output_dir / "train_type2.json", "w", encoding="utf-8") as f:
            json.dump(type2_data, f, indent=2, ensure_ascii=False)

        with open(output_dir / "train_type3.json", "w", encoding="utf-8") as f:
            json.dump(type3_data, f, indent=2, ensure_ascii=False)

        with open(output_dir / "train_all.json", "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)

        print(f"\nData saved to {output_dir}")
        print("\nNext steps:")
        print("1. Update dataset_info.json to add the new dataset entries")
        print("2. Verify the data format by checking a few examples")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTrying alternative approach to inspect dataset structure...")

        # Try to load and inspect a sample
        try:
            dataset = load_dataset("LMMs-Lab-Turtle/Vision-SR1-Cold-9K", split="train[:5]")
            print("\nDataset sample structure:")
            print(dataset[0].keys())
            print("\nFirst example:")
            for key, value in dataset[0].items():
                if key == 'image':
                    print(f"{key}: <PIL Image>")
                else:
                    print(f"{key}: {str(value)[:200]}...")
        except Exception as e2:
            print(f"Could not load sample: {e2}")

if __name__ == "__main__":
    main()
