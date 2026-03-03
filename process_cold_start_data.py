#!/usr/bin/env python3
"""
Process the existing cold_start_9k data file:
1. Clean prompts
2. Convert response format
3. Update image paths
4. Create categorized files
"""

import json
import re
from pathlib import Path
from tqdm import tqdm


def clean_prompt(prompt):
    """
    Remove the instruction text from the prompt.
    Keep only the actual question and <image> token.
    """
    # Remove the instruction pattern
    instruction_patterns = [
        r"You are tasked with analyzing an image/video to generate a detailed description.*?<image>",
        r"-\nYou are tasked.*?<image>",
        r"You are tasked.*?The output format should be:.*?\.<image>",
    ]

    cleaned = prompt
    for pattern in instruction_patterns:
        cleaned = re.sub(pattern, "<image>", cleaned, flags=re.DOTALL)

    # If patterns didn't match, try simpler approach
    if cleaned == prompt:
        # Look for <image> and keep the question before it
        lines = prompt.split('\n')
        question_lines = []
        for line in lines:
            if line.startswith('Question:') or line.startswith('Options:') or line.startswith('A.') or line.startswith('B.'):
                question_lines.append(line)
            elif '<image>' in line:
                question_lines.append('<image>')
                break

        if question_lines:
            cleaned = '\n'.join(question_lines)
        else:
            # Last resort: just add <image> at the end if not present
            if '<image>' not in cleaned:
                cleaned = cleaned + '\n<image>'

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
    print("Processing cold_start_9k data...")

    # Paths
    input_file = Path("/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/data/cold_start_9k/A-vision_see_think.json")
    output_dir = Path("/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/data/cold_start_9k")

    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples")

    # Process data
    type1_data = []
    type2_data = []
    type3_data = []
    all_data = []

    for idx, item in enumerate(tqdm(data)):
        try:
            messages = item['messages']

            # Extract user and assistant messages
            user_msg = None
            assistant_msg = None

            for msg in messages:
                if msg['role'] == 'user':
                    user_msg = msg
                elif msg['role'] == 'assistant':
                    assistant_msg = msg

            if not user_msg or not assistant_msg:
                continue

            # Clean and convert
            cleaned_prompt = clean_prompt(user_msg['content'])
            converted_response = convert_response(assistant_msg['content'])

            # Update image paths from "cold_start/X.jpg" to "cold_start_9k/images/X.jpg"
            updated_images = []
            if 'images' in item:
                for img_path in item['images']:
                    # Convert cold_start/X.jpg to cold_start_9k/images/X.jpg
                    img_filename = img_path.split('/')[-1]
                    updated_images.append(f"cold_start_9k/images/{img_filename}")

            # Create new entry
            new_entry = {
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

            if updated_images:
                new_entry["images"] = updated_images

            # Categorize
            category = categorize_response(converted_response)

            all_data.append(new_entry)
            if category == 'type1':
                type1_data.append(new_entry)
            elif category == 'type2':
                type2_data.append(new_entry)
            elif category == 'type3':
                type3_data.append(new_entry)

        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            continue

    # Save processed data
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
    print("\nSample processed entry:")
    if all_data:
        print(json.dumps(all_data[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
