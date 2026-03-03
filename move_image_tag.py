#!/usr/bin/env python3
import json
import re

def move_image_tag_to_front(content):
    """
    Move <image> tag from the end to the beginning of the content.
    """
    # Remove <image> tag from anywhere in the text
    content_without_image = content.replace('<image>', '').strip()

    # Add <image> tag at the beginning
    return '<image>\n' + content_without_image

def process_json_file(input_file, output_file):
    """
    Process the JSON file and move all <image> tags to the front of user prompts.
    """
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Processing {len(data)} entries...")
    modified_count = 0

    for entry in data:
        if 'messages' in entry:
            for message in entry['messages']:
                # Only process user messages
                if message.get('role') == 'user' and 'content' in message:
                    original_content = message['content']

                    # Check if <image> tag exists
                    if '<image>' in original_content:
                        # Move <image> to the front
                        message['content'] = move_image_tag_to_front(original_content)
                        modified_count += 1

    print(f"Modified {modified_count} user messages.")
    print(f"Writing to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Done!")

if __name__ == '__main__':
    input_file = '/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/data/cold_start_9k/train_all.json'
    output_file = '/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/data/cold_start_9k/train_all.json'

    # Create backup first
    import shutil
    backup_file = input_file + '.backup'
    print(f"Creating backup at {backup_file}...")
    shutil.copy(input_file, backup_file)

    process_json_file(input_file, output_file)
