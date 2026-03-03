#!/usr/bin/env python3
"""
Create a subset of the dataset using only available images.
This allows training to start while remaining images download.
"""

import json
from pathlib import Path

def main():
    print("Creating subset with available images...")

    # Paths
    data_dir = Path("/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/data/cold_start_9k")
    images_dir = data_dir / "images"

    # Get available images
    available_images = {f.stem for f in images_dir.glob("*.jpg")}
    print(f"Found {len(available_images)} available images")

    # Load original data
    with open(data_dir / "train_all.json", 'r') as f:
        all_data = json.load(f)

    print(f"Original dataset: {len(all_data)} samples")

    # Filter to only samples with available images
    filtered_data = []
    for item in all_data:
        if 'images' in item and item['images']:
            # Check if all images are available
            all_available = True
            for img_path in item['images']:
                filename = img_path.split('/')[-1].replace('.jpg', '')
                if filename not in available_images:
                    all_available = False
                    break

            if all_available:
                filtered_data.append(item)

    print(f"Filtered dataset: {len(filtered_data)} samples")

    # Save filtered dataset
    output_file = data_dir / "train_available.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {output_file}")
    print(f"\nYou can now train with this subset:")
    print(f"  Use dataset name: vision_sr1_cold_available")
    print(f"\nThis dataset has {len(filtered_data)} samples with available images")

if __name__ == "__main__":
    main()
