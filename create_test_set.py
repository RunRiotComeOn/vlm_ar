#!/usr/bin/env python3
"""
Create a test set by randomly sampling 200 samples from the original datasets
that are NOT in the GRPO train and val sets.
"""

import json
import sys
import pandas as pd
from pathlib import Path
import random
from typing import List, Dict, Set
from tqdm import tqdm

# Add dataset module to path
sys.path.append(str(Path(__file__).parent / "dataset"))
from load_datasets import (
    load_clevr, load_mathvista, load_vcr, load_tqa, load_okvqa
)

# Set random seed for reproducibility
random.seed(42)

# Datasets to use (5 datasets, excluding gqa which is large)
DATASETS = {
    'clevr': load_clevr,
    'vcr': load_vcr,
    'okvqa': load_okvqa,
    'tqa': load_tqa,
    'mathvista': load_mathvista,
}

GRPO_TRAIN_PATH = 'grpo_data/train.parquet'
GRPO_VAL_PATH = 'grpo_data/val.parquet'
OUTPUT_DIR = Path('test')
OUTPUT_FILE = OUTPUT_DIR / 'test.parquet'


def get_global_id(dataset_name: str, metadata: Dict) -> str:
    """Generate global_id from dataset name and metadata (same as prepare_grpo_data.py)."""
    if dataset_name == 'okvqa':
        return f"okvqa_{metadata['question_id']}"
    elif dataset_name == 'clevr':
        return f"clevr_{metadata['question_index']}"
    elif dataset_name == 'mathvista':
        return f"mathvista_{metadata['pid']}"
    elif dataset_name == 'vcr':
        return f"vcr_{metadata['question_id']}"
    elif dataset_name == 'tqa':
        return f"tqa_{metadata['question_id']}"
    else:
        return f"{dataset_name}_unknown"


def save_image(image, output_dir: Path, global_id: str) -> str:
    """Save PIL image to disk and return path."""
    from PIL import Image

    # Create images directory
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Convert RGBA to RGB if necessary
    if image.mode == 'RGBA':
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[3])
        image = rgb_image
    elif image.mode not in ('RGB', 'L'):
        image = image.convert('RGB')

    # Save image
    image_path = images_dir / f"{global_id}.jpg"
    image.save(image_path, "JPEG")

    # Return absolute path
    return str(image_path.absolute())


def load_grpo_samples() -> Set[str]:
    """Load GRPO train and val samples and return their data_source identifiers."""
    print("Loading GRPO train and val sets...")

    train_df = pd.read_parquet(GRPO_TRAIN_PATH)
    val_df = pd.read_parquet(GRPO_VAL_PATH)

    # Use data_source as unique identifier
    used_samples = set()

    for df in [train_df, val_df]:
        used_samples.update(df['data_source'].tolist())

    print(f"Found {len(used_samples)} samples in GRPO train+val")
    return used_samples


def create_test_set(target_size: int = 200):
    """Create test set by sampling from available data."""
    print(f"\n{'='*80}")
    print("Creating Test Set")
    print(f"{'='*80}\n")

    # Load GRPO samples
    used_samples = load_grpo_samples()

    # Load all available samples from each dataset
    all_available = []
    dataset_counts = {}

    for dataset_name, load_fn in DATASETS.items():
        print(f"\n{'='*80}")
        print(f"Loading {dataset_name}...")
        print(f"{'='*80}")

        try:
            # Load full dataset from HuggingFace
            dataset = load_fn()

            available_count = 0
            for item in tqdm(dataset, desc=f"Processing {dataset_name}"):
                # Generate global_id
                global_id = get_global_id(dataset_name, item['metadata'])

                # Only include if not in GRPO sets
                if global_id not in used_samples:
                    # Prepare answer in same format
                    answer = item['answer']
                    if isinstance(answer, list):
                        gt_answer = answer
                    else:
                        gt_answer = [str(answer)]

                    # Store as raw item (we'll process images later)
                    sample = {
                        'global_id': global_id,
                        'image': item['image'],  # PIL Image
                        'question': item['question'],
                        'gt_answer': gt_answer,
                        'dataset': dataset_name,
                        'metadata': item['metadata']
                    }

                    all_available.append(sample)
                    available_count += 1

            dataset_counts[dataset_name] = available_count
            print(f"Found {available_count} available samples (not in train/val)")

        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            dataset_counts[dataset_name] = 0

    print(f"\n{'='*80}")
    print(f"Total available samples: {len(all_available)}")
    print(f"{'='*80}")
    for dataset_name, count in dataset_counts.items():
        print(f"{dataset_name}: {count}")
    print(f"{'='*80}\n")

    # Sample target_size samples
    if len(all_available) < target_size:
        print(f"Warning: Only {len(all_available)} samples available, less than target {target_size}")
        test_samples = all_available
    else:
        test_samples = random.sample(all_available, target_size)

    # Process samples and save images
    print("\nSaving images and preparing test data...")
    OUTPUT_DIR.mkdir(exist_ok=True)

    processed_samples = []
    for sample in tqdm(test_samples, desc="Processing samples"):
        try:
            # Save image
            image_path = save_image(sample['image'], OUTPUT_DIR, sample['global_id'])

            # Format prompt in chat message format (same as GRPO data)
            question_with_image = f"<image>\n{sample['question']}"

            processed_sample = {
                'data_source': sample['global_id'],
                'prompt': [{"role": "user", "content": question_with_image}],
                'images': [{"image": image_path}],
                'gt_answer': sample['gt_answer'],
                'sample_type': 'unknown',  # Test set doesn't have type labels
                'dataset': sample['dataset'],
                'reward_model': 'adaptive_reasoning'
            }

            processed_samples.append(processed_sample)
        except Exception as e:
            print(f"\nError processing {sample['global_id']}: {e}")
            continue

    # Convert to DataFrame
    test_df = pd.DataFrame(processed_samples)

    # Print distribution
    print(f"\nTest set distribution:")
    print(f"Total samples: {len(test_df)}")
    print("\nBy dataset:")
    print(test_df['dataset'].value_counts())

    # Save to parquet
    test_df.to_parquet(OUTPUT_FILE, index=False)

    print(f"\n{'='*80}")
    print(f"Test set saved to: {OUTPUT_FILE}")
    print(f"{'='*80}\n")

    # Also save a summary
    summary = {
        'total_samples': len(test_df),
        'by_dataset': test_df['dataset'].value_counts().to_dict(),
        'excluded_from_grpo': len(used_samples),
        'source_datasets': list(DATASETS.keys())
    }

    summary_path = OUTPUT_DIR / 'test_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_path}\n")

    return test_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create test set from original datasets")
    parser.add_argument(
        "--size",
        type=int,
        default=200,
        help="Number of samples to include in test set (default: 200)"
    )

    args = parser.parse_args()

    create_test_set(target_size=args.size)
