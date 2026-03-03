#!/usr/bin/env python3
"""
Prepare training data for GRPO (Stage 2) training.

This script converts the classified dataset into verl's expected format
for GRPO training with adaptive reasoning.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import pandas as pd
from PIL import Image

# Add dataset module to path
sys.path.append(str(Path(__file__).parent / "dataset"))
from load_datasets import (
    load_okvqa, load_geometry3k, load_scienceqa, load_gqa,
    load_clevr, load_mathvista, load_tqa, load_ocrvqa, load_mathvision, load_chartqa
)


def load_dataset_by_name(dataset_name: str) -> List[Dict]:
    """Load dataset by name."""
    dataset_loaders = {
        'okvqa': load_okvqa,
        'geometry3k': load_geometry3k,
        'scienceqa': load_scienceqa,
        'gqa': load_gqa,
        'clevr': load_clevr,
        'mathvista': load_mathvista,
        'tqa': load_tqa,
        'ocrvqa': load_ocrvqa,
        'mathvision': load_mathvision,
        'chartqa': load_chartqa,
    }

    if dataset_name not in dataset_loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"\nLoading {dataset_name}...")
    return dataset_loaders[dataset_name]()


def get_global_id(dataset_name: str, metadata: Dict) -> str:
    """Generate global_id from dataset name and metadata.

    Unified logic matching step1_generate_response.py and type1_screening.py:
    Priority order: question_id -> pid -> problem_id -> index -> question_index -> 'unknown'
    """
    if 'question_id' in metadata:
        original_id = metadata['question_id']
    elif 'pid' in metadata:
        original_id = metadata['pid']
    elif 'problem_id' in metadata:
        original_id = metadata['problem_id']
    else:
        # Fallback to index or question_index
        original_id = metadata.get('index', metadata.get('question_index', 'unknown'))

    return f"{dataset_name}_{original_id}"


def save_image(image, output_dir: Path, global_id: str) -> str:
    """Save PIL image to disk and return path."""
    # Create images directory
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Convert RGBA to RGB if necessary (JPEG doesn't support alpha channel)
    if image.mode == 'RGBA':
        # Create a white background
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[3])  # Use alpha channel as mask
        image = rgb_image
    elif image.mode not in ('RGB', 'L'):
        # Convert other modes to RGB
        image = image.convert('RGB')

    # Save image
    image_path = images_dir / f"{global_id}.jpg"
    image.save(image_path, "JPEG")

    # Return absolute path
    return str(image_path.absolute())


def prepare_grpo_data(
    type1_ids_file: str,
    type2_ids_file: str,
    type3_ids_file: str,
    output_dir: str,
    datasets: List[str] = None,
    train_split_ratio: float = 0.95,
):
    """
    Prepare GRPO training data.

    verl expects data in parquet format with columns:
    - data_source: identifier
    - prompt: list of message dicts in chat format, e.g., [{"role": "user", "content": "..."}]
    - images: list of image paths
    - gt_answer: ground truth answer(s) as list
    - sample_type: type1/type2/type3 for analysis
    - dataset: source dataset name

    Args:
        type1_ids_file: Path to type1_ids.json
        type2_ids_file: Path to type2_ids.json
        type3_ids_file: Path to type3_ids.json
        output_dir: Output directory for GRPO data
        datasets: List of dataset names to process (None for all)
        train_split_ratio: Ratio of training data (rest is validation)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load IDs
    print("Loading type IDs...")
    with open(type1_ids_file, 'r') as f:
        type1_ids = set(json.load(f))

    with open(type2_ids_file, 'r') as f:
        type2_ids = set(json.load(f))

    with open(type3_ids_file, 'r') as f:
        type3_ids = set(json.load(f))

    # Combine all IDs (we'll train on all types)
    all_ids = type1_ids | type2_ids | type3_ids

    # Default datasets to process
    if datasets is None:
        datasets = ['okvqa', 'geometry3k', 'scienceqa', 'gqa', 'clevr', 'mathvista', 'tqa', 'ocrvqa', 'mathvision', 'chartqa']

    # Process each dataset
    all_samples = []

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name}...")
        print('='*60)

        try:
            dataset = load_dataset_by_name(dataset_name)
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            continue

        for item in tqdm(dataset, desc=f"Processing {dataset_name}"):
            global_id = get_global_id(dataset_name, item['metadata'])

            # Only include samples that were classified
            if global_id not in all_ids:
                continue

            # Save image
            try:
                image_path = save_image(item['image'], output_path, global_id)
            except Exception as e:
                print(f"Error saving image for {global_id}: {e}")
                continue

            # Determine type
            if global_id in type1_ids:
                sample_type = "type1"
            elif global_id in type2_ids:
                sample_type = "type2"
            else:
                sample_type = "type3"

            # Prepare sample
            # Handle answer being a list (e.g., from OKVQA)
            # Convert all answers to list format for consistency
            # Note: For MCQ datasets (ScienceQA, TQA, MathVista with choices),
            # item['answer'] is already the option letter (A/B/C/D) from load_datasets.py
            answer = item['answer']
            if isinstance(answer, list):
                # For OKVQA, we have multiple acceptable answers
                gt_answer = answer  # Keep as list for reward function
            else:
                # Convert single answer to list for consistency
                # For MCQ, this will be a list with a single option letter: ['A'], ['B'], etc.
                gt_answer = [str(answer)]

            # Format prompt in chat message format (verl expects list of message dicts)
            # Ensure question has <image> tag (don't duplicate if already present)
            question = item['question']
            if not question.strip().startswith("<image>"):
                question_with_image = f"<image>\n{question}"
            else:
                question_with_image = question

            sample = {
                'data_source': global_id,
                'prompt': [{"role": "user", "content": question_with_image}],  # Chat format
                'images': [{"image": image_path}],  # List of image dicts (qwen_vl_utils format)
                'reward_model': {
                    'ground_truth': gt_answer  # Ground truth answer(s) - nested for verl
                },
                'sample_type': sample_type,  # For analysis
                'dataset': dataset_name,  # Source dataset
            }

            all_samples.append(sample)

    # Convert to DataFrame
    df = pd.DataFrame(all_samples)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split into train and validation
    split_idx = int(len(df) * train_split_ratio)
    train_df = df[:split_idx]
    val_df = df[split_idx:]

    # Save as parquet
    train_file = output_path / "train.parquet"
    val_file = output_path / "val.parquet"

    train_df.to_parquet(train_file, index=False)
    val_df.to_parquet(val_file, index=False)

    # Print statistics
    print("\n" + "="*60)
    print("GRPO Data Preparation Complete")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print()
    print("Type distribution (training):")
    print(train_df['sample_type'].value_counts())
    print()
    print("Dataset distribution (training):")
    print(train_df['dataset'].value_counts())
    print()
    print(f"✓ Saved to:")
    print(f"  Train: {train_file}")
    print(f"  Val: {val_file}")

    # Save metadata
    metadata = {
        'total_samples': len(df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'train_split_ratio': train_split_ratio,
        'type_distribution': train_df['sample_type'].value_counts().to_dict(),
        'dataset_distribution': train_df['dataset'].value_counts().to_dict(),
        'train_file': str(train_file),
        'val_file': str(val_file),
    }

    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Metadata: {metadata_file}")
    print("="*60)

    return train_df, val_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare GRPO training data")
    parser.add_argument(
        "--type1_ids",
        type=str,
        default="dataset/output/type1_ids_sampled.json",
        help="Path to type1_ids.json"
    )
    parser.add_argument(
        "--type2_ids",
        type=str,
        default="dataset/output/type2_ids.json",
        help="Path to type2_ids.json"
    )
    parser.add_argument(
        "--type3_ids",
        type=str,
        default="dataset/output/type3_ids.json",
        help="Path to type3_ids.json"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="grpo_data",
        help="Output directory for GRPO data"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="List of datasets to process (default: all)"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.98,
        help="Training split ratio (default: 0.95)"
    )

    args = parser.parse_args()

    prepare_grpo_data(
        type1_ids_file=args.type1_ids,
        type2_ids_file=args.type2_ids,
        type3_ids_file=args.type3_ids,
        output_dir=args.output_dir,
        datasets=args.datasets,
        train_split_ratio=args.train_split,
    )
