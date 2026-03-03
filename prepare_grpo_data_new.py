#!/usr/bin/env python3
"""
Prepare GRPO training data from cold_start_9k SFT data.

Converts LLaMA-Factory/data/cold_start_9k/train_all.json into parquet
format expected by verl's GRPO training pipeline.

Parquet schema:
    - data_source: str (unique ID per sample)
    - prompt: list[dict] (user message only)
    - images: list[dict] with {"image": "/abs/path/to/img.jpg"}
    - reward_model: dict with {"ground_truth": "<answer content>"}
"""

import json
import re
import os
import random
from pathlib import Path

import pandas as pd


def extract_answer_from_assistant(text: str) -> str:
    """Extract answer content from <answer>...</answer> in assistant message."""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def prepare_grpo_data(
    input_json: str = "LLaMA-Factory/data/cold_start_9k/train_all.json",
    output_dir: str = "grpo_data_new",
    train_split: float = 0.95,
    seed: int = 42,
):
    """
    Convert cold_start_9k train_all.json to verl-compatible parquet.

    Args:
        input_json: Path to train_all.json
        output_dir: Output directory for parquet files
        train_split: Fraction of data for training (rest is validation)
        seed: Random seed
    """
    input_path = Path(input_json)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Base directory for resolving image paths
    data_base_dir = Path(input_path).parent.parent  # LLaMA-Factory/data/

    print(f"Loading data from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")

    samples = []
    skipped = 0

    for idx, entry in enumerate(data):
        messages = entry.get('messages', [])
        images = entry.get('images', [])

        # Extract user and assistant messages
        user_msg = None
        assistant_msg = None
        for msg in messages:
            if msg['role'] == 'user':
                user_msg = msg['content']
            elif msg['role'] == 'assistant':
                assistant_msg = msg['content']

        if user_msg is None or assistant_msg is None:
            skipped += 1
            continue

        # Extract ground truth from assistant's <answer> tag
        ground_truth = extract_answer_from_assistant(assistant_msg)

        # Resolve image path to absolute
        if images:
            rel_image = images[0]  # e.g., "cold_start_9k/images/7.jpg"
            abs_image = str((data_base_dir / rel_image).resolve())
        else:
            skipped += 1
            continue

        # Verify image exists
        if not os.path.isfile(abs_image):
            skipped += 1
            continue

        sample = {
            'data_source': f'cold_start_9k_{idx}',
            'prompt': [{"role": "user", "content": user_msg}],
            'images': [{"image": abs_image}],
            'reward_model': {'ground_truth': ground_truth},
        }
        samples.append(sample)

    print(f"Processed {len(samples)} samples, skipped {skipped}")

    # Shuffle
    random.seed(seed)
    random.shuffle(samples)

    # Split
    split_idx = int(len(samples) * train_split)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # Save parquet
    train_df = pd.DataFrame(train_samples)
    val_df = pd.DataFrame(val_samples)

    train_file = output_path / "train.parquet"
    val_file = output_path / "val.parquet"

    train_df.to_parquet(train_file, index=False)
    val_df.to_parquet(val_file, index=False)

    print(f"\nTrain: {len(train_df)} samples -> {train_file}")
    print(f"Val:   {len(val_df)} samples -> {val_file}")

    # Verify schema
    print("\nSchema verification:")
    check_df = pd.read_parquet(train_file)
    print(f"  Columns: {list(check_df.columns)}")
    print(f"  data_source example: {check_df['data_source'].iloc[0]}")
    print(f"  prompt example: {str(check_df['prompt'].iloc[0])[:120]}...")
    print(f"  images example: {check_df['images'].iloc[0]}")
    print(f"  reward_model example: {check_df['reward_model'].iloc[0]}")

    return train_df, val_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare GRPO data from cold_start_9k")
    parser.add_argument("--input", type=str,
                        default="LLaMA-Factory/data/cold_start_9k/train_all.json",
                        help="Path to train_all.json")
    parser.add_argument("--output_dir", type=str, default="grpo_data_new",
                        help="Output directory")
    parser.add_argument("--train_split", type=float, default=0.95,
                        help="Train split ratio (default: 0.95)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    prepare_grpo_data(
        input_json=args.input,
        output_dir=args.output_dir,
        train_split=args.train_split,
        seed=args.seed,
    )
