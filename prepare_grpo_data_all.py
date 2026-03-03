#!/usr/bin/env python3
"""
Prepare GRPO training data from multiple VQA datasets.

This script loads multiple datasets with configurable sample ratios
and prepares them for GRPO training in verl's expected format.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from PIL import Image
import random

# Add dataset module to path
sys.path.append(str(Path(__file__).parent / "dataset"))
from load_datasets import (
    load_okvqa, load_geometry3k, load_scienceqa, load_gqa,
    load_clevr, load_mathvista, load_tqa, load_ocrvqa, load_mathvision, load_chartqa
)


# Default sample ratios for each dataset (1.0 = use all data)
DEFAULT_SAMPLE_RATIOS = {
    'okvqa': 1.0,
    'geometry3k': 1.0,
    'scienceqa': 1.0,
    'gqa': 0.04,
    'clevr': 0.08,
    'mathvista': 1.0,
    'tqa': 1.0,
    'ocrvqa': 0.015,
    'mathvision': 1.0,
    'chartqa': 1.0,
}

# Dataset loader mapping
DATASET_LOADERS = {
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


def load_dataset_by_name(dataset_name: str) -> List[Dict]:
    """Load dataset by name."""
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_LOADERS.keys())}")

    print(f"\nLoading {dataset_name}...")
    return DATASET_LOADERS[dataset_name]()


def sample_dataset(data: List[Dict], sample_ratio: float, random_seed: int = 42) -> List[Dict]:
    """Sample a subset of the dataset based on sample ratio."""
    if sample_ratio >= 1.0:
        return data
    
    random.seed(random_seed)
    num_samples = max(1, int(len(data) * sample_ratio))
    return random.sample(data, num_samples)


def get_global_id(dataset_name: str, metadata: Dict, idx: int = 0) -> str:
    """Generate global_id from dataset name and metadata."""
    id_extractors = {
        'okvqa': lambda m: f"okvqa_{m.get('question_id', idx)}",
        'geometry3k': lambda m: f"geometry3k_{m.get('problem_id', idx)}",
        'scienceqa': lambda m: f"scienceqa_{m.get('id', m.get('index', idx))}",
        'gqa': lambda m: f"gqa_{m.get('question_id', idx)}",
        'clevr': lambda m: f"clevr_{m.get('question_index', idx)}",
        'mathvista': lambda m: f"mathvista_{m.get('pid', idx)}",
        'tqa': lambda m: f"tqa_{m.get('question_id', idx)}",
        'ocrvqa': lambda m: f"ocrvqa_{m.get('question_id', idx)}",
        'mathvision': lambda m: f"mathvision_{m.get('question_id', idx)}".replace('mathvision_mathvision_', 'mathvision_'),
        'chartqa': lambda m: f"chartqa_{m.get('question_id', idx)}".replace('chartqa_chartqa_', 'chartqa_'),
    }
    
    extractor = id_extractors.get(dataset_name, lambda m: f"{dataset_name}_{idx}")
    return extractor(metadata)


def save_image(image, output_dir: Path, global_id: str) -> str:
    """Save PIL image to disk and return path."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Convert to RGB if necessary (JPEG doesn't support alpha channel)
    if image.mode == 'RGBA':
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[3])
        image = rgb_image
    elif image.mode not in ('RGB', 'L'):
        image = image.convert('RGB')

    # Save image
    image_path = images_dir / f"{global_id}.jpg"
    image.save(image_path, "JPEG", quality=95)

    return str(image_path.absolute())


def get_dataset_stats(data: List[Dict], dataset_name: str) -> Dict:
    """Calculate statistics for a dataset."""
    if not data:
        return {
            'total_samples': 0,
            'has_choices_count': 0,
            'free_form_count': 0,
            'avg_question_length': 0,
            'unique_answers': 0,
        }
    
    has_choices = sum(1 for d in data if d.get('metadata', {}).get('has_choices', False))
    free_form = len(data) - has_choices
    
    question_lengths = [len(d.get('question', '')) for d in data]
    avg_question_length = sum(question_lengths) / len(question_lengths) if question_lengths else 0
    
    # Count unique answers
    answers = set()
    for d in data:
        ans = d.get('answer')
        if isinstance(ans, list):
            answers.update(str(a) for a in ans)
        else:
            answers.add(str(ans))
    
    return {
        'total_samples': len(data),
        'has_choices_count': has_choices,
        'free_form_count': free_form,
        'avg_question_length': round(avg_question_length, 2),
        'unique_answers': len(answers),
    }


def generate_summary_report(
    dataset_stats: Dict[str, Dict],
    sample_ratios: Dict[str, float],
    output_path: Path,
    train_samples: int,
    val_samples: int,
) -> str:
    """Generate a comprehensive summary report."""
    
    report_lines = [
        "=" * 80,
        "GRPO Training Data Summary Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        "",
        "## Dataset Overview",
        "",
        f"{'Dataset':<15} {'Sample Ratio':<12} {'Original':<10} {'Sampled':<10} {'MCQ':<8} {'Free-form':<10} {'Avg Q Len':<10}",
        "-" * 80,
    ]
    
    total_original = 0
    total_sampled = 0
    total_mcq = 0
    total_freeform = 0
    
    for dataset_name, stats in dataset_stats.items():
        ratio = sample_ratios.get(dataset_name, 1.0)
        original = stats.get('original_count', stats['total_samples'])
        sampled = stats['total_samples']
        mcq = stats['has_choices_count']
        freeform = stats['free_form_count']
        avg_len = stats['avg_question_length']
        
        total_original += original
        total_sampled += sampled
        total_mcq += mcq
        total_freeform += freeform
        
        report_lines.append(
            f"{dataset_name:<15} {ratio:<12.2%} {original:<10} {sampled:<10} {mcq:<8} {freeform:<10} {avg_len:<10.1f}"
        )
    
    report_lines.extend([
        "-" * 80,
        f"{'TOTAL':<15} {'':<12} {total_original:<10} {total_sampled:<10} {total_mcq:<8} {total_freeform:<10}",
        "",
        "## Train/Validation Split",
        f"  - Training samples: {train_samples}",
        f"  - Validation samples: {val_samples}",
        f"  - Split ratio: {train_samples/(train_samples+val_samples):.2%} / {val_samples/(train_samples+val_samples):.2%}",
        "",
        "## Question Type Distribution",
        f"  - Multiple Choice (MCQ): {total_mcq} ({total_mcq/total_sampled*100:.1f}%)" if total_sampled > 0 else "  - Multiple Choice (MCQ): 0",
        f"  - Free-form: {total_freeform} ({total_freeform/total_sampled*100:.1f}%)" if total_sampled > 0 else "  - Free-form: 0",
        "",
        "## Output Files",
        f"  - Train: {output_path / 'train.parquet'}",
        f"  - Val: {output_path / 'val.parquet'}",
        f"  - Metadata: {output_path / 'metadata.json'}",
        f"  - Summary: {output_path / 'summary.txt'}",
        f"  - Images: {output_path / 'images/'}",
        "",
        "=" * 80,
    ])
    
    report = "\n".join(report_lines)
    
    # Save report
    summary_file = output_path / "summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report


def prepare_grpo_data(
    output_dir: str,
    datasets: List[str] = None,
    sample_ratios: Dict[str, float] = None,
    train_split_ratio: float = 0.95,
    random_seed: int = 42,
):
    """
    Prepare GRPO training data from multiple datasets.

    verl expects data in parquet format with columns:
    - data_source: identifier
    - prompt: list of message dicts in chat format
    - images: list of image paths
    - reward_model: dict with ground_truth
    - sample_type: dataset type indicator
    - dataset: source dataset name

    Args:
        output_dir: Output directory for GRPO data
        datasets: List of dataset names to process (None for all)
        sample_ratios: Dict mapping dataset name to sample ratio (0.0-1.0)
        train_split_ratio: Ratio of training data (rest is validation)
        random_seed: Random seed for reproducibility
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set random seed
    random.seed(random_seed)

    # Default datasets to process
    if datasets is None:
        datasets = list(DATASET_LOADERS.keys())

    # Merge sample ratios with defaults
    effective_ratios = DEFAULT_SAMPLE_RATIOS.copy()
    if sample_ratios:
        effective_ratios.update(sample_ratios)

    # Process each dataset
    all_samples = []
    dataset_stats = {}

    for dataset_name in datasets:
        if dataset_name not in DATASET_LOADERS:
            print(f"Warning: Unknown dataset '{dataset_name}', skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {dataset_name}...")
        print(f"Sample ratio: {effective_ratios.get(dataset_name, 1.0):.2%}")
        print('='*60)

        try:
            # Load dataset
            full_data = load_dataset_by_name(dataset_name)
            original_count = len(full_data)
            
            # Sample dataset
            ratio = effective_ratios.get(dataset_name, 1.0)
            data = sample_dataset(full_data, ratio, random_seed)
            
            print(f"Original: {original_count}, After sampling: {len(data)}")

        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        dataset_samples = []
        for idx, item in enumerate(tqdm(data, desc=f"Processing {dataset_name}")):
            global_id = get_global_id(dataset_name, item.get('metadata', {}), idx)

            # Save image
            try:
                image_path = save_image(item['image'], output_path, global_id)
            except Exception as e:
                print(f"Error saving image for {global_id}: {e}")
                continue

            # Handle answer (can be list or single value)
            answer = item['answer']
            if isinstance(answer, list):
                gt_answer = answer
            else:
                gt_answer = [str(answer)]

            # Format prompt in chat message format
            question = item['question']
            if not question.strip().startswith("<image>"):
                question_with_image = f"<image>\n{question}"
            else:
                question_with_image = question

            sample = {
                'data_source': global_id,
                'prompt': [{"role": "user", "content": question_with_image}],
                'images': [{"image": image_path}],
                'reward_model': {
                    'ground_truth': gt_answer
                },
                'sample_type': dataset_name,
                'dataset': dataset_name,
                'metadata': item.get('metadata', {}),
            }

            dataset_samples.append(sample)

        all_samples.extend(dataset_samples)
        
        # Calculate stats for this dataset
        stats = get_dataset_stats(data, dataset_name)
        stats['original_count'] = original_count
        stats['processed_count'] = len(dataset_samples)
        dataset_stats[dataset_name] = stats
        
        print(f"✓ Processed {len(dataset_samples)} samples from {dataset_name}")

    if not all_samples:
        print("\nError: No samples were processed!")
        return None, None

    # Convert metadata dict to JSON string for parquet compatibility
    for sample in all_samples:
        if 'metadata' in sample:
            sample['metadata'] = json.dumps(sample['metadata'], ensure_ascii=False)

    # Convert to DataFrame
    df = pd.DataFrame(all_samples)

    # Shuffle
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Split into train and validation
    split_idx = int(len(df) * train_split_ratio)
    train_df = df[:split_idx]
    val_df = df[split_idx:]

    # Save as parquet
    train_file = output_path / "train.parquet"
    val_file = output_path / "val.parquet"

    train_df.to_parquet(train_file, index=False)
    val_df.to_parquet(val_file, index=False)

    # Generate and print summary report
    report = generate_summary_report(
        dataset_stats=dataset_stats,
        sample_ratios=effective_ratios,
        output_path=output_path,
        train_samples=len(train_df),
        val_samples=len(val_df),
    )
    print("\n" + report)

    # Save detailed metadata
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'random_seed': random_seed,
        'train_split_ratio': train_split_ratio,
        'total_samples': len(df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'sample_ratios': effective_ratios,
        'datasets_processed': list(dataset_stats.keys()),
        'dataset_stats': dataset_stats,
        'dataset_distribution': {
            'train': train_df['dataset'].value_counts().to_dict(),
            'val': val_df['dataset'].value_counts().to_dict(),
        },
        'output_files': {
            'train': str(train_file),
            'val': str(val_file),
            'images_dir': str(output_path / 'images'),
        }
    }

    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\n✓ Metadata saved to: {metadata_file}")
    print("=" * 60)

    return train_df, val_df


def parse_sample_ratios(ratio_args: List[str]) -> Dict[str, float]:
    """Parse sample ratio arguments like 'okvqa=0.5 gqa=0.3'."""
    ratios = {}
    if ratio_args:
        for arg in ratio_args:
            if '=' in arg:
                dataset, ratio = arg.split('=')
                ratios[dataset.strip()] = float(ratio.strip())
    return ratios


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare GRPO training data from multiple VQA datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all datasets with default settings
  python prepare_grpo_data.py --output_dir grpo_data

  # Process specific datasets
  python prepare_grpo_data.py --output_dir grpo_data --datasets okvqa gqa mathvista

  # Use custom sample ratios
  python prepare_grpo_data.py --output_dir grpo_data --sample_ratios okvqa=0.5 gqa=0.3 mathvista=1.0

  # Combined example
  python prepare_grpo_data.py --output_dir grpo_data \\
      --datasets okvqa gqa scienceqa mathvista \\
      --sample_ratios okvqa=0.2 gqa=0.1 \\
      --train_split 0.9
        """
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="grpo_data_all",
        help="Output directory for GRPO data"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help=f"List of datasets to process. Available: {list(DATASET_LOADERS.keys())}"
    )
    parser.add_argument(
        "--sample_ratios",
        type=str,
        nargs="+",
        default=None,
        help="Sample ratios for each dataset in format 'dataset=ratio' (e.g., 'okvqa=0.5 gqa=0.3')"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.95,
        help="Training split ratio (default: 0.95)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Parse sample ratios
    sample_ratios = parse_sample_ratios(args.sample_ratios)

    prepare_grpo_data(
        output_dir=args.output_dir,
        datasets=args.datasets,
        sample_ratios=sample_ratios,
        train_split_ratio=args.train_split,
        random_seed=args.seed,
    )
