#!/usr/bin/env python3
"""
Prepare training data for LLaMA-Factory SFT from Type 1, 2, 3 datasets.

This script converts the classified dataset into LLaMA-Factory's sharegpt format
for multimodal training with Qwen2.5-VL-3B-Instruct.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
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
    """Save PIL image to disk and return relative path."""
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

    # Return relative path from LLaMA-Factory/data/ directory
    # output_dir is LLaMA-Factory/data/vlm_adaptive_reasoning
    # so we need to return vlm_adaptive_reasoning/images/xxx.jpg
    return f"vlm_adaptive_reasoning/images/{global_id}.jpg"


def create_type1_sample(question: str, answer: str, image_path: str) -> Dict:
    """Create Type 1 sample: Direct answer without reasoning."""
    # Only add <image> tag if not already present
    if not question.strip().startswith("<image>"):
        question_content = f"<image>{question}"
    else:
        question_content = question

    # Wrap answer in <answer> tags
    response = f"<answer>\n{answer}\n</answer>"

    return {
        "messages": [
            {
                "content": question_content,
                "role": "user"
            },
            {
                "content": response,
                "role": "assistant"
            }
        ],
        "images": [image_path]
    }


def create_type2_sample(question: str, perception: str, answer: str, image_path: str) -> Dict:
    """Create Type 2 sample: Perception + Answer."""
    response = f"<perception>\n{perception}\n</perception>\n\n<answer>\n{answer}\n</answer>"

    # Only add <image> tag if not already present
    if not question.strip().startswith("<image>"):
        question_content = f"<image>{question}"
    else:
        question_content = question

    return {
        "messages": [
            {
                "content": question_content,
                "role": "user"
            },
            {
                "content": response,
                "role": "assistant"
            }
        ],
        "images": [image_path]
    }


def create_type3_sample(question: str, perception: str, reasoning: str, answer: str, image_path: str) -> Dict:
    """Create Type 3 sample: Perception + Reasoning + Answer."""
    response = f"<perception>\n{perception}\n</perception>\n\n<reasoning>\n{reasoning}\n</reasoning>\n\n<answer>\n{answer}\n</answer>"

    # Only add <image> tag if not already present
    if not question.strip().startswith("<image>"):
        question_content = f"<image>{question}"
    else:
        question_content = question

    return {
        "messages": [
            {
                "content": question_content,
                "role": "user"
            },
            {
                "content": response,
                "role": "assistant"
            }
        ],
        "images": [image_path]
    }


def prepare_training_data(
    type1_ids_file: str,
    type2_ids_file: str,
    type3_ids_file: str,
    type23_log_file: str,
    output_dir: str,
    datasets: List[str] = None
):
    """
    Prepare training data from classified datasets.

    Args:
        type1_ids_file: Path to type1_ids.json
        type2_ids_file: Path to type2_ids.json
        type3_ids_file: Path to type3_ids.json
        type23_log_file: Path to full_type23_classification_log.jsonl
        output_dir: Output directory for training data
        datasets: List of dataset names to process (None for all)
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

    # Load type2/3 classification log
    print("Loading type2/3 classification log...")
    type23_data = {}
    with open(type23_log_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            type23_data[item['global_id']] = item

    # Default datasets to process
    if datasets is None:
        datasets = ['okvqa', 'geometry3k', 'scienceqa', 'gqa', 'clevr', 'mathvista', 'tqa', 'ocrvqa', 'mathvision', 'chartqa']

    # Process each dataset
    all_samples = {
        'type1': [],
        'type2': [],
        'type3': [],
        'all': []
    }

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

            # Save image
            try:
                image_path = save_image(item['image'], output_path, global_id)
            except Exception as e:
                print(f"Error saving image for {global_id}: {e}")
                continue

            # Determine type and create sample
            sample = None
            sample_type = None

            if global_id in type1_ids:
                # Type 1: Direct answer
                # Handle answer being a list (e.g., from OKVQA)
                answer = item['answer']
                if isinstance(answer, list):
                    answer = answer[0] if answer else ""

                # For MCQ, answer is already the option letter from load_datasets
                # Just use it directly
                sample = create_type1_sample(
                    question=item['question'],
                    answer=str(answer),
                    image_path=image_path
                )
                sample_type = 'type1'

            elif global_id in type2_ids or global_id in type3_ids:
                # Type 2/3: Get from classification log
                if global_id not in type23_data:
                    print(f"Warning: {global_id} in type2/3 IDs but not in classification log")
                    continue

                log_item = type23_data[global_id]

                # For MCQ, use the option letter from item['answer'] (loaded from dataset)
                # For non-MCQ, use the answer from classification log
                metadata = item.get('metadata', {})
                has_choices = metadata.get('has_choices', False)

                if has_choices:
                    # MCQ: use option letter from dataset
                    answer = item['answer']
                    if isinstance(answer, list):
                        answer = answer[0] if answer else ""
                    answer = str(answer)
                else:
                    # Non-MCQ: use answer from log
                    answer = log_item.get('answer', log_item.get('answer_gt', ''))

                if global_id in type2_ids:
                    # Type 2: Perception + Answer
                    sample = create_type2_sample(
                        question=item['question'],
                        perception=log_item['perception'],
                        answer=answer,
                        image_path=image_path
                    )
                    sample_type = 'type2'
                else:
                    # Type 3: Perception + Reasoning + Answer
                    sample = create_type3_sample(
                        question=item['question'],
                        perception=log_item['perception'],
                        reasoning=log_item['reasoning'],
                        answer=answer,
                        image_path=image_path
                    )
                    sample_type = 'type3'

            if sample is not None:
                all_samples[sample_type].append(sample)
                all_samples['all'].append(sample)

    # Save datasets
    print("\n" + "="*60)
    print("Saving training data...")
    print("="*60)

    for key, samples in all_samples.items():
        if samples:
            output_file = output_path / f"train_{key}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved {len(samples)} samples to {output_file}")

    # Print statistics
    print("\n" + "="*60)
    print("Statistics:")
    print("="*60)
    print(f"Type 1 samples: {len(all_samples['type1'])}")
    print(f"Type 2 samples: {len(all_samples['type2'])}")
    print(f"Type 3 samples: {len(all_samples['type3'])}")
    print(f"Total samples: {len(all_samples['all'])}")

    return all_samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare training data for LLaMA-Factory")
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
        "--type23_log",
        type=str,
        default="dataset/output/full_type23_classification_log.jsonl",
        help="Path to type2/3 classification log"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="LLaMA-Factory/data/vlm_adaptive_reasoning",
        help="Output directory for training data"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="List of datasets to process (default: all)"
    )

    args = parser.parse_args()

    prepare_training_data(
        type1_ids_file=args.type1_ids,
        type2_ids_file=args.type2_ids,
        type3_ids_file=args.type3_ids,
        type23_log_file=args.type23_log,
        output_dir=args.output_dir,
        datasets=args.datasets
    )
