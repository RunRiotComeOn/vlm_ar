#!/usr/bin/env python3
"""
Prepare Type 3 training data for LLaMA-Factory SFT.

This script converts Type 3 responses (perception + reasoning + answer)
into LLaMA-Factory's sharegpt format for multimodal training.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from PIL import Image
import re

# Add dataset module to path
sys.path.append(str(Path(__file__).parent / "dataset"))
from load_datasets import (
    load_okvqa, load_geometry3k, load_scienceqa, load_gqa,
    load_clevr, load_mathvista, load_vcr, load_tqa, load_mathvision, load_chartqa
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
        'vcr': load_vcr,
        'tqa': load_tqa,
        'mathvision': load_mathvision,
        'chartqa': load_chartqa,
    }

    if dataset_name not in dataset_loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"\nLoading {dataset_name}...")
    return dataset_loaders[dataset_name]()


def get_global_id(dataset_name: str, metadata: Dict) -> str:
    """Generate global_id from dataset name and metadata."""
    if dataset_name == 'okvqa':
        return f"okvqa_{metadata['question_id']}"
    elif dataset_name == 'geometry3k':
        return f"geometry3k_{metadata['problem_id']}"
    elif dataset_name == 'scienceqa':
        return f"scienceqa_{metadata.get('id', metadata.get('index', 'unknown'))}"
    elif dataset_name == 'gqa':
        return f"gqa_{metadata['question_id']}"
    elif dataset_name == 'clevr':
        return f"clevr_{metadata['question_index']}"
    elif dataset_name == 'mathvista':
        return f"mathvista_{metadata['pid']}"
    elif dataset_name == 'vcr':
        return f"vcr_{metadata['question_id']}"
    elif dataset_name == 'tqa':
        return f"tqa_{metadata['question_id']}"
    elif dataset_name == 'mathvision':
        return metadata['question_id']  # Already has 'mathvision_' prefix from loader
    elif dataset_name == 'chartqa':
        return metadata['question_id']  # Already has 'chartqa_' prefix from loader
    else:
        return f"{dataset_name}_unknown"


def save_image(image, output_dir: Path, global_id: str) -> str:
    """Save PIL image to disk and return relative path."""
    # Create images directory
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Convert RGBA to RGB if necessary (JPEG doesn't support alpha channel)
    if image.mode == 'RGBA':
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[3])
        image = rgb_image
    elif image.mode not in ('RGB', 'L'):
        image = image.convert('RGB')

    # Save image
    image_path = images_dir / f"{global_id}.jpg"
    image.save(image_path, "JPEG")

    # Return relative path from LLaMA-Factory/data/ directory
    return f"vlm_adaptive_reasoning/images/{global_id}.jpg"


def parse_response(response: str) -> Dict[str, str]:
    """Parse response to extract perception, reasoning, and answer."""
    result = {
        'perception': '',
        'reasoning': '',
        'answer': ''
    }

    # Extract perception
    perception_match = re.search(r'<perception>(.*?)</perception>', response, re.DOTALL)
    if perception_match:
        result['perception'] = perception_match.group(1).strip()

    # Extract reasoning
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
    if reasoning_match:
        result['reasoning'] = reasoning_match.group(1).strip()

    # Extract answer
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_match:
        result['answer'] = answer_match.group(1).strip()

    return result


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


def prepare_sft_data(
    type3_responses_file: str,
    output_dir: str,
    datasets: List[str] = None
):
    """
    Prepare Type 3 SFT training data.

    Args:
        type3_responses_file: Path to type3_responses.jsonl
        output_dir: Output directory for training data
        datasets: List of dataset names to process (None for all)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load Type 3 responses
    print("Loading Type 3 responses...")
    type3_data = {}
    with open(type3_responses_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            type3_data[item['global_id']] = item

    print(f"Loaded {len(type3_data)} Type 3 responses")

    # Default datasets to process
    if datasets is None:
        datasets = ['okvqa', 'geometry3k', 'scienceqa', 'gqa', 'clevr', 'mathvista', 'ocrvqa', 'tqa', 'mathvision', 'chartqa']

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

            # Only process samples with Type 3 responses
            if global_id not in type3_data:
                continue

            # Save image
            try:
                image_path = save_image(item['image'], output_path, global_id)
            except Exception as e:
                print(f"Error saving image for {global_id}: {e}")
                continue

            # Get Type 3 response
            type3_item = type3_data[global_id]
            # response = type3_item.get('refined_response', '')
            response = type3_item.get('raw_response', '')

            # Parse response to extract components
            parsed = parse_response(response)

            # Handle answer format
            answer = item['answer']
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
            answer = str(answer)

            # Use parsed answer if available, otherwise use ground truth
            if parsed['answer']:
                answer = parsed['answer']

            # Create Type 3 sample
            sample = create_type3_sample(
                question=item['question'],
                perception=parsed['perception'],
                reasoning=parsed['reasoning'],
                answer=answer,
                image_path=image_path
            )

            all_samples.append(sample)

    # Save dataset
    print("\n" + "="*60)
    print("Saving SFT training data...")
    print("="*60)

    if all_samples:
        output_file = output_path / "train_all.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {len(all_samples)} Type 3 samples to {output_file}")
    else:
        print("⚠ No samples to save")

    # Print statistics
    print("\n" + "="*60)
    print("Statistics:")
    print("="*60)
    print(f"Type 3 samples: {len(all_samples)}")

    return all_samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Type 3 SFT training data")
    parser.add_argument(
        "--type3_responses",
        type=str,
        default="dataset/output/type3_responses.jsonl",
        help="Path to type3_responses.jsonl"
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

    prepare_sft_data(
        type3_responses_file=args.type3_responses,
        output_dir=args.output_dir,
        datasets=args.datasets
    )
