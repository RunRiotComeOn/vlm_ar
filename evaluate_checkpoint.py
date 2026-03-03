#!/usr/bin/env python3
"""
Evaluate GRPO checkpoint on validation set and generate detailed report.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import re

# Add reward function path
sys.path.insert(0, str(Path(__file__).parent / "reward_functions"))
from adaptive_reasoning_reward import AdaptiveReasoningReward


class CheckpointEvaluator:
    def __init__(
        self,
        checkpoint_path: str,
        val_data_path: str,
        output_dir: str,
        project_name: str = "verl_grpo_adaptive_reasoning",
        experiment_name: str = "qwen2_5vl_3b_adaptive",
        checkpoint_name: str = "step_90",
        max_samples: int = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 2048,
    ):
        """
        Initialize evaluator.

        Args:
            checkpoint_path: Path to GRPO checkpoint
            val_data_path: Path to validation parquet file
            output_dir: Base directory to save evaluation results
            project_name: Project name (added as folder level)
            experiment_name: Experiment name (added as folder level)
            checkpoint_name: Checkpoint name (added as folder level)
            max_samples: Maximum number of samples to evaluate (None = all)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_new_tokens: Maximum tokens to generate
        """
        self.checkpoint_path = checkpoint_path
        self.val_data_path = val_data_path
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.checkpoint_name = checkpoint_name

        # Build output directory with project/experiment/checkpoint hierarchy
        self.output_dir = Path(output_dir) / project_name / experiment_name / checkpoint_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_samples = max_samples
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        # Load reward function
        self.reward_fn = AdaptiveReasoningReward()

        print(f"Loading checkpoint from: {checkpoint_path}")
        self.load_model()

        print(f"Loading validation data from: {val_data_path}")
        self.load_data()

    def load_model(self):
        """Load model and processor from checkpoint."""
        checkpoint_path = Path(self.checkpoint_path)

        # Check if this is a merged HuggingFace model or FSDP checkpoint
        if (checkpoint_path / "config.json").exists():
            # Already a HuggingFace model
            model_path = checkpoint_path
        elif (checkpoint_path / "actor" / "config.json").exists():
            # FSDP checkpoint with actor directory (merged)
            model_path = checkpoint_path / "actor"
        elif (checkpoint_path / "actor" / "huggingface" / "config.json").exists():
            # FSDP checkpoint not yet merged - error
            raise ValueError(
                f"FSDP checkpoint at {checkpoint_path} needs to be merged first.\n"
                f"Run: python -m verl.model_merger merge --backend fsdp "
                f"--local_dir {checkpoint_path}/actor --target_dir {checkpoint_path}/actor_merged"
            )
        else:
            raise ValueError(f"Invalid checkpoint format at {checkpoint_path}")

        print(f"Loading model from {model_path}")

        # Load model with AutoModel to handle different architectures
        self.model = AutoModelForVision2Seq.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        # Load processor (should be in the same directory)
        self.processor = AutoProcessor.from_pretrained(str(model_path))

        print("Model loaded successfully!")

    def load_data(self):
        """Load validation data."""
        self.val_df = pd.read_parquet(self.val_data_path)

        if self.max_samples is not None:
            self.val_df = self.val_df.head(self.max_samples)

        print(f"Loaded {len(self.val_df)} validation samples")

    def generate_response(self, prompt: str, images: Any) -> str:
        """
        Generate response for a single sample.

        Args:
            prompt: Text prompt
            images: List of image paths or list of dicts with 'image' key

        Returns:
            Generated response string
        """
        # Prepare messages in Qwen2-VL format
        messages = [
            {
                "role": "user",
                "content": []
            }
        ]

        # Handle different image formats
        # images can be: list of strings, list of dicts, or numpy array
        if isinstance(images, (list, tuple)) or hasattr(images, '__iter__'):
            for img in images:
                if isinstance(img, dict) and 'image' in img:
                    img_path = img['image']
                elif isinstance(img, str):
                    img_path = img
                else:
                    continue

                if os.path.exists(img_path):
                    messages[0]["content"].append({
                        "type": "image",
                        "image": img_path
                    })

        # Add text prompt
        messages[0]["content"].append({
            "type": "text",
            "text": prompt
        })

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
            )

        # Decode output
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return response

    def evaluate(self) -> Dict[str, Any]:
        """
        Run evaluation on all validation samples.

        Returns:
            Dictionary with evaluation results
        """
        results = []

        print(f"\n{'='*80}")
        print(f"Running Evaluation on {len(self.val_df)} samples")
        print(f"{'='*80}\n")

        for idx, row in tqdm(self.val_df.iterrows(), total=len(self.val_df)):
            try:
                # Generate response
                response = self.generate_response(
                    prompt=row['prompt'],
                    images=row['images']
                )

                # Calculate reward breakdown
                breakdown = self.reward_fn.get_reward_breakdown(
                    response=response,
                    ground_truth=row['gt_answer']
                )

                # Store result
                result = {
                    'index': idx,
                    'dataset': row.get('dataset', 'unknown'),
                    'sample_type': row.get('sample_type', 'unknown'),
                    'prompt': row['prompt'],
                    'images': row['images'],
                    'ground_truth': row['gt_answer'],
                    'response': response,
                    'predicted_answer': breakdown['predicted_answer'],
                    'is_correct': breakdown['is_correct'],
                    'has_perception': breakdown['has_perception'],
                    'has_reasoning': breakdown['has_reasoning'],
                    'base_reward': breakdown['base_reward'],
                    'perception_penalty': breakdown['perception_penalty'],
                    'reasoning_penalty': breakdown['reasoning_penalty'],
                    'length_penalty': breakdown['length_penalty'],
                    'total_reward': breakdown['total_reward'],
                }

                results.append(result)

            except Exception as e:
                print(f"\nError processing sample {idx}: {e}")
                continue

        return self.compute_metrics(results)

    def compute_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Compute aggregate metrics from results."""
        if not results:
            return {
                'results': [],
                'summary': {'error': 'No results to compute metrics'}
            }

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)

        # Overall metrics
        summary = {
            'total_samples': len(df),
            'accuracy': df['is_correct'].mean(),
            'avg_reward': df['total_reward'].mean(),
            'std_reward': df['total_reward'].std(),
            'perception_usage': df['has_perception'].mean(),
            'reasoning_usage': df['has_reasoning'].mean(),
        }

        # Metrics by sample type
        if 'sample_type' in df.columns:
            summary['by_type'] = {}
            for sample_type in df['sample_type'].unique():
                type_df = df[df['sample_type'] == sample_type]
                summary['by_type'][sample_type] = {
                    'count': len(type_df),
                    'accuracy': type_df['is_correct'].mean(),
                    'avg_reward': type_df['total_reward'].mean(),
                    'perception_usage': type_df['has_perception'].mean(),
                    'reasoning_usage': type_df['has_reasoning'].mean(),
                }

        # Metrics by dataset
        if 'dataset' in df.columns:
            summary['by_dataset'] = {}
            for dataset in df['dataset'].unique():
                dataset_df = df[df['dataset'] == dataset]
                summary['by_dataset'][str(dataset)] = {
                    'count': len(dataset_df),
                    'accuracy': dataset_df['is_correct'].mean(),
                    'avg_reward': dataset_df['total_reward'].mean(),
                }

        # Reward distribution
        summary['reward_distribution'] = {
            '0.0 (incorrect)': (df['total_reward'] == 0.0).sum(),
            '0.5 (type3)': ((df['total_reward'] >= 0.45) & (df['total_reward'] <= 0.55)).sum(),
            '0.8 (type2)': ((df['total_reward'] >= 0.75) & (df['total_reward'] <= 0.85)).sum(),
            '1.0 (type1)': (df['total_reward'] >= 0.95).sum(),
        }

        return {
            'results': results,
            'summary': summary,
            'dataframe': df
        }

    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    def save_results(self, eval_output: Dict[str, Any]):
        """Save evaluation results to files."""
        # Convert to JSON serializable format
        json_data = {
            'project_name': self.project_name,
            'experiment_name': self.experiment_name,
            'checkpoint_name': self.checkpoint_name,
            'checkpoint_path': str(self.checkpoint_path),
            'validation_data': str(self.val_data_path),
            'temperature': self.temperature,
            'top_p': self.top_p,
            'summary': self._convert_to_json_serializable(eval_output['summary']),
            'results': self._convert_to_json_serializable(eval_output['results'])
        }

        # Save detailed JSON report
        json_path = self.output_dir / "evaluation_detailed.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {json_path}")

        # Save summary report (human-readable)
        summary_path = self.output_dir / "evaluation_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("GRPO Checkpoint Evaluation Report\n")
            f.write("="*80 + "\n\n")

            f.write(f"Project: {self.project_name}\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Checkpoint: {self.checkpoint_name}\n")
            f.write(f"Checkpoint Path: {self.checkpoint_path}\n")
            f.write(f"Validation Data: {self.val_data_path}\n")
            f.write(f"Temperature: {self.temperature}\n")
            f.write(f"Top-p: {self.top_p}\n\n")

            summary = eval_output['summary']

            f.write("-"*80 + "\n")
            f.write("Overall Metrics\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Samples: {summary['total_samples']}\n")
            f.write(f"Accuracy: {summary['accuracy']:.2%}\n")
            f.write(f"Average Reward: {summary['avg_reward']:.4f} ± {summary['std_reward']:.4f}\n")
            f.write(f"Perception Tag Usage: {summary['perception_usage']:.2%}\n")
            f.write(f"Reasoning Tag Usage: {summary['reasoning_usage']:.2%}\n\n")

            f.write("-"*80 + "\n")
            f.write("Reward Distribution\n")
            f.write("-"*80 + "\n")
            for reward_type, count in summary['reward_distribution'].items():
                f.write(f"{reward_type}: {count}\n")
            f.write("\n")

            if 'by_type' in summary:
                f.write("-"*80 + "\n")
                f.write("Metrics by Sample Type\n")
                f.write("-"*80 + "\n")
                for sample_type, metrics in summary['by_type'].items():
                    f.write(f"\n{sample_type}:\n")
                    f.write(f"  Count: {metrics['count']}\n")
                    f.write(f"  Accuracy: {metrics['accuracy']:.2%}\n")
                    f.write(f"  Avg Reward: {metrics['avg_reward']:.4f}\n")
                    f.write(f"  Perception Usage: {metrics['perception_usage']:.2%}\n")
                    f.write(f"  Reasoning Usage: {metrics['reasoning_usage']:.2%}\n")

            if 'by_dataset' in summary:
                f.write("\n" + "-"*80 + "\n")
                f.write("Metrics by Dataset\n")
                f.write("-"*80 + "\n")
                for dataset, metrics in summary['by_dataset'].items():
                    f.write(f"\n{dataset}:\n")
                    f.write(f"  Count: {metrics['count']}\n")
                    f.write(f"  Accuracy: {metrics['accuracy']:.2%}\n")
                    f.write(f"  Avg Reward: {metrics['avg_reward']:.4f}\n")

            f.write("\n" + "="*80 + "\n")

        print(f"Summary report saved to: {summary_path}")

        # Save results as CSV
        csv_path = self.output_dir / "evaluation_results.csv"
        eval_output['dataframe'].to_csv(csv_path, index=False)
        print(f"Results CSV saved to: {csv_path}")

        # Print summary to console
        print(f"\n{'='*80}")
        print("Evaluation Summary")
        print(f"{'='*80}")
        print(f"Accuracy: {summary['accuracy']:.2%}")
        print(f"Average Reward: {summary['avg_reward']:.4f} ± {summary['std_reward']:.4f}")
        print(f"Perception Usage: {summary['perception_usage']:.2%}")
        print(f"Reasoning Usage: {summary['reasoning_usage']:.2%}")
        print(f"{'='*80}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate GRPO checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="saves/qwen2_5vl-3b/grpo/adaptive_reasoning/global_step_90",
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="test/test.parquet",
        help="Path to validation parquet file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Base output directory (project/experiment/checkpoint folders will be added)"
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="step_90",
        help="Checkpoint name (folder level, e.g., step_90)"
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="verl_grpo_adaptive_reasoning",
        help="Project name (folder level)"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="qwen2_5vl_3b_adaptive",
        help="Experiment name (folder level)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter"
    )

    args = parser.parse_args()

    # Run evaluation
    evaluator = CheckpointEvaluator(
        checkpoint_path=args.checkpoint,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        checkpoint_name=args.checkpoint_name,
        max_samples=args.max_samples,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    eval_output = evaluator.evaluate()
    evaluator.save_results(eval_output)


if __name__ == "__main__":
    main()
