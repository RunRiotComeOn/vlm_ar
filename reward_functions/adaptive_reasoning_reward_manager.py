"""
Custom Reward Manager for Adaptive Reasoning with Batch Metrics Support

This reward manager wraps the AdaptiveReasoningReward function and provides
batch-level metrics that can be logged to TensorBoard.

Supports both V6 and V7:
- V6: Type1 bonus decay
- V7: Type1 bonus decay + Ratio control mechanism
"""

from collections import defaultdict
from typing import Any
import torch
from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

# Import our reward functions
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from adaptive_reasoning_reward_v6 import AdaptiveReasoningRewardV6
from adaptive_reasoning_reward_v7 import AdaptiveReasoningRewardV7


@register("adaptive_reasoning")
class AdaptiveReasoningRewardManager(AbstractRewardManager):
    """
    Reward manager for adaptive reasoning that supports batch metrics.

    This manager:
    1. Processes batches of responses using AdaptiveReasoningReward
    2. Computes batch-level metrics (format distribution, accuracy, etc.)
    3. Returns these metrics alongside rewards for TensorBoard logging
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        # Version selection
        reward_version: str = "v7",  # "v6" or "v7"
        # AdaptiveReasoningReward parameters
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,
        type1_format_bonus: float = 0.0,
        type2_format_bonus: float = 0.0,
        type3_format_bonus: float = 0.0,
        type1_error_penalty: float = -0.5,
        type2_error_penalty: float = -0.3,
        type3_error_penalty: float = 0.0,
        length_threshold: int = 150,
        ideal_length: float = 150.0,
        min_scalar: float = 0.3,
        enable_bonus_decay: bool = False,
        decay_strategy: str = "linear",
        decay_start_step: int = 0,
        decay_end_step: int = 100,
        type1_bonus_min: float = 0.0,
        decay_rate: float = 0.95,
        # V7 specific: Ratio control parameters
        enable_ratio_penalty: bool = False,
        ratio_penalty_start_step: int = 60,
        target_type1_ratio: float = 0.3,
        target_type2_ratio: float = 0.4,
        target_type3_ratio: float = 0.3,
        ratio_tolerance: float = 0.15,
        ratio_penalty_min_scalar: float = 0.5,
        ratio_window_size: int = 256,
        # Other settings
        normalize_answers: bool = True,
    ) -> None:
        """
        Initialize the AdaptiveReasoningRewardManager.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: Number of batches to print for debugging.
            compute_score: Custom compute_score function (unused, we use our own).
            reward_fn_key: Key for accessing data source.
            reward_version: Version to use ("v6" or "v7")
            **kwargs: Parameters passed to AdaptiveReasoningReward.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.reward_version = reward_version

        # Common parameters for both V6 and V7
        common_params = {
            "correct_reward": correct_reward,
            "incorrect_reward": incorrect_reward,
            "type1_format_bonus": type1_format_bonus,
            "type2_format_bonus": type2_format_bonus,
            "type3_format_bonus": type3_format_bonus,
            "type1_error_penalty": type1_error_penalty,
            "type2_error_penalty": type2_error_penalty,
            "type3_error_penalty": type3_error_penalty,
            "length_threshold": length_threshold,
            "ideal_length": ideal_length,
            "min_scalar": min_scalar,
            "enable_bonus_decay": enable_bonus_decay,
            "decay_strategy": decay_strategy,
            "decay_start_step": decay_start_step,
            "decay_end_step": decay_end_step,
            "type1_bonus_min": type1_bonus_min,
            "decay_rate": decay_rate,
            "normalize_answers": normalize_answers,
        }

        # Initialize the adaptive reasoning reward function
        if reward_version == "v7":
            # V7 with ratio control
            self.reward_fn = AdaptiveReasoningRewardV7(
                **common_params,
                enable_ratio_penalty=enable_ratio_penalty,
                ratio_penalty_start_step=ratio_penalty_start_step,
                target_type1_ratio=target_type1_ratio,
                target_type2_ratio=target_type2_ratio,
                target_type3_ratio=target_type3_ratio,
                ratio_tolerance=ratio_tolerance,
                ratio_penalty_min_scalar=ratio_penalty_min_scalar,
                ratio_window_size=ratio_window_size,
            )
        elif reward_version == "v6":
            # V6 without ratio control
            self.reward_fn = AdaptiveReasoningRewardV6(**common_params)
        else:
            raise ValueError(f"Unknown reward version: {reward_version}. Must be 'v6' or 'v7'.")

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """
        Compute rewards for a batch of responses with metrics support.

        Args:
            data: DataProto containing batch of responses
            return_dict: If True, return dict with reward_tensor and metrics

        Returns:
            reward_tensor or dict with reward_tensor and reward_extra_info
        """
        # If there is rm score, we directly return rm score
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # Collect all responses and ground truths for batch processing
        responses = []
        ground_truths = []
        valid_response_lengths = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            responses.append(response_str)
            ground_truths.append(ground_truth)
            valid_response_lengths.append(valid_response_length)

            # Print for debugging
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)

        # Compute rewards and metrics for the entire batch
        result = self.reward_fn(responses, ground_truths, return_dict=True)
        rewards = result['rewards']
        batch_metrics = result['metrics']

        # Fill reward tensor
        for i, (reward, valid_length) in enumerate(zip(rewards, valid_response_lengths)):
            reward_tensor[i, valid_length - 1] = reward

            # Store per-sample info
            reward_extra_info['score'].append(reward)

        if return_dict:
            # Add batch metrics to reward_extra_info
            # Use a special prefix to distinguish batch metrics from per-sample info
            for metric_name, metric_value in batch_metrics.items():
                # Store as a single value (not a list) for batch metrics
                reward_extra_info[f'adaptive_reasoning/{metric_name}'] = metric_value

            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
