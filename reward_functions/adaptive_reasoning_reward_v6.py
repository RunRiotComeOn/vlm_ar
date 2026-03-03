"""
Adaptive Reasoning Reward Function V6 - With Type1 Bonus Decay

Key Changes from V5:
- ADDED: Decay mechanism for type1_format_bonus (to prevent over-exploitation)
- Supports multiple decay strategies: linear, exponential, cosine
- Tracks training steps and adjusts bonus accordingly
- All other features from V5 preserved

Decay Strategies:
1. Linear: Linearly decrease from initial to minimum value
2. Exponential: Exponentially decay with configurable rate
3. Cosine: Cosine annealing schedule (smooth decay)

Use Case:
If type1_ratio reaches 0.8 too quickly (e.g., at step 12), enable decay to:
- Reduce exploitation of Type 1 format
- Encourage more balanced exploration of Type 2 and Type 3
- Maintain training stability
"""

import re
import math
from typing import List, Dict, Any
from collections import deque
import numpy as np


class AdaptiveReasoningRewardV6:
    """
    Reward function with type1_format_bonus decay mechanism.

    Reward Formula:
        if correct:
            reward = (correctness_reward + format_bonus) × length_scalar
        if incorrect:
            reward = incorrect_reward + error_penalty

        Then apply length_scalar to final reward.

    Decay Mechanism:
        type1_bonus = calculate_decayed_bonus(current_step)
        - Decays from initial value to minimum value
        - Based on training step and decay strategy
    """

    def __init__(
        self,
        # Correctness rewards
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,

        # Format bonuses (Type 1 > Type 2 > Type 3)
        type1_format_bonus: float = 0.5,
        type2_format_bonus: float = 0.3,
        type3_format_bonus: float = 0.0,

        # Error penalties (only applied when incorrect)
        type1_error_penalty: float = -0.5,
        type2_error_penalty: float = 0.0,
        type3_error_penalty: float = 0.0,

        # Length penalty parameters
        length_threshold: int = 300,
        ideal_length: float = 300.0,
        min_scalar: float = 0.3,

        # Decay parameters for type1_format_bonus
        enable_bonus_decay: bool = False,
        decay_strategy: str = "linear",  # "linear", "exponential", "cosine"
        decay_start_step: int = 0,
        decay_end_step: int = 30,
        type1_bonus_min: float = 0.0,
        decay_rate: float = 0.95,  # For exponential decay

        # Other settings
        normalize_answers: bool = True,
    ):
        """
        Args:
            correct_reward: Reward for correct answer
            incorrect_reward: Base reward for incorrect answer
            type1_format_bonus: Initial bonus for Type 1 when correct
            type2_format_bonus: Bonus for Type 2 when correct
            type3_format_bonus: Bonus for Type 3 when correct
            type1_error_penalty: Penalty for Type 1 when incorrect
            type2_error_penalty: Penalty for Type 2 when incorrect
            type3_error_penalty: Penalty for Type 3 when incorrect
            length_threshold: Token count below which there's no penalty
            ideal_length: Reference length for calculating scalar
            min_scalar: Minimum allowed scalar
            enable_bonus_decay: Whether to decay type1_format_bonus
            decay_strategy: Strategy for decay ("linear", "exponential", "cosine")
            decay_start_step: Training step to start decaying
            decay_end_step: Training step to finish decaying
            type1_bonus_min: Minimum value for type1 bonus after decay
            decay_rate: Decay rate for exponential strategy
            normalize_answers: Whether to normalize answers
        """
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward

        # Store initial and current bonus values
        self.type1_format_bonus_initial = type1_format_bonus
        self.type1_format_bonus = type1_format_bonus
        self.type2_format_bonus = type2_format_bonus
        self.type3_format_bonus = type3_format_bonus

        self.type1_error_penalty = type1_error_penalty
        self.type2_error_penalty = type2_error_penalty
        self.type3_error_penalty = type3_error_penalty

        self.length_threshold = length_threshold
        self.ideal_length = ideal_length
        self.min_scalar = min_scalar
        self.normalize_answers = normalize_answers

        # Decay parameters
        self.enable_bonus_decay = enable_bonus_decay
        self.decay_strategy = decay_strategy
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.type1_bonus_min = type1_bonus_min
        self.decay_rate = decay_rate
        self.current_step = 0

    def set_training_step(self, step: int):
        """
        Update current training step and recalculate type1 bonus.

        This should be called at the beginning of each training step.
        """
        self.current_step = step
        if self.enable_bonus_decay:
            self.type1_format_bonus = self._calculate_decayed_bonus(step)

    def _calculate_decayed_bonus(self, step: int) -> float:
        """
        Calculate the decayed type1_format_bonus based on current step.

        Args:
            step: Current training step

        Returns:
            Decayed bonus value
        """
        # Before decay starts
        if step < self.decay_start_step:
            return self.type1_format_bonus_initial

        # After decay ends
        if step >= self.decay_end_step:
            return self.type1_bonus_min

        # During decay period
        initial = self.type1_format_bonus_initial
        minimum = self.type1_bonus_min
        start = self.decay_start_step
        end = self.decay_end_step

        if self.decay_strategy == "linear":
            # Linear decay: bonus = initial - (initial - min) * progress
            progress = (step - start) / (end - start)
            bonus = initial - (initial - minimum) * progress

        elif self.decay_strategy == "exponential":
            # Exponential decay: bonus = min + (initial - min) * decay_rate^(step - start)
            steps_elapsed = step - start
            bonus = minimum + (initial - minimum) * (self.decay_rate ** steps_elapsed)

        elif self.decay_strategy == "cosine":
            # Cosine annealing: smooth decay using cosine function
            progress = (step - start) / (end - start)
            bonus = minimum + (initial - minimum) * 0.5 * (1 + math.cos(math.pi * progress))

        else:
            raise ValueError(f"Unknown decay strategy: {self.decay_strategy}")

        # Clamp to [minimum, initial]
        bonus = max(minimum, min(initial, bonus))
        return bonus

    def extract_answer(self, response: str) -> str:
        """Extract answer from response."""
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()
        return response.strip()

    def has_perception_tag(self, response: str) -> bool:
        """Check if response contains <perception> tag."""
        return bool(re.search(r'<perception>.*?</perception>', response, re.DOTALL | re.IGNORECASE))

    def has_reasoning_tag(self, response: str) -> bool:
        """Check if response contains <reasoning> tag."""
        return bool(re.search(r'<reasoning>.*?</reasoning>', response, re.DOTALL | re.IGNORECASE))

    def get_response_type(self, response: str) -> int:
        """
        Determine the response type.

        Returns:
            1: Type 1 (no perception or reasoning tags)
            2: Type 2 (perception but no reasoning)
            3: Type 3 (both perception and reasoning)
        """
        has_perception = self.has_perception_tag(response)
        has_reasoning = self.has_reasoning_tag(response)

        if has_perception and has_reasoning:
            return 3
        elif has_perception:
            return 2
        else:
            return 1

    def count_tokens(self, text: str) -> int:
        """Estimate token count (simple approximation)."""
        words = len(text.split())
        return int(words * 1.3)

    def calculate_length_scalar(self, token_count: int) -> float:
        """Calculate length penalty scalar."""
        if token_count <= self.length_threshold:
            return 1.0

        scalar = self.ideal_length / token_count
        scalar = max(self.min_scalar, min(1.0, scalar))
        return scalar

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if not self.normalize_answers:
            return answer

        answer = answer.lower().strip()
        answer = answer.rstrip('.,!?;:')
        answer = ' '.join(answer.split())

        # Remove articles
        if answer.startswith('the '):
            answer = answer[4:]
        if answer.startswith('a '):
            answer = answer[2:]
        if answer.startswith('an '):
            answer = answer[3:]

        return answer

    def check_answer_correctness(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth."""
        pred_answer = self.extract_answer(predicted)
        pred_norm = self.normalize_answer(pred_answer)

        # Handle multiple ground truth answers
        if isinstance(ground_truth, list):
            gt_answers = ground_truth
        else:
            gt_answers = [ground_truth]

        for gt in gt_answers:
            gt_norm = self.normalize_answer(str(gt))

            # Multiple choice detection
            if len(gt_norm) == 1 and gt_norm.upper() in ['A', 'B', 'C', 'D', 'E']:
                pattern = rf'\b{gt_norm.upper()}\b'
                if re.search(pattern, pred_answer.upper()):
                    return True
                continue

            # Exact match
            if pred_norm == gt_norm:
                return True

            # Substring match
            if gt_norm in pred_norm or pred_norm in gt_norm:
                return True

            # Numerical comparison
            try:
                pred_num = float(pred_norm)
                gt_num = float(gt_norm)
                if abs(pred_num - gt_num) < 1e-6:
                    return True
            except (ValueError, TypeError):
                pass

        return False

    def __call__(
        self,
        responses: List[str],
        ground_truths: List[Any],
        return_dict: bool = False,
        **kwargs
    ):
        """
        Calculate rewards for a batch of responses.

        Args:
            responses: List of model responses
            ground_truths: List of ground truth answers
            return_dict: If True, return dict with rewards and metrics
            **kwargs: Additional arguments (ignored)

        Returns:
            List[float] or dict: Reward scores, or dict with 'rewards' and 'metrics'
        """
        rewards = []

        # Batch statistics for metrics
        batch_stats = {
            'type1_count': 0,
            'type2_count': 0,
            'type3_count': 0,
            'correct_count': 0,
            'incorrect_count': 0,
            'type1_correct': 0,
            'type2_correct': 0,
            'type3_correct': 0,
            'type1_lengths': [],
            'type2_lengths': [],
            'type3_lengths': [],
            'base_rewards': [],
            'format_bonuses': [],
            'length_scalars': [],
            'total_rewards': [],
        }

        for response, gt in zip(responses, ground_truths):
            # 1. Check correctness
            is_correct = self.check_answer_correctness(response, gt)

            # 2. Determine response type
            response_type = self.get_response_type(response)

            # 3. Get format bonus or error penalty based on correctness
            if is_correct:
                base_reward = self.correct_reward
                if response_type == 1:
                    format_bonus = self.type1_format_bonus  # Uses current (possibly decayed) value
                elif response_type == 2:
                    format_bonus = self.type2_format_bonus
                else:  # type 3
                    format_bonus = self.type3_format_bonus
                error_penalty = 0.0
            else:
                base_reward = self.incorrect_reward
                format_bonus = 0.0
                if response_type == 1:
                    error_penalty = self.type1_error_penalty
                elif response_type == 2:
                    error_penalty = self.type2_error_penalty
                else:  # type 3
                    error_penalty = self.type3_error_penalty

            # 4. Calculate length scalar
            token_count = self.count_tokens(response)
            length_scalar = self.calculate_length_scalar(token_count)

            # 5. Final reward calculation
            if is_correct:
                reward = (base_reward + format_bonus) * length_scalar
            else:
                reward = (base_reward + error_penalty) * length_scalar

            rewards.append(reward)

            # Collect statistics
            if response_type == 1:
                batch_stats['type1_count'] += 1
                batch_stats['type1_lengths'].append(token_count)
                if is_correct:
                    batch_stats['type1_correct'] += 1
            elif response_type == 2:
                batch_stats['type2_count'] += 1
                batch_stats['type2_lengths'].append(token_count)
                if is_correct:
                    batch_stats['type2_correct'] += 1
            else:  # type 3
                batch_stats['type3_count'] += 1
                batch_stats['type3_lengths'].append(token_count)
                if is_correct:
                    batch_stats['type3_correct'] += 1

            if is_correct:
                batch_stats['correct_count'] += 1
            else:
                batch_stats['incorrect_count'] += 1

            batch_stats['base_rewards'].append(base_reward)
            batch_stats['format_bonuses'].append(format_bonus if is_correct else error_penalty)
            batch_stats['length_scalars'].append(length_scalar)
            batch_stats['total_rewards'].append(reward)

        if not return_dict:
            return rewards

        # Calculate metrics
        total_samples = len(responses)
        metrics = self._compute_batch_metrics(batch_stats, total_samples)

        return {
            'rewards': rewards,
            'metrics': metrics
        }

    def _compute_batch_metrics(self, batch_stats: Dict, total_samples: int) -> Dict[str, float]:
        """
        Compute comprehensive metrics from batch statistics.

        Returns metrics for TensorBoard logging.
        """
        metrics = {}

        # Format distribution
        if total_samples > 0:
            metrics['format/type1_ratio'] = batch_stats['type1_count'] / total_samples
            metrics['format/type2_ratio'] = batch_stats['type2_count'] / total_samples
            metrics['format/type3_ratio'] = batch_stats['type3_count'] / total_samples

        # Per-type accuracy
        if batch_stats['type1_count'] > 0:
            metrics['format/type1_correct_rate'] = batch_stats['type1_correct'] / batch_stats['type1_count']
            metrics['format/type1_avg_length'] = float(np.mean(batch_stats['type1_lengths']))
        else:
            metrics['format/type1_correct_rate'] = 0.0
            metrics['format/type1_avg_length'] = 0.0

        if batch_stats['type2_count'] > 0:
            metrics['format/type2_correct_rate'] = batch_stats['type2_correct'] / batch_stats['type2_count']
            metrics['format/type2_avg_length'] = float(np.mean(batch_stats['type2_lengths']))
        else:
            metrics['format/type2_correct_rate'] = 0.0
            metrics['format/type2_avg_length'] = 0.0

        if batch_stats['type3_count'] > 0:
            metrics['format/type3_correct_rate'] = batch_stats['type3_correct'] / batch_stats['type3_count']
            metrics['format/type3_avg_length'] = float(np.mean(batch_stats['type3_lengths']))
        else:
            metrics['format/type3_correct_rate'] = 0.0
            metrics['format/type3_avg_length'] = 0.0

        # Reward components
        if batch_stats['base_rewards']:
            metrics['reward/base_mean'] = float(np.mean(batch_stats['base_rewards']))
            metrics['reward/format_bonus_mean'] = float(np.mean(batch_stats['format_bonuses']))
            metrics['reward/length_scalar_mean'] = float(np.mean(batch_stats['length_scalars']))
            metrics['reward/total_mean'] = float(np.mean(batch_stats['total_rewards']))

        # Overall accuracy
        if total_samples > 0:
            metrics['accuracy/overall'] = batch_stats['correct_count'] / total_samples

        # Add decay-specific metrics
        if self.enable_bonus_decay:
            metrics['decay/type1_bonus_current'] = self.type1_format_bonus
            metrics['decay/current_step'] = float(self.current_step)

        return metrics

    def get_reward_breakdown(
        self,
        response: str,
        ground_truth: Any
    ) -> Dict[str, float]:
        """Get detailed breakdown of reward components for a single response."""
        is_correct = self.check_answer_correctness(response, ground_truth)
        response_type = self.get_response_type(response)
        token_count = self.count_tokens(response)

        base_reward = self.correct_reward if is_correct else self.incorrect_reward

        if is_correct:
            if response_type == 1:
                format_bonus = self.type1_format_bonus
            elif response_type == 2:
                format_bonus = self.type2_format_bonus
            else:
                format_bonus = self.type3_format_bonus
            error_penalty = 0.0
        else:
            format_bonus = 0.0
            if response_type == 1:
                error_penalty = self.type1_error_penalty
            elif response_type == 2:
                error_penalty = self.type2_error_penalty
            else:
                error_penalty = self.type3_error_penalty

        length_scalar = self.calculate_length_scalar(token_count)

        if is_correct:
            total_reward = (base_reward + format_bonus) * length_scalar
        else:
            total_reward = (base_reward + error_penalty) * length_scalar

        return {
            'total_reward': total_reward,
            'is_correct': is_correct,
            'response_type': response_type,
            'base_reward': base_reward,
            'format_bonus': format_bonus if is_correct else 0.0,
            'error_penalty': error_penalty if not is_correct else 0.0,
            'token_count': token_count,
            'length_scalar': length_scalar,
            'has_perception': self.has_perception_tag(response),
            'has_reasoning': self.has_reasoning_tag(response),
            'predicted_answer': self.extract_answer(response),
            'type1_bonus_current': self.type1_format_bonus,
        }


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("Adaptive Reasoning Reward Function V6 - With Type1 Bonus Decay")
    print("="*80)

    # Test decay strategies
    print("\n📉 Testing Decay Strategies:\n")

    strategies = ["linear", "exponential", "cosine"]

    for strategy in strategies:
        print(f"\n{strategy.upper()} Decay:")
        reward_fn = AdaptiveReasoningRewardV6(
            type1_format_bonus=0.5,
            enable_bonus_decay=True,
            decay_strategy=strategy,
            decay_start_step=10,
            decay_end_step=50,
            type1_bonus_min=0.0,
            decay_rate=0.95,  # For exponential
        )

        test_steps = [0, 5, 10, 20, 30, 40, 50, 60]
        print(f"  Step -> Type1 Bonus")
        for step in test_steps:
            reward_fn.set_training_step(step)
            print(f"  {step:3d} -> {reward_fn.type1_format_bonus:.4f}")

    # Test with actual responses
    print("\n" + "="*80)
    print("Testing Reward Calculation with Decay")
    print("="*80)

    reward_fn = AdaptiveReasoningRewardV6(
        type1_format_bonus=0.5,
        type2_format_bonus=0.3,
        type3_format_bonus=0.0,
        enable_bonus_decay=True,
        decay_strategy="linear",
        decay_start_step=0,
        decay_end_step=20,
        type1_bonus_min=0.1,
    )

    test_response = "Red"
    test_gt = "red"

    print(f"\nTest Response: '{test_response}'")
    print(f"Ground Truth: '{test_gt}'")
    print(f"\nRewards at different training steps:")

    for step in [0, 5, 10, 15, 20, 25]:
        reward_fn.set_training_step(step)
        breakdown = reward_fn.get_reward_breakdown(test_response, test_gt)
        print(f"\nStep {step}:")
        print(f"  Type1 Bonus: {breakdown['type1_bonus_current']:.4f}")
        print(f"  Total Reward: {breakdown['total_reward']:.4f}")

    print("\n" + "="*80)
    print("Key Features:")
    print("  ✓ Type1 bonus decays over training (prevents over-exploitation)")
    print("  ✓ Three decay strategies: linear, exponential, cosine")
    print("  ✓ Configurable start/end steps and minimum value")
    print("  ✓ Tracks current bonus in metrics for monitoring")
    print("  ✓ All V5 features preserved (error penalties, length penalty, etc.)")
    print("="*80)
