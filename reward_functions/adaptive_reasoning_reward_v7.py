"""
Adaptive Reasoning Reward Function V7 - With Type Ratio Control

Key Changes from V6:
- ADDED: Ratio-based penalty mechanism to control type distribution
- Each type gets a target ratio expectation (e.g., Type1=0.3, Type2=0.4, Type3=0.3)
- If actual ratio deviates too much from target, apply scalar penalty
- Penalty activates after a specified training step (default: step 60)
- Uses sliding window to track recent type distribution

All V6 features preserved:
- Type1 bonus decay mechanism
- Error penalties per type
- Length-based scalar penalty
- Comprehensive metrics tracking

Ratio Penalty Mechanism:
1. Track type distribution using sliding window
2. Calculate deviation from target ratios
3. Apply scalar penalty based on deviation magnitude
4. Penalty strength configurable via parameters
"""

import re
import math
from typing import List, Dict, Any, Optional
from collections import deque
import numpy as np


class AdaptiveReasoningRewardV7:
    """
    Reward function with ratio control and type1 bonus decay.

    Reward Formula:
        base_reward = (correctness_reward + format_bonus) or (incorrect_reward + error_penalty)
        length_scalar = calculate_length_scalar(token_count)
        ratio_scalar = calculate_ratio_scalar(response_type)  # NEW in V7
        final_reward = base_reward × length_scalar × ratio_scalar

    Ratio Control Mechanism:
        - Tracks type distribution in sliding window
        - Compares actual ratio to target ratio for each type
        - Applies scalar penalty if deviation exceeds tolerance
        - Only activates after ratio_penalty_start_step
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

        # Decay parameters for type1_format_bonus (from V6)
        enable_bonus_decay: bool = False,
        decay_strategy: str = "linear",  # "linear", "exponential", "cosine"
        decay_start_step: int = 0,
        decay_end_step: int = 30,
        type1_bonus_min: float = 0.0,
        decay_rate: float = 0.95,  # For exponential decay

        # NEW V7: Ratio control parameters
        enable_ratio_penalty: bool = False,
        ratio_penalty_start_step: int = 60,
        target_type1_ratio: float = 0.3,
        target_type2_ratio: float = 0.4,
        target_type3_ratio: float = 0.3,
        ratio_tolerance: float = 0.15,  # Allow ±15% deviation before penalty
        ratio_penalty_min_scalar: float = 0.5,  # Minimum scalar when ratio deviates
        ratio_window_size: int = 256,  # Sliding window size for tracking ratios

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
            min_scalar: Minimum allowed scalar for length penalty
            enable_bonus_decay: Whether to decay type1_format_bonus
            decay_strategy: Strategy for decay ("linear", "exponential", "cosine")
            decay_start_step: Training step to start decaying
            decay_end_step: Training step to finish decaying
            type1_bonus_min: Minimum value for type1 bonus after decay
            decay_rate: Decay rate for exponential strategy
            enable_ratio_penalty: Whether to enable ratio-based penalty (V7)
            ratio_penalty_start_step: Step to start applying ratio penalty (V7)
            target_type1_ratio: Target ratio for Type 1 (V7)
            target_type2_ratio: Target ratio for Type 2 (V7)
            target_type3_ratio: Target ratio for Type 3 (V7)
            ratio_tolerance: Tolerance before applying penalty (V7)
            ratio_penalty_min_scalar: Minimum scalar when ratio deviates (V7)
            ratio_window_size: Window size for tracking type distribution (V7)
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

        # Decay parameters (from V6)
        self.enable_bonus_decay = enable_bonus_decay
        self.decay_strategy = decay_strategy
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.type1_bonus_min = type1_bonus_min
        self.decay_rate = decay_rate
        self.current_step = 0

        # NEW V7: Ratio control parameters
        self.enable_ratio_penalty = enable_ratio_penalty
        self.ratio_penalty_start_step = ratio_penalty_start_step
        self.target_type1_ratio = target_type1_ratio
        self.target_type2_ratio = target_type2_ratio
        self.target_type3_ratio = target_type3_ratio
        self.ratio_tolerance = ratio_tolerance
        self.ratio_penalty_min_scalar = ratio_penalty_min_scalar
        self.ratio_window_size = ratio_window_size

        # Sliding window for tracking type distribution
        self.type_history = deque(maxlen=ratio_window_size)

        # Current ratio scalars (for metrics)
        self.current_ratio_scalars = {1: 1.0, 2: 1.0, 3: 1.0}

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

    def _calculate_ratio_scalar(self, response_type: int) -> float:
        """
        NEW V7: Calculate scalar penalty based on type ratio deviation.

        Args:
            response_type: Type of response (1, 2, or 3)

        Returns:
            Scalar multiplier (1.0 if no penalty, lower if ratio deviates)
        """
        # Don't apply penalty if disabled or before start step
        if not self.enable_ratio_penalty or self.current_step < self.ratio_penalty_start_step:
            return 1.0

        # Need enough samples in window
        if len(self.type_history) < 10:
            return 1.0

        # Calculate current ratios from window
        type_counts = {1: 0, 2: 0, 3: 0}
        for t in self.type_history:
            type_counts[t] += 1

        total = len(self.type_history)
        current_ratios = {
            1: type_counts[1] / total,
            2: type_counts[2] / total,
            3: type_counts[3] / total,
        }

        # Target ratios
        target_ratios = {
            1: self.target_type1_ratio,
            2: self.target_type2_ratio,
            3: self.target_type3_ratio,
        }

        # Calculate deviation for this response type
        current_ratio = current_ratios[response_type]
        target_ratio = target_ratios[response_type]
        deviation = current_ratio - target_ratio

        # FIXED: Penalize when deviation exceeds tolerance in EITHER direction
        # If abs(deviation) is within tolerance, no penalty
        # If abs(deviation) > tolerance, apply penalty to discourage this type
        if abs(deviation) <= self.ratio_tolerance:
            # Within tolerance range - no penalty
            scalar = 1.0
        else:
            # Outside tolerance - apply penalty proportional to excess deviation
            # Penalty applies whether over-used OR under-used (both hurt balance)
            excess_deviation = abs(deviation) - self.ratio_tolerance

            # Scale penalty: more deviation = stronger penalty
            # scalar goes from 1.0 (at tolerance) down to min_scalar (at max deviation)
            # Max possible deviation is 1.0 - tolerance (if this type is 100% or 0%)
            max_deviation = 1.0 - self.ratio_tolerance
            if max_deviation > 0:
                penalty_factor = min(1.0, excess_deviation / max_deviation)
                scalar = 1.0 - penalty_factor * (1.0 - self.ratio_penalty_min_scalar)
            else:
                scalar = 1.0

        return max(self.ratio_penalty_min_scalar, min(1.0, scalar))

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
            'ratio_scalars': [],  # NEW V7
            'total_rewards': [],
        }

        # Update ratio scalars for this batch
        for response_type in [1, 2, 3]:
            self.current_ratio_scalars[response_type] = self._calculate_ratio_scalar(response_type)

        for response, gt in zip(responses, ground_truths):
            # 1. Check correctness
            is_correct = self.check_answer_correctness(response, gt)

            # 2. Determine response type
            response_type = self.get_response_type(response)

            # Add to history for ratio tracking
            self.type_history.append(response_type)

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

            # 5. NEW V7: Calculate ratio scalar
            ratio_scalar = self.current_ratio_scalars[response_type]

            # 6. Final reward calculation with ratio penalty
            if is_correct:
                reward = (base_reward + format_bonus) * length_scalar * ratio_scalar
            else:
                reward = (base_reward + error_penalty) * length_scalar * ratio_scalar

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
            batch_stats['ratio_scalars'].append(ratio_scalar)  # NEW V7
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
            metrics['reward/ratio_scalar_mean'] = float(np.mean(batch_stats['ratio_scalars']))  # NEW V7
            metrics['reward/total_mean'] = float(np.mean(batch_stats['total_rewards']))

        # Overall accuracy
        if total_samples > 0:
            metrics['accuracy/overall'] = batch_stats['correct_count'] / total_samples

        # Add decay-specific metrics (from V6)
        if self.enable_bonus_decay:
            metrics['decay/type1_bonus_current'] = self.type1_format_bonus
            metrics['decay/current_step'] = float(self.current_step)

        # NEW V7: Add ratio control metrics
        if self.enable_ratio_penalty:
            metrics['ratio_control/enabled'] = 1.0 if self.current_step >= self.ratio_penalty_start_step else 0.0
            metrics['ratio_control/type1_scalar'] = self.current_ratio_scalars[1]
            metrics['ratio_control/type2_scalar'] = self.current_ratio_scalars[2]
            metrics['ratio_control/type3_scalar'] = self.current_ratio_scalars[3]
            metrics['ratio_control/target_type1_ratio'] = self.target_type1_ratio
            metrics['ratio_control/target_type2_ratio'] = self.target_type2_ratio
            metrics['ratio_control/target_type3_ratio'] = self.target_type3_ratio

            # Current ratios from window
            if len(self.type_history) > 0:
                type_counts = {1: 0, 2: 0, 3: 0}
                for t in self.type_history:
                    type_counts[t] += 1
                total = len(self.type_history)
                metrics['ratio_control/window_type1_ratio'] = type_counts[1] / total
                metrics['ratio_control/window_type2_ratio'] = type_counts[2] / total
                metrics['ratio_control/window_type3_ratio'] = type_counts[3] / total

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
        ratio_scalar = self._calculate_ratio_scalar(response_type)

        if is_correct:
            total_reward = (base_reward + format_bonus) * length_scalar * ratio_scalar
        else:
            total_reward = (base_reward + error_penalty) * length_scalar * ratio_scalar

        return {
            'total_reward': total_reward,
            'is_correct': is_correct,
            'response_type': response_type,
            'base_reward': base_reward,
            'format_bonus': format_bonus if is_correct else 0.0,
            'error_penalty': error_penalty if not is_correct else 0.0,
            'token_count': token_count,
            'length_scalar': length_scalar,
            'ratio_scalar': ratio_scalar,  # NEW V7
            'has_perception': self.has_perception_tag(response),
            'has_reasoning': self.has_reasoning_tag(response),
            'predicted_answer': self.extract_answer(response),
            'type1_bonus_current': self.type1_format_bonus,
        }


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("Adaptive Reasoning Reward Function V7 - With Type Ratio Control")
    print("="*80)

    print("\n📊 Testing Ratio Control Mechanism:\n")

    reward_fn = AdaptiveReasoningRewardV7(
        type1_format_bonus=0.5,
        type2_format_bonus=0.3,
        type3_format_bonus=0.0,
        enable_ratio_penalty=True,
        ratio_penalty_start_step=10,
        target_type1_ratio=0.3,
        target_type2_ratio=0.4,
        target_type3_ratio=0.3,
        ratio_tolerance=0.1,
        ratio_penalty_min_scalar=0.5,
        ratio_window_size=20,
    )

    # Simulate training steps with different type distributions
    print("Simulating Type 1 over-exploitation (80% Type 1):")

    # Fill window with 80% Type 1, 10% Type 2, 10% Type 3
    for _ in range(16):
        reward_fn.type_history.append(1)
    for _ in range(2):
        reward_fn.type_history.append(2)
    for _ in range(2):
        reward_fn.type_history.append(3)

    reward_fn.set_training_step(15)  # After ratio penalty starts

    for response_type in [1, 2, 3]:
        scalar = reward_fn._calculate_ratio_scalar(response_type)
        print(f"  Type {response_type} ratio scalar: {scalar:.4f}")

    print("\nSimulating balanced distribution (30%, 40%, 30%):")

    # Clear and fill with balanced distribution
    reward_fn.type_history.clear()
    for _ in range(6):
        reward_fn.type_history.append(1)
    for _ in range(8):
        reward_fn.type_history.append(2)
    for _ in range(6):
        reward_fn.type_history.append(3)

    for response_type in [1, 2, 3]:
        scalar = reward_fn._calculate_ratio_scalar(response_type)
        print(f"  Type {response_type} ratio scalar: {scalar:.4f}")

    print("\n" + "="*80)
    print("Testing Complete Reward Calculation:")
    print("="*80)

    test_responses = [
        "Red",  # Type 1
        "<perception>I see a red object</perception>\n<answer>Red</answer>",  # Type 2
        "<perception>I see a red object</perception>\n<reasoning>The color is clearly red based on visual analysis</reasoning>\n<answer>Red</answer>",  # Type 3
    ]
    test_gt = "red"

    reward_fn = AdaptiveReasoningRewardV7(
        type1_format_bonus=0.2,
        type2_format_bonus=0.4,
        type3_format_bonus=0.6,
        enable_ratio_penalty=True,
        ratio_penalty_start_step=0,
        target_type1_ratio=0.3,
        target_type2_ratio=0.4,
        target_type3_ratio=0.3,
    )

    # Simulate Type 1 over-use
    for _ in range(50):
        reward_fn.type_history.append(1)

    result = reward_fn(test_responses, [test_gt, test_gt, test_gt], return_dict=True)

    print("\nWith Type 1 over-exploitation (ratio penalty active):")
    for i, (resp, rew) in enumerate(zip(test_responses, result['rewards'])):
        rtype = reward_fn.get_response_type(resp)
        breakdown = reward_fn.get_reward_breakdown(resp, test_gt)
        print(f"\nType {rtype}:")
        print(f"  Response: {resp[:50]}...")
        print(f"  Reward: {rew:.4f}")
        print(f"  Ratio scalar: {breakdown['ratio_scalar']:.4f}")

    print("\n" + "="*80)
    print("Key Features:")
    print("  ✓ V7: Ratio control to prevent type over-exploitation")
    print("  ✓ V7: Configurable target ratios per type")
    print("  ✓ V7: Activates after specified training step")
    print("  ✓ V7: Uses sliding window to track distribution")
    print("  ✓ V6: Type1 bonus decay mechanism")
    print("  ✓ V6: Multiple decay strategies")
    print("  ✓ All previous features preserved")
    print("="*80)
