"""
Adaptive Reasoning Reward Function V5 - Simplified with Error Penalties

Key Changes from Previous Version:
- REMOVED: diversity_scalar (no format diversity enforcement)
- REVERSED format bonus: Type 1 > Type 2 > Type 3 (encourages efficiency)
- ADDED: Error penalties for Type 1 (-0.5) and Type 2 (-0.3)
- Kept: Length penalty and comprehensive metrics

This reward function encourages efficient reasoning:
- Type 1 (direct answer): Highest bonus when correct, heavy penalty when wrong
- Type 2 (perception + answer): Medium bonus when correct, medium penalty when wrong
- Type 3 (full reasoning): No bonus, no penalty - neutral baseline
- Length Penalty: Dynamic scalar based on token count (prevents verbosity)
- Comprehensive Metrics: Track format distribution, accuracy, length, and reward components

Design Philosophy:
- Simple questions → Use Type 1 → High reward if correct, heavy penalty if wrong
- Medium questions → Use Type 2 → Medium reward if correct, medium penalty if wrong
- Complex questions → Use Type 3 → Safe baseline, no extra risk/reward
- Overly verbose answers → Heavy penalty regardless of type
"""

import re
from typing import List, Dict, Any
from collections import deque
import numpy as np


class AdaptiveReasoningRewardV5:
    """
    Simplified reward function with reversed format bonuses and error penalties.

    Reward Formula:
        if correct:
            reward = (correctness_reward + format_bonus) × length_scalar
        if incorrect:
            reward = incorrect_reward + error_penalty

        Then apply length_scalar to final reward.

    Where:
        - correctness_reward: 1.0 if correct
        - incorrect_reward: 0.0 if incorrect (base)
        - format_bonus (correct only): Type 1 > Type 2 > Type 3
        - error_penalty (incorrect only): Type 1 = -0.5, Type 2 = -0.3, Type 3 = 0.0
        - length_scalar: Penalty for overly long responses

    Metrics Logged (for TensorBoard):
        - format/type{1,2,3}_ratio: % of each response type in batch
        - format/type{1,2,3}_correct_rate: Accuracy per type
        - format/type{1,2,3}_avg_length: Average token length per type
        - reward/base_mean: Mean base reward (correctness)
        - reward/format_bonus_mean: Mean format bonus
        - reward/length_scalar_mean: Mean length penalty
        - reward/total_mean: Mean total reward
        - accuracy/overall: Overall accuracy
    """

    def __init__(
        self,
        # Correctness rewards
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,

        # Format bonuses (Type 1 > Type 2 > Type 3) - REVERSED!
        type1_format_bonus: float = 0.5,
        type2_format_bonus: float = 0.3,
        type3_format_bonus: float = 0.0,

        # Error penalties (only applied when incorrect)
        type1_error_penalty: float = -0.5,
        type2_error_penalty: float = -0.3,
        type3_error_penalty: float = 0.0,

        # Length penalty parameters
        length_threshold: int = 300,      # No penalty below this
        ideal_length: float = 300.0,      # Reference for penalty calculation
        min_scalar: float = 0.3,          # Minimum scalar (max penalty)

        # Other settings
        normalize_answers: bool = True,
    ):
        """
        Args:
            correct_reward: Reward for correct answer
            incorrect_reward: Base reward for incorrect answer (before penalty)
            type1_format_bonus: Bonus for Type 1 when correct
            type2_format_bonus: Bonus for Type 2 when correct
            type3_format_bonus: Bonus for Type 3 when correct
            type1_error_penalty: Penalty for Type 1 when incorrect
            type2_error_penalty: Penalty for Type 2 when incorrect
            type3_error_penalty: Penalty for Type 3 when incorrect
            length_threshold: Token count below which there's no penalty
            ideal_length: Reference length for calculating scalar
            min_scalar: Minimum allowed scalar (prevents reward from going too low)
            normalize_answers: Whether to normalize answers before comparison
        """
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
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
        """
        Estimate token count (simple approximation).

        For more accurate counting, you can replace this with:
        - tiktoken for OpenAI models
        - transformers tokenizer for specific models
        """
        # Simple approximation: split by whitespace and punctuation
        # Typically 1 token ≈ 0.75 words for English
        words = len(text.split())
        return int(words * 1.3)  # Rough estimate

    def calculate_length_scalar(self, token_count: int) -> float:
        """
        Calculate length penalty scalar.

        Logic:
            - token_count <= threshold: scalar = 1.0 (no penalty)
            - token_count > threshold: scalar = ideal_length / token_count
            - scalar is clamped to [min_scalar, 1.0]

        Example (threshold=300, ideal_length=300):
            - 100 tokens → 1.0 (no penalty)
            - 300 tokens → 1.0 (no penalty)
            - 400 tokens → 300/400 = 0.75 (25% penalty)
            - 600 tokens → 300/600 = 0.5 (50% penalty)
            - 1000 tokens → 300/1000 = 0.3 (70% penalty, clamped to min_scalar)
        """
        if token_count <= self.length_threshold:
            return 1.0

        # Calculate scalar
        scalar = self.ideal_length / token_count

        # Clamp to [min_scalar, 1.0]
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
                    format_bonus = self.type1_format_bonus
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
                # Correct: (base + format_bonus) × length_scalar
                reward = (base_reward + format_bonus) * length_scalar
            else:
                # Incorrect: (base + error_penalty) × length_scalar
                reward = (base_reward + error_penalty) * length_scalar

            # 6. Ensure non-negative (but allow negative from error penalties)
            # Actually, we DO allow negative rewards to penalize errors
            # reward = max(0.0, reward)  # Removed this line to allow negative rewards

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

        Returns metrics for TensorBoard logging with format:
        - format/type{1,2,3}_ratio: Percentage of each response type
        - format/type{1,2,3}_correct_rate: Accuracy per type
        - format/type{1,2,3}_avg_length: Average token length per type
        - reward/base_mean: Mean base reward (correctness)
        - reward/format_bonus_mean: Mean format bonus/penalty
        - reward/length_scalar_mean: Mean length penalty
        - reward/total_mean: Mean total reward
        - accuracy/overall: Overall accuracy
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

        return metrics

    def get_reward_breakdown(
        self,
        response: str,
        ground_truth: Any
    ) -> Dict[str, float]:
        """
        Get detailed breakdown of reward components for a single response.
        Useful for debugging and analysis.
        """
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
        }


# Global reward instance (initialized once)
_reward_instance = None


def create_reward_function(data_source, solution_str, ground_truth, extra_info=None, **init_kwargs):
    """
    Reward function interface expected by verl.

    Args:
        data_source: Data source identifier (unused)
        solution_str: Model's response string
        ground_truth: Ground truth answer(s)
        extra_info: Additional information (unused)
        **init_kwargs: Initialization arguments for AdaptiveReasoningRewardV5

    Returns:
        float: Reward score
    """
    global _reward_instance

    if _reward_instance is None:
        _reward_instance = AdaptiveReasoningRewardV5(**init_kwargs)

    rewards = _reward_instance([solution_str], [ground_truth])
    return rewards[0]


# Example usage and testing
if __name__ == "__main__":
    reward_fn = AdaptiveReasoningRewardV5(
        type1_format_bonus=0.2,
        type2_format_bonus=0.1,
        type3_format_bonus=0.0,
        type1_error_penalty=-0.5,
        type2_error_penalty=-0.3,
        type3_error_penalty=0.0,
    )

    # Test cases
    test_cases = [
        {
            'response': 'Red',
            'gt': 'red',
            'description': 'Type 1 - Correct (should get +0.2 bonus)',
        },
        {
            'response': 'Blue',
            'gt': 'red',
            'description': 'Type 1 - Incorrect (should get -0.5 penalty)',
        },
        {
            'response': '<perception>I see a red car.</perception>\n<answer>Red</answer>',
            'gt': 'red',
            'description': 'Type 2 - Correct (should get +0.1 bonus)',
        },
        {
            'response': '<perception>I see a blue car.</perception>\n<answer>Blue</answer>',
            'gt': 'red',
            'description': 'Type 2 - Incorrect (should get -0.3 penalty)',
        },
        {
            'response': '<perception>The car is red.</perception>\n<reasoning>It is clearly red.</reasoning>\n<answer>Red</answer>',
            'gt': 'red',
            'description': 'Type 3 - Correct (no bonus)',
        },
        {
            'response': '<perception>The car is blue.</perception>\n<reasoning>It looks blue.</reasoning>\n<answer>Blue</answer>',
            'gt': 'red',
            'description': 'Type 3 - Incorrect (no penalty)',
        },
    ]

    print("="*80)
    print("Adaptive Reasoning Reward Function V5")
    print("Reversed Format Bonus + Error Penalties")
    print("="*80)
    print("\nReward Design:")
    print("  Correct:   reward = (1.0 + format_bonus) × length_scalar")
    print("  Incorrect: reward = (0.0 + error_penalty) × length_scalar")
    print(f"\n  Format Bonus (correct):  Type1={reward_fn.type1_format_bonus}, Type2={reward_fn.type2_format_bonus}, Type3={reward_fn.type3_format_bonus}")
    print(f"  Error Penalty (incorrect): Type1={reward_fn.type1_error_penalty}, Type2={reward_fn.type2_error_penalty}, Type3={reward_fn.type3_error_penalty}")
    print(f"  Length Penalty: threshold={reward_fn.length_threshold}, ideal={reward_fn.ideal_length}")
    print("="*80)

    # Test batch processing with metrics
    print("\n📊 Testing Batch Processing with Metrics:")
    responses = [tc['response'] for tc in test_cases]
    ground_truths = [tc['gt'] for tc in test_cases]

    result = reward_fn(responses, ground_truths, return_dict=True)
    rewards = result['rewards']
    metrics = result['metrics']

    print(f"\nBatch Metrics:")
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.4f}")

    print(f"\nIndividual Test Results:")
    for i, (tc, reward) in enumerate(zip(test_cases, rewards)):
        breakdown = reward_fn.get_reward_breakdown(tc['response'], tc['gt'])
        print(f"\n{i+1}. {tc['description']}")
        print(f"   Response Type: {breakdown['response_type']}, Correct: {breakdown['is_correct']}")
        print(f"   Base: {breakdown['base_reward']:.2f}, Bonus: {breakdown['format_bonus']:.2f}, Penalty: {breakdown['error_penalty']:.2f}")
        print(f"   Length Scalar: {breakdown['length_scalar']:.2f}, Tokens: {breakdown['token_count']}")
        print(f"   ✨ Total Reward: {reward:.4f}")

    print("\n" + "="*80)
    print("Key Features:")
    print("  ✓ Type 1 correct → +0.2 bonus (encourages efficiency when confident)")
    print("  ✓ Type 1 incorrect → -0.5 penalty (discourages risky guessing)")
    print("  ✓ Type 2 correct → +0.1 bonus (moderate reward)")
    print("  ✓ Type 2 incorrect → -0.3 penalty (moderate risk)")
    print("  ✓ Type 3 → neutral (safe baseline, no extra reward/risk)")
    print("  ✓ Length penalty still applies to all types")
    print("  ✓ NO diversity scaling (removed complexity)")
    print("="*80)
