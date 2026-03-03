"""
Adaptive Reasoning Reward Function V4 - With Comprehensive Metrics Logging

This reward function encourages complete reasoning while controlling verbosity and preventing
premature convergence to a single format:
- Format Bonus: Type 3 > Type 2 > Type 1 (encourages structured reasoning)
- Length Penalty: Dynamic scalar based on token count (prevents verbosity)
- Diversity Scaling: G / F(oi) (prevents convergence to single format)
- Comprehensive Metrics: Track format distribution, accuracy, length, and reward components

Design Philosophy:
- Simple questions with Type 3: Short answer → No penalty → High reward ✓
- Complex questions with Type 3: Long answer → Some penalty, but format bonus compensates ✓
- Overly verbose answers: Heavy penalty regardless of type ✗
- Overused formats: Reduced reward to encourage diversity ✓
- Full observability: All metrics logged to TensorBoard for analysis ✓
"""

import re
from typing import List, Dict, Any
from collections import deque
import numpy as np


class AdaptiveReasoningReward:
    """
    Reward function with format bonus, dynamic length penalty, diversity scaling, and metrics.

    Reward Formula:
        reward = (correctness_reward + format_bonus) × length_scalar × diversity_scalar

    Where:
        - correctness_reward: 1.0 if correct, 0.0 if incorrect
        - format_bonus: Type-specific bonus (Type 3 > Type 2 > Type 1)
        - length_scalar: Penalty for overly long responses
        - diversity_scalar: G / F(oi) - encourages format diversity
          * G: Total samples in window
          * F(oi): Count of this format type in window
          * Underused formats get boosted, overused get penalized

    Metrics Logged (for TensorBoard):
        - format/type{1,2,3}_ratio: % of each response type in batch
        - format/type{1,2,3}_correct_rate: Accuracy per type
        - format/type{1,2,3}_avg_length: Average token length per type
        - format/window_type{1,2,3}_ratio: Format distribution in sliding window
        - reward/base_mean: Mean base reward (correctness)
        - reward/format_bonus_mean: Mean format bonus
        - reward/length_scalar_mean: Mean length penalty
        - reward/diversity_scalar_mean: Mean diversity scaling
        - reward/total_mean: Mean total reward
        - accuracy/overall: Overall accuracy
    """

    def __init__(
        self,
        # Correctness rewards
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,

        # Format bonuses (Type 3 > Type 2 > Type 1)
        type1_format_bonus: float = 0.0,
        type2_format_bonus: float = 0.1,
        type3_format_bonus: float = 0.2,

        # Length penalty parameters
        length_threshold: int = 300,      # No penalty below this
        ideal_length: float = 300.0,      # Reference for penalty calculation
        min_scalar: float = 0.3,          # Minimum scalar (max penalty)

        # Format diversity parameters
        enable_diversity_scaling: bool = True,  # Enable diversity scaling
        diversity_window_size: int = 1000,      # Window size for tracking formats
        diversity_weight: float = 0.3,          # Weight for diversity scaling (0-1)
        min_samples_for_diversity: int = 100,   # Start diversity after N samples

        # Other settings
        normalize_answers: bool = True,
    ):
        """
        Args:
            correct_reward: Reward for correct answer
            incorrect_reward: Reward for incorrect answer
            type1_format_bonus: Bonus for Type 1 (direct answer)
            type2_format_bonus: Bonus for Type 2 (perception + answer)
            type3_format_bonus: Bonus for Type 3 (full reasoning)
            length_threshold: Token count below which there's no penalty
            ideal_length: Reference length for calculating scalar
            min_scalar: Minimum allowed scalar (prevents reward from going too low)
            enable_diversity_scaling: Whether to apply diversity scaling
            diversity_window_size: Size of sliding window for format tracking
            diversity_weight: How much to weight diversity (0=none, 1=full)
            min_samples_for_diversity: Min samples before applying diversity
            normalize_answers: Whether to normalize answers before comparison
        """
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.type1_format_bonus = type1_format_bonus
        self.type2_format_bonus = type2_format_bonus
        self.type3_format_bonus = type3_format_bonus
        self.length_threshold = length_threshold
        self.ideal_length = ideal_length
        self.min_scalar = min_scalar
        self.normalize_answers = normalize_answers

        # Format diversity tracking
        self.enable_diversity_scaling = enable_diversity_scaling
        self.diversity_window_size = diversity_window_size
        self.diversity_weight = diversity_weight
        self.min_samples_for_diversity = min_samples_for_diversity
        self.recent_formats = deque(maxlen=diversity_window_size)

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

        Example (threshold=200, ideal_length=300):
            - 100 tokens → 1.0 (no penalty)
            - 200 tokens → 1.0 (no penalty)
            - 300 tokens → 300/300 = 1.0 (no penalty)
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

    def calculate_diversity_scaling(self, format_type: int) -> float:
        """
        Calculate Format Diversity Scaling Factor: G / F(oi)

        This prevents premature convergence on a single format by:
        - Boosting rewards for underused formats
        - Reducing rewards for overused formats

        Args:
            format_type: The response type (1, 2, or 3)

        Returns:
            Scaling factor in range [0.5, 2.0]

        Formula:
            - G: Total samples in window (len(recent_formats))
            - F(oi): Count of format type oi in window
            - Expected proportion: 1/3 (uniform distribution)
            - Raw scaling = expected_proportion / actual_proportion
            - Final = 1.0 + diversity_weight * (raw_scaling - 1.0)

        Example:
            - Type 3 used 70% of time: scaling < 1.0 (penalize)
            - Type 3 used 10% of time: scaling > 1.0 (boost)
            - Type 3 used 33% of time: scaling ≈ 1.0 (neutral)
        """
        if not self.enable_diversity_scaling:
            return 1.0

        # Need enough samples for meaningful statistics
        if len(self.recent_formats) < self.min_samples_for_diversity:
            return 1.0

        # Count each format type in recent window
        format_counts = {1: 0, 2: 0, 3: 0}
        for fmt in self.recent_formats:
            format_counts[fmt] += 1

        total = len(self.recent_formats)

        # Expected proportion (uniform distribution across 3 types)
        expected_proportion = 1.0 / 3.0

        # Actual proportion for this format
        actual_count = format_counts[format_type]
        actual_proportion = actual_count / total

        # Calculate raw scaling factor
        if actual_proportion > 0:
            raw_scaling = expected_proportion / actual_proportion
        else:
            # Format never used → strongly boost it
            raw_scaling = 2.0

        # Apply diversity weight (0 = no diversity, 1 = full diversity)
        # scaling = 1 + weight * (raw_scaling - 1)
        scaling = 1.0 + self.diversity_weight * (raw_scaling - 1.0)

        # Clamp to reasonable range [0.5, 2.0]
        scaling = max(0.5, min(2.0, scaling))

        return scaling

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
            'diversity_scalars': [],
            'total_rewards': [],
        }

        for response, gt in zip(responses, ground_truths):
            # 1. Check correctness
            is_correct = self.check_answer_correctness(response, gt)
            base_reward = self.correct_reward if is_correct else self.incorrect_reward

            # 2. Determine response type and get format bonus
            response_type = self.get_response_type(response)
            if response_type == 3:
                format_bonus = self.type3_format_bonus
            elif response_type == 2:
                format_bonus = self.type2_format_bonus
            else:
                format_bonus = self.type1_format_bonus

            # 3. Calculate length scalar
            token_count = self.count_tokens(response)
            length_scalar = self.calculate_length_scalar(token_count)

            # 4. Calculate diversity scaling factor
            diversity_scalar = self.calculate_diversity_scaling(response_type)

            # 5. Final reward = (base + format) × length_scalar × diversity_scalar
            reward = (base_reward + format_bonus) * length_scalar * diversity_scalar

            # 6. Update format tracking (after reward calculation)
            self.recent_formats.append(response_type)

            # 7. Ensure non-negative
            reward = max(0.0, reward)

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
            batch_stats['format_bonuses'].append(format_bonus)
            batch_stats['length_scalars'].append(length_scalar)
            batch_stats['diversity_scalars'].append(diversity_scalar)
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
        - reward/format_bonus_mean: Mean format bonus
        - reward/length_scalar_mean: Mean length penalty
        - reward/diversity_scalar_mean: Mean diversity scaling
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
            metrics['reward/diversity_scalar_mean'] = float(np.mean(batch_stats['diversity_scalars']))
            metrics['reward/total_mean'] = float(np.mean(batch_stats['total_rewards']))

        # Overall accuracy
        if total_samples > 0:
            metrics['accuracy/overall'] = batch_stats['correct_count'] / total_samples

        # Format diversity window stats (for debugging)
        if len(self.recent_formats) > 0:
            format_counts = {1: 0, 2: 0, 3: 0}
            for fmt in self.recent_formats:
                format_counts[fmt] += 1
            total = len(self.recent_formats)
            metrics['format/window_type1_ratio'] = format_counts[1] / total
            metrics['format/window_type2_ratio'] = format_counts[2] / total
            metrics['format/window_type3_ratio'] = format_counts[3] / total
            metrics['format/window_size'] = float(total)

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

        if response_type == 3:
            format_bonus = self.type3_format_bonus
        elif response_type == 2:
            format_bonus = self.type2_format_bonus
        else:
            format_bonus = self.type1_format_bonus

        length_scalar = self.calculate_length_scalar(token_count)
        diversity_scalar = self.calculate_diversity_scaling(response_type)
        total_reward = (base_reward + format_bonus) * length_scalar * diversity_scalar
        total_reward = max(0.0, total_reward)

        # Get format distribution for debugging
        format_dist = {1: 0, 2: 0, 3: 0}
        if len(self.recent_formats) > 0:
            for fmt in self.recent_formats:
                format_dist[fmt] += 1
            total = len(self.recent_formats)
            format_dist = {k: v/total for k, v in format_dist.items()}

        return {
            'total_reward': total_reward,
            'is_correct': is_correct,
            'response_type': response_type,
            'base_reward': base_reward,
            'format_bonus': format_bonus,
            'token_count': token_count,
            'length_scalar': length_scalar,
            'diversity_scalar': diversity_scalar,
            'format_distribution': format_dist,
            'window_size': len(self.recent_formats),
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
        **init_kwargs: Initialization arguments for AdaptiveReasoningReward

    Returns:
        float: Reward score
    """
    global _reward_instance

    if _reward_instance is None:
        _reward_instance = AdaptiveReasoningReward(**init_kwargs)

    rewards = _reward_instance([solution_str], [ground_truth])
    return rewards[0]


# Example usage and testing
if __name__ == "__main__":
    reward_fn = AdaptiveReasoningReward(
        type2_format_bonus=0.1,
        type3_format_bonus=0.2,
        enable_diversity_scaling=True,
        diversity_weight=0.3,
    )

    # Test cases
    test_cases = [
        {
            'response': 'Red',
            'gt': 'red',
            'description': 'Type 1 - Simple correct answer',
        },
        {
            'response': '<perception>I see a red car in the image.</perception>\n\n<answer>Red</answer>',
            'gt': 'red',
            'description': 'Type 2 - Perception + answer',
        },
        {
            'response': '<perception>The image shows a red sports car.</perception>\n\n<reasoning>Based on the color I observe, the car is clearly red.</reasoning>\n\n<answer>Red</answer>',
            'gt': 'red',
            'description': 'Type 3 - Full reasoning',
        },
        {
            'response': '<perception>' + ' '.join(['The car is red.'] * 50) + '</perception>\n\n<reasoning>' + ' '.join(['It is red.'] * 50) + '</reasoning>\n\n<answer>Red</answer>',
            'gt': 'red',
            'description': 'Type 3 - Overly verbose',
        },
        {
            'response': 'Blue',
            'gt': 'red',
            'description': 'Type 1 - Incorrect answer',
        },
    ]

    print("="*80)
    print("Adaptive Reasoning Reward Function V4")
    print("Format Bonus + Dynamic Length Penalty + Diversity Scaling + Metrics")
    print("="*80)
    print("\nReward Design:")
    print("  Formula: reward = (correctness + format_bonus) × length_scalar × diversity_scalar")
    print(f"  Format Bonus: Type1={reward_fn.type1_format_bonus}, Type2={reward_fn.type2_format_bonus}, Type3={reward_fn.type3_format_bonus}")
    print(f"  Length Penalty: threshold={reward_fn.length_threshold}, ideal={reward_fn.ideal_length}")
    print(f"  Diversity Scaling: enabled={reward_fn.enable_diversity_scaling}, weight={reward_fn.diversity_weight}")
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

    print(f"\nIndividual Rewards:")
    for i, (tc, reward) in enumerate(zip(test_cases, rewards)):
        print(f"  {i+1}. {tc['description']}: {reward:.4f}")

    # Simulate format bias (add many Type 1 samples)
    print("\n📊 Simulating format bias (adding 80 Type 1 samples)...")
    for _ in range(80):
        reward_fn.recent_formats.append(1)

    for test in test_cases:
        breakdown = reward_fn.get_reward_breakdown(test['response'], test['gt'])
        print(f"\n{test['description']}")
        print(f"  Response: {test['response'][:60]}...")
        print(f"  Correct: {breakdown['is_correct']}, Type: {breakdown['response_type']}, Tokens: {breakdown['token_count']}")
        print(f"  Base: {breakdown['base_reward']:.2f}, Format: {breakdown['format_bonus']:.2f}, Length: {breakdown['length_scalar']:.2f}, Diversity: {breakdown['diversity_scalar']:.2f}")

        if breakdown['window_size'] >= 100:
            dist = breakdown['format_distribution']
            print(f"  📈 Format Dist: T1={dist[1]:.1%}, T2={dist[2]:.1%}, T3={dist[3]:.1%}")

        print(f"  ✨ Total Reward: {breakdown['total_reward']:.2f}")

    print("\n" + "="*80)
    print("Key Insights:")
    print("  • Type 1 is overused (80%) → diversity_scalar < 1.0 → reward reduced")
    print("  • Type 2/3 are underused → diversity_scalar > 1.0 → reward boosted")
    print("  • This encourages the model to explore different formats!")
    print("  • All metrics are logged to TensorBoard for monitoring!")
    print("="*80)
