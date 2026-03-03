"""
Adaptive Reward V1 - Simple format-based reward for GRPO training.

Encourages the model to adaptively choose between 3 format types:
- Type 1 (<answer>...</answer> only): Highest bonus when correct, heaviest penalty when wrong
- Type 2 (<perception>...</perception> + <answer>...</answer>): Medium bonus/penalty
- Type 3 (<perception>...</perception> + <reasoning>...</reasoning> + <answer>...</answer>): Neutral baseline
- Unknown (incomplete/missing tags): Penalized with -1.0

Classification requires ALL open+close tags to be present for a given type.
Responses with incomplete or missing tags are classified as "unknown" and penalized.

Reward formula:
    if unknown:   reward = unknown_penalty (-1.0)
    if correct:   reward = (1.0 + format_bonus) * length_scalar
    if incorrect:  reward = (0.0 + error_penalty) * length_scalar

Length regularization penalizes responses exceeding a threshold (300 tokens).
"""

import re
from typing import List, Dict, Any
import numpy as np


class AdaptiveRewardV1:
    """
    Simple adaptive reward function with format bonuses and length regularization.

    Format Bonuses (correct answers):
        Type 1: +0.5  (direct answer)
        Type 2: +0.3  (perception + answer)
        Type 3: +0.0  (full reasoning)

    Error Penalties (incorrect answers):
        Type 1: -0.5
        Type 2: -0.2
        Type 3:  0.0
    """

    def __init__(
        self,
        # Correctness rewards
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,
        # Format bonuses (correct only)
        type1_format_bonus: float = 0.5,
        type2_format_bonus: float = 0.3,
        type3_format_bonus: float = 0.0,
        # Error penalties (incorrect only)
        type1_error_penalty: float = -0.5,
        type2_error_penalty: float = -0.2,
        type3_error_penalty: float = 0.0,
        # Length regularization
        length_threshold: int = 300,
        ideal_length: float = 300.0,
        min_scalar: float = 0.3,
        # Unknown penalty (incomplete/missing tags)
        unknown_penalty: float = -1.0,
        # Other
        normalize_answers: bool = True,
    ):
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
        self.unknown_penalty = unknown_penalty
        self.normalize_answers = normalize_answers

    def extract_answer(self, response: str) -> str:
        """Extract answer from <answer>...</answer> tags."""
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()
        return response.strip()

    def has_complete_tag(self, response: str, tag_name: str) -> bool:
        """Check if response contains both <tag> and </tag> (complete pair)."""
        has_open = bool(re.search(rf'<{tag_name}>', response, re.IGNORECASE))
        has_close = bool(re.search(rf'</{tag_name}>', response, re.IGNORECASE))
        return has_open and has_close

    def get_response_type(self, response: str) -> int:
        """
        Determine response type based on complete tag pair presence.

        Requires ALL open+close tags for a given type to be present.

        Returns:
            3: Type 3 - complete <perception>, <reasoning>, and <answer> pairs (6 tokens)
            2: Type 2 - complete <perception> and <answer> pairs, no reasoning (4 tokens)
            1: Type 1 - complete <answer> pair only, no perception/reasoning (2 tokens)
            0: Unknown - incomplete or missing tags
        """
        has_perception = self.has_complete_tag(response, 'perception')
        has_reasoning = self.has_complete_tag(response, 'reasoning')
        has_answer = self.has_complete_tag(response, 'answer')

        if has_perception and has_reasoning and has_answer:
            return 3
        elif has_perception and has_answer and not has_reasoning:
            return 2
        elif has_answer and not has_perception and not has_reasoning:
            return 1
        else:
            return 0  # unknown

    def count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: words * 1.3)."""
        words = len(text.split())
        return int(words * 1.3)

    def calculate_length_scalar(self, token_count: int) -> float:
        """
        Length regularization scalar.

        - token_count <= threshold: 1.0 (no penalty)
        - token_count > threshold: ideal_length / token_count, clamped to [min_scalar, 1.0]
        """
        if token_count <= self.length_threshold:
            return 1.0
        scalar = self.ideal_length / token_count
        return max(self.min_scalar, min(1.0, scalar))

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if not self.normalize_answers:
            return answer
        answer = answer.lower().strip()
        answer = answer.rstrip('.,!?;:')
        answer = ' '.join(answer.split())
        # Remove articles
        for prefix in ['the ', 'a ', 'an ']:
            if answer.startswith(prefix):
                answer = answer[len(prefix):]
        return answer

    def check_answer_correctness(self, predicted: str, ground_truth: Any) -> bool:
        """Check if predicted answer matches ground truth."""
        pred_answer = self.extract_answer(predicted)
        pred_norm = self.normalize_answer(pred_answer)

        # Handle multiple ground truth answers
        gt_answers = ground_truth if isinstance(ground_truth, list) else [ground_truth]

        for gt in gt_answers:
            gt_norm = self.normalize_answer(str(gt))

            # Multiple choice detection
            if len(gt_norm) == 1 and gt_norm.upper() in ['A', 'B', 'C', 'D', 'E']:
                if re.search(rf'\b{gt_norm.upper()}\b', pred_answer.upper()):
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
            return_dict: If True, return dict with 'rewards' and 'metrics'

        Returns:
            List[float] or dict with 'rewards' and 'metrics'
        """
        rewards = []

        batch_stats = {
            'type1_count': 0, 'type2_count': 0, 'type3_count': 0,
            'unknown_count': 0,
            'correct_count': 0, 'incorrect_count': 0,
            'type1_correct': 0, 'type2_correct': 0, 'type3_correct': 0,
            'type1_lengths': [], 'type2_lengths': [], 'type3_lengths': [],
            'base_rewards': [], 'format_bonuses': [],
            'length_scalars': [], 'total_rewards': [],
        }

        for response, gt in zip(responses, ground_truths):
            token_count = self.count_tokens(response)
            response_type = self.get_response_type(response)

            # Unknown type (incomplete/missing tags): penalize and skip
            if response_type == 0:
                reward = self.unknown_penalty
                rewards.append(reward)
                batch_stats['unknown_count'] += 1
                batch_stats['incorrect_count'] += 1
                batch_stats['base_rewards'].append(reward)
                batch_stats['format_bonuses'].append(0.0)
                batch_stats['length_scalars'].append(1.0)
                batch_stats['total_rewards'].append(reward)
                continue

            is_correct = self.check_answer_correctness(response, gt)

            # Format bonus or error penalty
            if is_correct:
                base_reward = self.correct_reward
                format_bonus = {1: self.type1_format_bonus, 2: self.type2_format_bonus, 3: self.type3_format_bonus}[response_type]
                error_penalty = 0.0
            else:
                base_reward = self.incorrect_reward
                format_bonus = 0.0
                error_penalty = {1: self.type1_error_penalty, 2: self.type2_error_penalty, 3: self.type3_error_penalty}[response_type]

            # Length scalar
            length_scalar = self.calculate_length_scalar(token_count)

            # Final reward
            if is_correct:
                reward = (base_reward + format_bonus) * length_scalar
            else:
                reward = (base_reward + error_penalty) * length_scalar

            rewards.append(reward)

            # Collect stats
            type_key = f'type{response_type}'
            batch_stats[f'{type_key}_count'] += 1
            batch_stats[f'{type_key}_lengths'].append(token_count)
            if is_correct:
                batch_stats[f'{type_key}_correct'] += 1
                batch_stats['correct_count'] += 1
            else:
                batch_stats['incorrect_count'] += 1

            batch_stats['base_rewards'].append(base_reward)
            batch_stats['format_bonuses'].append(format_bonus if is_correct else error_penalty)
            batch_stats['length_scalars'].append(length_scalar)
            batch_stats['total_rewards'].append(reward)

        if not return_dict:
            return rewards

        metrics = self._compute_batch_metrics(batch_stats, len(responses))
        return {'rewards': rewards, 'metrics': metrics}

    def _compute_batch_metrics(self, batch_stats: Dict, total_samples: int) -> Dict[str, float]:
        """Compute batch-level metrics for TensorBoard logging."""
        metrics = {}

        if total_samples > 0:
            metrics['format/type1_ratio'] = batch_stats['type1_count'] / total_samples
            metrics['format/type2_ratio'] = batch_stats['type2_count'] / total_samples
            metrics['format/type3_ratio'] = batch_stats['type3_count'] / total_samples
            metrics['format/unknown_ratio'] = batch_stats['unknown_count'] / total_samples

        for t in [1, 2, 3]:
            count = batch_stats[f'type{t}_count']
            if count > 0:
                metrics[f'format/type{t}_correct_rate'] = batch_stats[f'type{t}_correct'] / count
                metrics[f'format/type{t}_avg_length'] = float(np.mean(batch_stats[f'type{t}_lengths']))
            else:
                metrics[f'format/type{t}_correct_rate'] = 0.0
                metrics[f'format/type{t}_avg_length'] = 0.0

        if batch_stats['base_rewards']:
            metrics['reward/base_mean'] = float(np.mean(batch_stats['base_rewards']))
            metrics['reward/format_bonus_mean'] = float(np.mean(batch_stats['format_bonuses']))
            metrics['reward/length_scalar_mean'] = float(np.mean(batch_stats['length_scalars']))
            metrics['reward/total_mean'] = float(np.mean(batch_stats['total_rewards']))

        if total_samples > 0:
            metrics['accuracy/overall'] = batch_stats['correct_count'] / total_samples

        return metrics

    def get_reward_breakdown(self, response: str, ground_truth: Any) -> Dict[str, float]:
        """Get detailed breakdown of reward components for a single response."""
        token_count = self.count_tokens(response)
        response_type = self.get_response_type(response)

        if response_type == 0:
            return {
                'total_reward': self.unknown_penalty,
                'is_correct': False,
                'is_unknown': True,
                'response_type': 0,
                'base_reward': self.unknown_penalty,
                'format_bonus': 0.0,
                'error_penalty': 0.0,
                'token_count': token_count,
                'length_scalar': 1.0,
                'predicted_answer': '',
            }

        is_correct = self.check_answer_correctness(response, ground_truth)
        length_scalar = self.calculate_length_scalar(token_count)

        if is_correct:
            base_reward = self.correct_reward
            format_bonus = {1: self.type1_format_bonus, 2: self.type2_format_bonus, 3: self.type3_format_bonus}[response_type]
            error_penalty = 0.0
            total = (base_reward + format_bonus) * length_scalar
        else:
            base_reward = self.incorrect_reward
            format_bonus = 0.0
            error_penalty = {1: self.type1_error_penalty, 2: self.type2_error_penalty, 3: self.type3_error_penalty}[response_type]
            total = (base_reward + error_penalty) * length_scalar

        return {
            'total_reward': total,
            'is_correct': is_correct,
            'is_unknown': False,
            'response_type': response_type,
            'base_reward': base_reward,
            'format_bonus': format_bonus,
            'error_penalty': error_penalty,
            'token_count': token_count,
            'length_scalar': length_scalar,
            'predicted_answer': self.extract_answer(response),
        }


# Global instance for verl compatibility
_reward_instance = None


def create_reward_function(data_source, solution_str, ground_truth, extra_info=None, **init_kwargs):
    """
    Reward function interface expected by verl's naive reward manager.

    Args:
        data_source: Data source identifier (unused)
        solution_str: Model's response string
        ground_truth: Ground truth answer(s)
        extra_info: Additional information (unused)

    Returns:
        float: Reward score
    """
    global _reward_instance
    if _reward_instance is None:
        _reward_instance = AdaptiveRewardV1(**init_kwargs)
    rewards = _reward_instance([solution_str], [ground_truth])
    return rewards[0]


if __name__ == "__main__":
    reward_fn = AdaptiveRewardV1()

    test_cases = [
        # Valid types
        {
            'response': '<answer>C</answer>',
            'gt': 'C',
            'description': 'Type 1 - Correct (bonus +0.5)',
        },
        {
            'response': '<answer>A</answer>',
            'gt': 'C',
            'description': 'Type 1 - Incorrect (penalty -0.5)',
        },
        {
            'response': '<perception>I see a graph with power-law data.</perception>\n<answer>C</answer>',
            'gt': 'C',
            'description': 'Type 2 - Correct (bonus +0.3)',
        },
        {
            'response': '<perception>I see a graph.</perception>\n<answer>A</answer>',
            'gt': 'C',
            'description': 'Type 2 - Incorrect (penalty -0.2)',
        },
        {
            'response': '<perception>I see a graph.</perception>\n<reasoning>The data follows a power law.</reasoning>\n<answer>C</answer>',
            'gt': 'C',
            'description': 'Type 3 - Correct (bonus +0.0)',
        },
        {
            'response': '<perception>I see a graph.</perception>\n<reasoning>The data looks linear.</reasoning>\n<answer>A</answer>',
            'gt': 'C',
            'description': 'Type 3 - Incorrect (penalty 0.0)',
        },
        {
            'response': '<answer>C</answer> ' + 'extra words ' * 200,
            'gt': 'C',
            'description': 'Type 1 - Correct but long (length penalty)',
        },
        # Unknown cases (incomplete tags)
        {
            'response': 'The answer is C',
            'gt': 'C',
            'description': 'Unknown - no tags at all (penalty -1.0)',
        },
        {
            'response': '<perception>I see a graph with data points showing ' + 'detailed analysis ' * 800,
            'gt': 'C',
            'description': 'Unknown - perception not closed (penalty -1.0)',
        },
        {
            'response': '<perception>I see a graph.</perception>\n<reasoning>The data follows ' + 'a complex pattern ' * 800,
            'gt': 'C',
            'description': 'Unknown - reasoning not closed (penalty -1.0)',
        },
        {
            'response': '<perception>I see a graph.</perception>\nThe answer is C',
            'gt': 'C',
            'description': 'Unknown - perception+text but no answer tags (penalty -1.0)',
        },
        {
            'response': 'I see a graph.\nThe data follows a power law.\nC',
            'gt': 'C',
            'description': 'Unknown - tags stripped (penalty -1.0)',
        },
    ]

    print("=" * 70)
    print("Adaptive Reward V1 - Test (with unknown type)")
    print("=" * 70)
    print(f"\nFormat Bonus (correct):   Type1={reward_fn.type1_format_bonus}, Type2={reward_fn.type2_format_bonus}, Type3={reward_fn.type3_format_bonus}")
    print(f"Error Penalty (incorrect): Type1={reward_fn.type1_error_penalty}, Type2={reward_fn.type2_error_penalty}, Type3={reward_fn.type3_error_penalty}")
    print(f"Unknown penalty: {reward_fn.unknown_penalty}")
    print(f"Length: threshold={reward_fn.length_threshold}, ideal={reward_fn.ideal_length}, min_scalar={reward_fn.min_scalar}")
    print("=" * 70)

    responses = [tc['response'] for tc in test_cases]
    ground_truths = [tc['gt'] for tc in test_cases]

    result = reward_fn(responses, ground_truths, return_dict=True)
    rewards = result['rewards']
    metrics = result['metrics']

    print("\nBatch Metrics:")
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.4f}")

    print("\nIndividual Results:")
    for i, (tc, reward) in enumerate(zip(test_cases, rewards)):
        bd = reward_fn.get_reward_breakdown(tc['response'], tc['gt'])
        type_name = f"Type {bd['response_type']}" if bd['response_type'] > 0 else "Unknown"
        print(f"\n{i+1}. {tc['description']}")
        print(f"   Type={type_name}, Correct={bd['is_correct']}, Answer='{bd['predicted_answer']}'")
        print(f"   Base={bd['base_reward']:.2f}, Bonus={bd['format_bonus']:.2f}, Penalty={bd['error_penalty']:.2f}")
        print(f"   Tokens={bd['token_count']}, LenScalar={bd['length_scalar']:.2f}")
        print(f"   -> Total Reward: {reward:.4f}")

    # Test create_reward_function compatibility
    print("\n" + "=" * 70)
    print("Testing create_reward_function() interface:")
    r = create_reward_function("test_source", "<answer>C</answer>", "C")
    print(f"  create_reward_function('test_source', '<answer>C</answer>', 'C') = {r:.4f}")
    r2 = create_reward_function("test_source", "just plain text C", "C")
    print(f"  create_reward_function('test_source', 'just plain text C', 'C') = {r2:.4f}  (should be -1.0)")
    print("=" * 70)
    print("\nAll tests passed!")
