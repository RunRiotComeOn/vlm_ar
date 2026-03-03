"""
Adaptive Reasoning Reward Function V2 - Error Penalty Differentiation

This reward function uses differentiated error penalties to encourage adaptive reasoning:
- Type 1 (Direct Answer): High reward when correct, but NEGATIVE penalty when wrong
- Type 2 (Perception + Answer): Medium reward when correct, NO penalty when wrong
- Type 3 (Perception + Reasoning + Answer): Lower reward when correct, NO penalty when wrong

The key insight: The model learns to use Type 1 only when confident, and falls back to
Type 2/3 when uncertain, avoiding the negative penalty.
"""

import re
from typing import List, Dict, Any


class AdaptiveReasoningReward:
    """
    Reward function with differentiated error penalties.

    Reward Design:
    Type 1 (no tags):
        - Correct: +1.0
        - Incorrect: -0.5 (negative penalty discourages blind guessing)

    Type 2 (perception only):
        - Correct: +0.8 (slight penalty for perception overhead)
        - Incorrect: 0.0 (no penalty - at least model tried to perceive)

    Type 3 (perception + reasoning):
        - Correct: +0.7 (penalty for both perception and reasoning overhead)
        - Incorrect: 0.0 (no penalty - model did proper reasoning)

    This encourages:
    - Using Type 1 for simple/confident cases (highest reward)
    - Using Type 2/3 for uncertain cases (avoids negative penalty)
    - Adaptive behavior based on question difficulty
    """

    def __init__(
        self,
        # Type 1 rewards
        type1_correct_reward: float = 1.0,
        type1_incorrect_penalty: float = -0.5,

        # Type 2 rewards (perception only)
        type2_correct_reward: float = 0.8,
        type2_incorrect_reward: float = 0.0,

        # Type 3 rewards (perception + reasoning)
        type3_correct_reward: float = 0.7,
        type3_incorrect_reward: float = 0.0,

        # Other settings
        normalize_answers: bool = True,
    ):
        """
        Args:
            type1_correct_reward: Reward for Type 1 correct answer
            type1_incorrect_penalty: Penalty for Type 1 incorrect (should be negative)
            type2_correct_reward: Reward for Type 2 correct answer
            type2_incorrect_reward: Reward for Type 2 incorrect (usually 0)
            type3_correct_reward: Reward for Type 3 correct answer
            type3_incorrect_reward: Reward for Type 3 incorrect (usually 0)
            normalize_answers: Whether to normalize answers before comparison
        """
        self.type1_correct_reward = type1_correct_reward
        self.type1_incorrect_penalty = type1_incorrect_penalty
        self.type2_correct_reward = type2_correct_reward
        self.type2_incorrect_reward = type2_incorrect_reward
        self.type3_correct_reward = type3_correct_reward
        self.type3_incorrect_reward = type3_incorrect_reward
        self.normalize_answers = normalize_answers

    def extract_answer(self, response: str) -> str:
        """
        Extract answer from response.

        Priority:
        1. Content within <answer></answer> tags
        2. Full response if no tags
        """
        # Try to extract from <answer> tags
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()

        # Otherwise, use full response
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

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if not self.normalize_answers:
            return answer

        # Convert to lowercase
        answer = answer.lower().strip()

        # Remove punctuation at the end
        answer = answer.rstrip('.,!?;:')

        # Remove extra whitespace
        answer = ' '.join(answer.split())

        # Handle common variations
        # Remove "the" at the beginning
        if answer.startswith('the '):
            answer = answer[4:]

        # Remove "a" or "an" at the beginning
        if answer.startswith('a '):
            answer = answer[2:]
        if answer.startswith('an '):
            answer = answer[3:]

        return answer

    def check_answer_correctness(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if predicted answer matches ground truth.

        Supports:
        - Multiple choice questions (single letter A/B/C/D)
        - Exact match (after normalization)
        - Multiple ground truth answers (list)
        - Fuzzy matching for numerical answers
        """
        # Extract answer from <answer> tags if present
        pred_answer = self.extract_answer(predicted)
        pred_norm = self.normalize_answer(pred_answer)

        # Handle multiple ground truth answers
        if isinstance(ground_truth, list):
            gt_answers = ground_truth
        else:
            gt_answers = [ground_truth]

        for gt in gt_answers:
            gt_norm = self.normalize_answer(str(gt))

            # === Multiple Choice Question Detection ===
            # If ground truth is a single letter (A/B/C/D/E), it's an MCQ
            if len(gt_norm) == 1 and gt_norm.upper() in ['A', 'B', 'C', 'D', 'E']:
                # For MCQ, check if the option letter appears in the prediction
                # Match patterns like "A", "A.", "A)", "option A", etc.
                import re
                pattern = rf'\b{gt_norm.upper()}\b'
                if re.search(pattern, pred_answer.upper()):
                    return True
                # If the letter doesn't appear, it's wrong
                continue  # Try next ground truth if multiple

            # === Regular matching for open-ended questions ===
            # Exact match
            if pred_norm == gt_norm:
                return True

            # Check if predicted answer contains ground truth
            if gt_norm in pred_norm or pred_norm in gt_norm:
                return True

            # Try numerical comparison
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
        **kwargs
    ) -> List[float]:
        """
        Calculate rewards for a batch of responses.

        Args:
            responses: List of model responses
            ground_truths: List of ground truth answers
            **kwargs: Additional arguments (ignored)

        Returns:
            List of reward scores
        """
        rewards = []

        for response, gt in zip(responses, ground_truths):
            # Check correctness
            is_correct = self.check_answer_correctness(response, gt)

            # Determine response type
            response_type = self.get_response_type(response)

            # Calculate reward based on type and correctness
            if response_type == 1:  # Type 1: Direct answer
                reward = self.type1_correct_reward if is_correct else self.type1_incorrect_penalty
            elif response_type == 2:  # Type 2: Perception + answer
                reward = self.type2_correct_reward if is_correct else self.type2_incorrect_reward
            else:  # Type 3: Perception + reasoning + answer
                reward = self.type3_correct_reward if is_correct else self.type3_incorrect_reward

            rewards.append(reward)

        return rewards

    def get_reward_breakdown(
        self,
        response: str,
        ground_truth: Any
    ) -> Dict[str, float]:
        """
        Get detailed breakdown of reward components for a single response.
        Useful for debugging and analysis.
        """
        predicted_answer = self.extract_answer(response)
        is_correct = self.check_answer_correctness(response, ground_truth)
        response_type = self.get_response_type(response)

        # Calculate reward
        if response_type == 1:
            reward = self.type1_correct_reward if is_correct else self.type1_incorrect_penalty
        elif response_type == 2:
            reward = self.type2_correct_reward if is_correct else self.type2_incorrect_reward
        else:
            reward = self.type3_correct_reward if is_correct else self.type3_incorrect_reward

        breakdown = {
            'total_reward': reward,
            'is_correct': is_correct,
            'response_type': response_type,
            'has_perception': self.has_perception_tag(response),
            'has_reasoning': self.has_reasoning_tag(response),
            'predicted_answer': predicted_answer,
        }

        return breakdown


# Global reward instance (initialized once)
_reward_instance = None


def create_reward_function(data_source, solution_str, ground_truth, extra_info=None, **init_kwargs):
    """
    Reward function that computes score for a single response.
    This is the interface expected by verl.

    Args:
        data_source: Data source identifier (unused in this implementation)
        solution_str: Model's response string
        ground_truth: Ground truth answer(s)
        extra_info: Additional information (unused)
        **init_kwargs: Initialization arguments for AdaptiveReasoningReward
                       (only used on first call to create the instance)

    Returns:
        float: Reward score
    """
    global _reward_instance

    # Initialize reward instance on first call
    if _reward_instance is None:
        _reward_instance = AdaptiveReasoningReward(**init_kwargs)

    # Compute reward for single response
    rewards = _reward_instance([solution_str], [ground_truth])
    return rewards[0]


# Example usage
if __name__ == "__main__":
    reward_fn = AdaptiveReasoningReward()

    # Test cases
    test_cases = [
        {
            'response': 'Red',
            'gt': 'red',
            'expected_reward': 1.0,
            'description': 'Type 1 - Direct correct answer (highest reward)'
        },
        {
            'response': 'Blue',
            'gt': 'red',
            'expected_reward': -0.5,
            'description': 'Type 1 - Direct incorrect answer (NEGATIVE penalty)'
        },
        {
            'response': '<perception>The car in the image is red.</perception>\n\n<answer>Red</answer>',
            'gt': 'red',
            'expected_reward': 0.8,
            'description': 'Type 2 - Perception + correct answer'
        },
        {
            'response': '<perception>The car appears to be blue.</perception>\n\n<answer>Blue</answer>',
            'gt': 'red',
            'expected_reward': 0.0,
            'description': 'Type 2 - Perception + incorrect answer (no penalty)'
        },
        {
            'response': '<perception>The car in the image is red.</perception>\n\n<reasoning>Based on the color observed, the car is red.</reasoning>\n\n<answer>Red</answer>',
            'gt': 'red',
            'expected_reward': 0.7,
            'description': 'Type 3 - Full reasoning + correct answer'
        },
        {
            'response': '<perception>The car is blue.</perception>\n\n<reasoning>The color is blue.</reasoning>\n\n<answer>Blue</answer>',
            'gt': 'red',
            'expected_reward': 0.0,
            'description': 'Type 3 - Full reasoning + incorrect answer (no penalty)'
        },
    ]

    print("="*80)
    print("Adaptive Reasoning Reward Function V2 - Error Penalty Differentiation")
    print("="*80)
    print("\nKey Design:")
    print("  Type 1 Correct:   +1.0  (highest reward - use when confident)")
    print("  Type 1 Incorrect: -0.5  (NEGATIVE penalty - discourages guessing)")
    print("  Type 2 Correct:   +0.8  (slight perception overhead)")
    print("  Type 2 Incorrect:  0.0  (no penalty - at least tried to perceive)")
    print("  Type 3 Correct:   +0.7  (perception + reasoning overhead)")
    print("  Type 3 Incorrect:  0.0  (no penalty - did proper reasoning)")
    print("="*80)

    all_passed = True
    for test in test_cases:
        breakdown = reward_fn.get_reward_breakdown(test['response'], test['gt'])
        print(f"\n{test['description']}")
        print(f"Response: {test['response'][:80]}...")
        print(f"Ground Truth: {test['gt']}")
        print(f"Predicted: {breakdown['predicted_answer']}")
        print(f"Correct: {breakdown['is_correct']}")
        print(f"Response Type: {breakdown['response_type']}")
        print(f"Total Reward: {breakdown['total_reward']:.2f}")
        print(f"Expected: {test['expected_reward']:.2f}")

        passed = abs(breakdown['total_reward'] - test['expected_reward']) < 0.01
        print(f"{'✓ PASS' if passed else '✗ FAIL'}")
        if not passed:
            all_passed = False

    print("\n" + "="*80)
    print(f"{'All tests PASSED! ✓' if all_passed else 'Some tests FAILED! ✗'}")
    print("="*80)
