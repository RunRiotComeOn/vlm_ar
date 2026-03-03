"""
Adaptive Reasoning Reward Function for GRPO Training

This reward function encourages the model to adaptively choose the reasoning mode:
- Type 1 (Direct Answer): Highest reward when correct
- Type 2 (Perception + Answer): Medium reward due to perception penalty
- Type 3 (Perception + Reasoning + Answer): Lower reward due to both penalties

The model learns to use the simplest approach that gets the answer correct.
"""

import re
from typing import List, Dict, Any


class AdaptiveReasoningReward:
    """
    Reward function that penalizes unnecessary reasoning steps.

    Reward Design:
    - Correct answer: +1.0 base score
    - Incorrect answer: 0.0
    - Has <perception> tag: -0.2 penalty
    - Has <reasoning> tag: -0.3 penalty

    Examples:
    - Type 1 (direct correct answer): 1.0
    - Type 2 (perception + correct answer): 1.0 - 0.2 = 0.8
    - Type 3 (perception + reasoning + correct answer): 1.0 - 0.2 - 0.3 = 0.5
    - Any incorrect answer: 0.0
    """

    def __init__(
        self,
        correct_reward: float = 1.0,
        perception_penalty: float = 0.2,
        reasoning_penalty: float = 0.3,
        incorrect_reward: float = 0.0,
        length_penalty_coef: float = 0.0,  # Optional: penalize overly long responses
        normalize_answers: bool = True,
    ):
        """
        Args:
            correct_reward: Base reward for correct answer
            perception_penalty: Penalty for using <perception> tag
            reasoning_penalty: Penalty for using <reasoning> tag
            incorrect_reward: Reward for incorrect answer
            length_penalty_coef: Coefficient for length penalty (tokens)
            normalize_answers: Whether to normalize answers before comparison
        """
        self.correct_reward = correct_reward
        self.perception_penalty = perception_penalty
        self.reasoning_penalty = reasoning_penalty
        self.incorrect_reward = incorrect_reward
        self.length_penalty_coef = length_penalty_coef
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

    def calculate_length_penalty(self, response: str) -> float:
        """Calculate penalty based on response length."""
        if self.length_penalty_coef == 0:
            return 0.0

        # Simple token count (split by whitespace)
        token_count = len(response.split())

        # Penalize responses longer than 200 tokens
        if token_count > 200:
            return self.length_penalty_coef * (token_count - 200) / 100.0

        return 0.0

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
            # Extract answer from response
            predicted_answer = self.extract_answer(response)

            # Check correctness
            is_correct = self.check_answer_correctness(predicted_answer, gt)

            # Base reward
            if is_correct:
                reward = self.correct_reward
            else:
                reward = self.incorrect_reward

            # Apply penalties only if answer is correct
            # (We want to encourage efficiency for correct answers)
            if is_correct:
                # Perception penalty
                if self.has_perception_tag(response):
                    reward -= self.perception_penalty

                # Reasoning penalty
                if self.has_reasoning_tag(response):
                    reward -= self.reasoning_penalty

                # Length penalty
                length_penalty = self.calculate_length_penalty(response)
                reward -= length_penalty

            # Ensure reward is non-negative
            reward = max(0.0, reward)

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
        is_correct = self.check_answer_correctness(predicted_answer, ground_truth)

        breakdown = {
            'base_reward': self.correct_reward if is_correct else self.incorrect_reward,
            'perception_penalty': -self.perception_penalty if (is_correct and self.has_perception_tag(response)) else 0.0,
            'reasoning_penalty': -self.reasoning_penalty if (is_correct and self.has_reasoning_tag(response)) else 0.0,
            'length_penalty': -self.calculate_length_penalty(response) if is_correct else 0.0,
            'is_correct': is_correct,
            'has_perception': self.has_perception_tag(response),
            'has_reasoning': self.has_reasoning_tag(response),
            'predicted_answer': predicted_answer,
        }

        breakdown['total_reward'] = sum([
            breakdown['base_reward'],
            breakdown['perception_penalty'],
            breakdown['reasoning_penalty'],
            breakdown['length_penalty']
        ])

        # Ensure non-negative
        breakdown['total_reward'] = max(0.0, breakdown['total_reward'])

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
            'description': 'Type 1 - Direct correct answer'
        },
        {
            'response': '<perception>The car in the image is red.</perception>\n\n<answer>Red</answer>',
            'gt': 'red',
            'expected_reward': 0.8,
            'description': 'Type 2 - Perception + correct answer'
        },
        {
            'response': '<perception>The car in the image is red.</perception>\n\n<reasoning>Based on the color observed, the car is red.</reasoning>\n\n<answer>Red</answer>',
            'gt': 'red',
            'expected_reward': 0.5,
            'description': 'Type 3 - Perception + Reasoning + correct answer'
        },
        {
            'response': 'Blue',
            'gt': 'red',
            'expected_reward': 0.0,
            'description': 'Incorrect answer'
        },
        {
            'response': '<perception>The car is blue.</perception>\n\n<reasoning>The color is blue.</reasoning>\n\n<answer>Blue</answer>',
            'gt': 'red',
            'expected_reward': 0.0,
            'description': 'Incorrect with reasoning (still gets 0)'
        },
    ]

    print("="*80)
    print("Adaptive Reasoning Reward Function Test")
    print("="*80)

    for test in test_cases:
        breakdown = reward_fn.get_reward_breakdown(test['response'], test['gt'])
        print(f"\n{test['description']}")
        print(f"Response: {test['response'][:100]}...")
        print(f"Ground Truth: {test['gt']}")
        print(f"Predicted: {breakdown['predicted_answer']}")
        print(f"Correct: {breakdown['is_correct']}")
        print(f"Has Perception: {breakdown['has_perception']}")
        print(f"Has Reasoning: {breakdown['has_reasoning']}")
        print(f"Reward Breakdown:")
        print(f"  Base: {breakdown['base_reward']:.2f}")
        print(f"  Perception Penalty: {breakdown['perception_penalty']:.2f}")
        print(f"  Reasoning Penalty: {breakdown['reasoning_penalty']:.2f}")
        print(f"  Total: {breakdown['total_reward']:.2f}")
        print(f"Expected: {test['expected_reward']:.2f}")
        print(f"✓ PASS" if abs(breakdown['total_reward'] - test['expected_reward']) < 0.01 else "✗ FAIL")

    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)
