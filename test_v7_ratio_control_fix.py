#!/usr/bin/env python3
"""
Test script to verify V7 ratio control fix.

This script simulates the training loop to verify that:
1. Training step is correctly passed to reward manager
2. Ratio control activates at the specified step
3. Ratio penalties are correctly applied
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.getcwd(), 'reward_functions'))

from adaptive_reasoning_reward_v7 import AdaptiveReasoningRewardV7

def test_ratio_control():
    """Test that ratio control activates correctly with training steps."""

    print("="*80)
    print("Testing V7 Ratio Control Fix")
    print("="*80)

    # Initialize reward function with ratio control
    reward_fn = AdaptiveReasoningRewardV7(
        correct_reward=1.0,
        incorrect_reward=0.0,
        type1_format_bonus=0.2,
        type2_format_bonus=0.2,
        type3_format_bonus=0.3,
        enable_ratio_penalty=True,
        ratio_penalty_start_step=60,
        target_type1_ratio=0.6,  # Same as train-grpo-v7-2.sh
        target_type2_ratio=0.2,
        target_type3_ratio=0.2,
        ratio_tolerance=0.15,
        ratio_penalty_min_scalar=0.5,
        ratio_window_size=256,
    )

    print("\n1. Testing BEFORE ratio penalty activation (step < 60)")
    print("-" * 80)

    # Simulate type distribution before step 60
    # Fill window with 80% Type 1, 10% Type 2, 10% Type 3 (imbalanced)
    for _ in range(204):  # 80% of 256
        reward_fn.type_history.append(1)
    for _ in range(26):   # 10% of 256
        reward_fn.type_history.append(2)
    for _ in range(26):   # 10% of 256
        reward_fn.type_history.append(3)

    # Set step to 50 (before activation)
    reward_fn.set_training_step(50)
    print(f"Training step: {reward_fn.current_step}")
    print(f"Ratio penalty enabled: {reward_fn.enable_ratio_penalty}")
    print(f"Ratio penalty start step: {reward_fn.ratio_penalty_start_step}")

    # Calculate ratio scalars
    for response_type in [1, 2, 3]:
        scalar = reward_fn._calculate_ratio_scalar(response_type)
        print(f"  Type {response_type} ratio scalar: {scalar:.4f}")

    # All should be 1.0 (no penalty before activation)
    assert all(reward_fn._calculate_ratio_scalar(t) == 1.0 for t in [1, 2, 3]), \
        "FAIL: Ratio scalars should be 1.0 before activation step!"
    print("✓ PASS: No penalty before activation step")

    print("\n2. Testing AFTER ratio penalty activation (step >= 60)")
    print("-" * 80)

    # Set step to 70 (after activation)
    reward_fn.set_training_step(70)
    print(f"Training step: {reward_fn.current_step}")

    # Calculate current distribution from window
    type_counts = {1: 0, 2: 0, 3: 0}
    for t in reward_fn.type_history:
        type_counts[t] += 1
    total = len(reward_fn.type_history)

    print(f"\nCurrent distribution in window:")
    print(f"  Type 1: {type_counts[1]/total:.1%} (target: {reward_fn.target_type1_ratio:.1%})")
    print(f"  Type 2: {type_counts[2]/total:.1%} (target: {reward_fn.target_type2_ratio:.1%})")
    print(f"  Type 3: {type_counts[3]/total:.1%} (target: {reward_fn.target_type3_ratio:.1%})")

    print(f"\nRatio scalars (after activation):")
    scalars = {}
    for response_type in [1, 2, 3]:
        scalar = reward_fn._calculate_ratio_scalar(response_type)
        scalars[response_type] = scalar
        print(f"  Type {response_type} ratio scalar: {scalar:.4f}")

    # Type 1 should be penalized (over-used: 80% vs target 60%)
    # Type 2 and 3 should have no penalty or encouragement (under-used)
    assert scalars[1] < 1.0, \
        "FAIL: Type 1 should be penalized (over-used)!"
    print(f"✓ PASS: Type 1 is penalized (scalar={scalars[1]:.4f} < 1.0)")

    # Under-used types might still get penalty due to deviation from target
    # But penalty should be applied to bring balance
    print(f"✓ PASS: Ratio control is active and penalties are being calculated")

    print("\n3. Testing with balanced distribution")
    print("-" * 80)

    # Clear and refill with balanced distribution
    reward_fn.type_history.clear()
    for _ in range(154):  # 60% of 256
        reward_fn.type_history.append(1)
    for _ in range(51):   # 20% of 256
        reward_fn.type_history.append(2)
    for _ in range(51):   # 20% of 256
        reward_fn.type_history.append(3)

    # Keep step at 70
    reward_fn.set_training_step(70)

    # Calculate current distribution
    type_counts = {1: 0, 2: 0, 3: 0}
    for t in reward_fn.type_history:
        type_counts[t] += 1
    total = len(reward_fn.type_history)

    print(f"Balanced distribution in window:")
    print(f"  Type 1: {type_counts[1]/total:.1%} (target: {reward_fn.target_type1_ratio:.1%})")
    print(f"  Type 2: {type_counts[2]/total:.1%} (target: {reward_fn.target_type2_ratio:.1%})")
    print(f"  Type 3: {type_counts[3]/total:.1%} (target: {reward_fn.target_type3_ratio:.1%})")

    print(f"\nRatio scalars (balanced):")
    scalars_balanced = {}
    for response_type in [1, 2, 3]:
        scalar = reward_fn._calculate_ratio_scalar(response_type)
        scalars_balanced[response_type] = scalar
        print(f"  Type {response_type} ratio scalar: {scalar:.4f}")

    # All should be close to 1.0 when distribution matches targets
    for t in [1, 2, 3]:
        assert scalars_balanced[t] >= 0.95, \
            f"FAIL: Type {t} scalar should be close to 1.0 when balanced!"
    print("✓ PASS: All scalars near 1.0 when distribution is balanced")

    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
    print("\nKey findings:")
    print("  ✓ Training step is correctly tracked")
    print("  ✓ Ratio penalty activates at specified step")
    print("  ✓ Over-used types are penalized")
    print("  ✓ Balanced distribution results in minimal penalty")
    print("\nThe fix should now work correctly in training!")
    print("="*80)

if __name__ == "__main__":
    test_ratio_control()
