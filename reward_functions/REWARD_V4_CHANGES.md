# Adaptive Reasoning Reward V4 - Changes Summary

## Overview

This document summarizes the key changes in `adaptive_reasoning_reward_v4.py` compared to the previous version.

## Key Changes

### 1. ✅ Removed Diversity Scaling

**Before:**
```python
diversity_scalar = calculate_diversity_scaling(response_type)
reward = (base_reward + format_bonus) × length_scalar × diversity_scalar
```

**After:**
```python
# No diversity scaling
reward = (base_reward + format_bonus) × length_scalar
```

**Rationale:** Simplifies the reward function and removes the complexity of tracking format distribution windows. The model should naturally learn the best format for each task without artificial diversity enforcement.

---

### 2. ✅ Reversed Format Bonus (Type 1 > Type 2 > Type 3)

**Before:**
```python
Type 1 (direct): 0.0
Type 2 (perception): 0.1
Type 3 (full reasoning): 0.2
```

**After:**
```python
Type 1 (direct): 0.2
Type 2 (perception): 0.1
Type 3 (full reasoning): 0.0
```

**Rationale:** Encourages efficient reasoning. Simple questions should use Type 1 for maximum reward. Complex questions can still use Type 3 as a safe baseline.

---

### 3. ✅ Added Error Penalties for Type 1 and Type 2

**New Feature:**
```python
Type 1 incorrect: -0.5 penalty
Type 2 incorrect: -0.3 penalty
Type 3 incorrect: 0.0 (no penalty, safe baseline)
```

**Implementation:**
```python
if is_correct:
    reward = (1.0 + format_bonus) × length_scalar
else:
    reward = (0.0 + error_penalty) × length_scalar
```

**Rationale:** Creates risk-reward tradeoff:
- Type 1: High risk, high reward (use when very confident)
- Type 2: Medium risk, medium reward (use when moderately confident)
- Type 3: No risk, baseline reward (use when uncertain or complex)

---

## Reward Examples

### Correct Answers (with length_scalar = 1.0)

| Type | Old Reward | New Reward | Change |
|------|-----------|-----------|--------|
| Type 1 | 1.0 | 1.2 | +20% |
| Type 2 | 1.1 | 1.1 | 0% |
| Type 3 | 1.2 | 1.0 | -16.7% |

### Incorrect Answers (with length_scalar = 1.0)

| Type | Old Reward | New Reward | Change |
|------|-----------|-----------|--------|
| Type 1 | 0.0 | -0.5 | -∞ |
| Type 2 | 0.0 | -0.3 | -∞ |
| Type 3 | 0.0 | 0.0 | 0% |

---

## Expected Model Behavior

### Simple Questions (e.g., "What color is the car?")
- **Old:** Model might use Type 3 for +0.2 bonus
- **New:** Model will use Type 1 for +0.2 bonus if confident, Type 3 if uncertain

### Medium Questions (e.g., "How many objects are on the table?")
- **Old:** Model prefers Type 3 for highest reward
- **New:** Model balances between Type 2 (+0.1, -0.3 risk) and Type 3 (safe baseline)

### Complex Questions (e.g., "Explain the relationship between objects")
- **Old:** Model uses Type 3 for +0.2 bonus
- **New:** Model uses Type 3 as safe baseline (no bonus, no penalty)

---

## Migration Guide

### Updating Training Configuration

**Old:**
```python
reward_fn_kwargs = {
    'type1_format_bonus': 0.0,
    'type2_format_bonus': 0.1,
    'type3_format_bonus': 0.2,
    'enable_diversity_scaling': True,
    'diversity_weight': 0.3,
}
```

**New:**
```python
reward_fn_kwargs = {
    'type1_format_bonus': 0.2,
    'type2_format_bonus': 0.1,
    'type3_format_bonus': 0.0,
    'type1_error_penalty': -0.5,
    'type2_error_penalty': -0.3,
    'type3_error_penalty': 0.0,
    # Diversity scaling parameters removed
}
```

### Importing the New Reward Function

```python
# Old
from reward_functions.adaptive_reasoning_reward import create_reward_function

# New
from reward_functions.adaptive_reasoning_reward_v4 import create_reward_function
```

---

## Metrics Changes

### Removed Metrics
- `reward/diversity_scalar_mean` (no longer tracked)
- `format/window_type{1,2,3}_ratio` (no longer tracked)
- `format/window_size` (no longer tracked)

### Kept Metrics
- `accuracy/overall`
- `format/type{1,2,3}_ratio`
- `format/type{1,2,3}_correct_rate`
- `format/type{1,2,3}_avg_length`
- `reward/base_mean`
- `reward/format_bonus_mean` (now includes error penalties)
- `reward/length_scalar_mean`
- `reward/total_mean`

---

## Testing Results

```
Type 1 Correct:  1.2 (was 1.0)  ✓ +20% reward
Type 1 Incorrect: -0.5 (was 0.0) ✗ penalty added
Type 2 Correct:  1.1 (was 1.1)  → unchanged
Type 2 Incorrect: -0.3 (was 0.0) ✗ penalty added
Type 3 Correct:  1.0 (was 1.2)  ↓ -16.7% reward
Type 3 Incorrect: 0.0 (was 0.0)  → unchanged (safe baseline)
```

---

## Design Philosophy

### V3 Philosophy
"Encourage complete reasoning (Type 3) while preventing format convergence through diversity scaling"

### V4 Philosophy
"Encourage efficient reasoning - use simple formats when confident, complex formats when needed, with risk-reward tradeoffs"

---

## Expected Training Dynamics

1. **Early Training:** Model will likely use Type 3 (safe baseline) due to low confidence
2. **Mid Training:** Model starts using Type 1/2 for easier questions where confidence is higher
3. **Late Training:** Model learns optimal format selection based on question complexity and confidence

The error penalties create a natural exploration-exploitation tradeoff without artificial diversity enforcement.

---

## File Locations

- **New V4:** `/nas03/yixuh/vlm-adaptive-resoning/reward_functions/adaptive_reasoning_reward_v4.py`
- **Old V3:** `/nas03/yixuh/vlm-adaptive-resoning/reward_functions/adaptive_reasoning_reward.py` (preserved)
- **This Doc:** `/nas03/yixuh/vlm-adaptive-resoning/reward_functions/REWARD_V4_CHANGES.md`
