# Stage 2: GRPO Training for Adaptive Reasoning

This guide covers Stage 2 training using GRPO (Group Relative Policy Optimization) to teach the model to **adaptively** choose the appropriate reasoning mode for each question.

## Overview

### Training Pipeline

```
┌─────────────┐      ┌─────────────┐      ┌──────────────────┐
│  Stage 1:   │  =>  │  Stage 2:   │  =>  │  Adaptive Model  │
│  SFT on     │      │  GRPO with  │      │  Chooses best    │
│  Type 1,2,3 │      │  Adaptive   │      │  reasoning mode  │
│             │      │  Reward     │      │  per question    │
└─────────────┘      └─────────────┘      └──────────────────┘
```

### Goal

Train the model to **intelligently select** between three reasoning modes:

| Mode | Format | Best For | Reward (if correct) |
|------|--------|----------|---------------------|
| **Type 1** | Direct Answer | Simple questions | 1.0 (highest) |
| **Type 2** | Perception + Answer | Needs description | 0.8 (medium) |
| **Type 3** | Perception + Reasoning + Answer | Complex problems | 0.5 (lower) |

The model learns: *"Use the simplest approach that gets the answer right"*

## Reward Function Design

### Adaptive Reasoning Reward

```python
Reward = Base Score - Perception Penalty - Reasoning Penalty

If Answer is Correct:
  Base Score = 1.0
  - Has <perception> tag: -0.2
  - Has <reasoning> tag: -0.3

If Answer is Incorrect:
  Reward = 0.0 (regardless of tags)
```

### Examples

```python
# Type 1: Direct answer (optimal for simple questions)
Q: "What color is the car?"
A: "Red"
Reward: 1.0 ✓

# Type 2: With perception (less efficient but still good)
Q: "What color is the car?"
A: "<perception>The car is red.</perception>\n<answer>Red</answer>"
Reward: 0.8

# Type 3: Full reasoning (least efficient for simple questions)
Q: "What color is the car?"
A: "<perception>...</perception>\n<reasoning>...</reasoning>\n<answer>Red</answer>"
Reward: 0.5

# Incorrect answer (always gets 0, regardless of reasoning)
Q: "What color is the car?"
A: "<perception>...</perception>\n<reasoning>...</reasoning>\n<answer>Blue</answer>"
Reward: 0.0 ✗
```

### Key Insight

The reward structure creates a **learning pressure** for the model to:
1. **Always get the answer right** (incorrect = 0.0 reward)
2. **Use minimal reasoning** for simple questions (higher reward)
3. **Use detailed reasoning** only when necessary (complex questions where simple approach fails)

## Data Preparation

### Step 1: Prepare GRPO Dataset

Convert your classified dataset into verl's parquet format:

```bash
python prepare_grpo_data.py \
    --type1_ids dataset/output/type1_ids.json \
    --type2_ids dataset/output/type2_ids.json \
    --type3_ids dataset/output/type3_ids.json \
    --output_dir grpo_data \
    --train_split 0.95
```

This creates:
```
grpo_data/
├── train.parquet        # Training data
├── val.parquet          # Validation data
├── images/              # Extracted images
└── metadata.json        # Dataset statistics
```

### Data Format

Parquet files contain:

| Column | Type | Description |
|--------|------|-------------|
| `data_source` | str | Sample ID (e.g., "okvqa_12345") |
| `prompt` | str | Question text |
| `images` | list | Absolute paths to image files |
| `gt_answer` | str/list | Ground truth answer(s) |
| `sample_type` | str | Original type (type1/type2/type3) |
| `dataset` | str | Source dataset name |

## Training

### Prerequisites

1. **Completed Stage 1 SFT training**
   ```bash
   ls LLaMA-Factory/saves/qwen2_5vl-3b/full/sft_all/
   ```

2. **Prepared GRPO data**
   ```bash
   ls grpo_data/train.parquet grpo_data/val.parquet
   ```

3. **Install verl**
   ```bash
   cd verl
   pip install -e .
   cd ..
   ```

### Quick Start

```bash
./train-grpo.sh --sft_model LLaMA-Factory/saves/qwen2_5vl-3b/full/sft_all --gpus 4
```

### Full Command

```bash
./train-grpo.sh \
    --sft_model LLaMA-Factory/saves/qwen2_5vl-3b/full/sft_all \
    --data_dir grpo_data \
    --output_dir saves/qwen2_5vl-3b/grpo/adaptive_reasoning \
    --gpus 4 \
    --engine vllm
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sft_model` | Auto-detect | Path to Stage 1 SFT checkpoint |
| `--data_dir` | `grpo_data` | GRPO training data directory |
| `--output_dir` | `saves/qwen2_5vl-3b/grpo/adaptive_reasoning` | Output directory |
| `--gpus` | 4 | Number of GPUs |
| `--engine` | `vllm` | Inference engine (vllm/sglang) |

## GRPO Configuration

### Key Settings

```yaml
# Algorithm
algorithm.adv_estimator: grpo
algorithm.use_kl_in_reward: False

# Data
data.train_batch_size: 256
data.max_response_length: 2048

# Rollout (Group Sampling)
actor_rollout_ref.rollout.n: 4           # Sample 4 responses per prompt
actor_rollout_ref.rollout.temperature: 0.8
actor_rollout_ref.rollout.top_p: 0.95

# Training
actor_rollout_ref.actor.optim.lr: 5e-7
actor_rollout_ref.actor.ppo_mini_batch_size: 64
actor_rollout_ref.actor.ppo_epochs: 1
actor_rollout_ref.actor.clip_ratio: 0.2

# KL Divergence
actor_rollout_ref.actor.use_kl_loss: True
actor_rollout_ref.actor.kl_loss_coef: 0.02
actor_rollout_ref.actor.kl_loss_type: low_var_kl

# Reward Model
reward_model.path: reward_functions.adaptive_reasoning_reward
```

### GRPO Explained

**Group Sampling**: For each question, generate 4 different responses

```
Question: "What color is the car?"

Response 1: "Red"                              # Type 1
Response 2: "<perception>...</perception>\n<answer>Red</answer>"  # Type 2
Response 3: "<perception>...</perception>\n<reasoning>...</reasoning>\n<answer>Red</answer>"  # Type 3
Response 4: "Blue"                             # Wrong

Rewards: [1.0, 0.8, 0.5, 0.0]
Group Average: 0.575

Advantages: [+0.425, +0.225, -0.075, -0.575]
           (Response 1 is best!)
```

The model learns: Response 1 (direct answer) is best for this simple question.

## Monitoring

### TensorBoard

```bash
tensorboard --logdir=saves/qwen2_5vl-3b/grpo/adaptive_reasoning
```

Key metrics to watch:
- `reward/mean`: Average reward (should increase)
- `reward/type1_ratio`: Percentage using Type 1 (should increase for simple questions)
- `reward/type2_ratio`: Percentage using Type 2
- `reward/type3_ratio`: Percentage using Type 3 (should increase for complex questions)
- `train/kl`: KL divergence from reference model
- `train/loss`: Training loss

### Expected Behavior

During training, you should see:

1. **Early epochs**: Model uses all three types randomly
2. **Mid training**: Model learns to avoid incorrect answers
3. **Late training**: Model learns to choose simpler modes when possible

**Convergence indicators:**
- Reward plateau at ~0.8-0.9 (mixture of Type 1 and Type 2)
- Type 3 usage only for genuinely complex questions
- Low KL divergence (< 0.1)

## Hardware Requirements

### Memory (Estimated)

**Single GPU Setup:**
- Not recommended (GRPO needs parallel rollouts)

**Multi-GPU Setup (Recommended):**
- 4x A100 40GB: Suitable for 3B model
- 4x A100 80GB: Optimal

### Training Time (Estimated)

- ~8-12 hours for 20 epochs (4x A100)
- Depends on dataset size and rollout settings

## Inference & Evaluation

### Test the Model

```bash
cd verl
python examples/inference/inference_grpo.py \
    --model_path ../saves/qwen2_5vl-3b/grpo/adaptive_reasoning \
    --image_path test_image.jpg \
    --question "What is in this image?"
```

### Expected Behavior

**Simple questions:**
```
Q: "What color is the car?"
A: "Red"                           # Type 1 (direct)
```

**Medium complexity:**
```
Q: "What is the person doing?"
A: "<perception>A person is riding a bicycle on a road.</perception>

<answer>Riding a bicycle</answer>"  # Type 2 (perception)
```

**Complex questions:**
```
Q: "What is the area of the triangle in the image?"
A: "<perception>The image shows a triangle with base 8 and height 6.</perception>

<reasoning>To find the area: A = (1/2) × base × height = (1/2) × 8 × 6 = 24 square units.</reasoning>

<answer>24</answer>"                # Type 3 (full reasoning)
```

## Customization

### Adjust Reward Penalties

Edit `reward_functions/adaptive_reasoning_reward.py`:

```python
reward_fn = AdaptiveReasoningReward(
    correct_reward=1.0,
    perception_penalty=0.2,      # Increase to discourage perception more
    reasoning_penalty=0.3,       # Increase to discourage reasoning more
    length_penalty_coef=0.01,    # Add penalty for long responses
)
```

### Adjust Sampling

In `train-grpo.sh`, modify:

```bash
actor_rollout_ref.rollout.n=8           # More samples per prompt
actor_rollout_ref.rollout.temperature=1.0  # More diverse sampling
```

## Troubleshooting

### Low Rewards

**Issue**: Average reward stays < 0.3

**Solutions:**
1. Check SFT model quality (should already answer correctly)
2. Reduce penalty coefficients
3. Increase learning rate
4. Check reward function implementation

### Model Always Uses Type 3

**Issue**: Model always includes perception + reasoning

**Solutions:**
1. Increase reasoning_penalty (try 0.4 or 0.5)
2. Ensure SFT model was trained on Type 1 examples
3. Add more Type 1 examples to training data

### Model Never Uses Type 3

**Issue**: Model avoids reasoning even on complex questions

**Solutions:**
1. Reduce reasoning_penalty (try 0.1 or 0.2)
2. Add more complex questions to validation set
3. Use curriculum learning (start with higher penalty, decrease over time)

### Out of Memory

**Solutions:**
1. Reduce `data.train_batch_size`
2. Reduce `actor_rollout_ref.rollout.n` (fewer samples per prompt)
3. Enable `actor_rollout_ref.actor.fsdp_config.param_offload=True`
4. Use more GPUs

## Advanced Topics

### Curriculum Learning

Gradually increase difficulty:

```bash
# Epoch 1-5: High penalties (learn to use Type 1)
reasoning_penalty=0.5

# Epoch 6-10: Medium penalties
reasoning_penalty=0.3

# Epoch 11-20: Normal penalties
reasoning_penalty=0.2
```

### Mixed Reward Functions

Combine multiple objectives:

```python
# 80% adaptive reasoning + 20% length penalty
reward = 0.8 * adaptive_reward + 0.2 * length_reward
```

### Question-Type-Specific Rewards

Adjust rewards based on question difficulty:

```python
if is_math_question:
    reasoning_penalty = 0.1  # Encourage reasoning for math
elif is_simple_question:
    reasoning_penalty = 0.5  # Strongly discourage reasoning
```

## File Structure

```
vlm-adaptive-resoning/
├── verl/                                    # GRPO training framework
├── reward_functions/
│   └── adaptive_reasoning_reward.py         # Adaptive reward function
├── grpo_data/                               # GRPO training data
│   ├── train.parquet
│   ├── val.parquet
│   └── images/
├── prepare_grpo_data.py                     # Data preparation script
├── train-grpo.sh                            # Training launcher
├── train_configs/
│   └── run_qwen2_5vl_3b_grpo_adaptive.sh   # Detailed config
└── saves/qwen2_5vl-3b/grpo/                # Checkpoints
    └── adaptive_reasoning/
```

## Citation

If you use this training pipeline, please cite:

```bibtex
@article{verl2024,
  title={verl: Volcano Engine Reinforcement Learning for LLMs},
  author={ByteDance Seed Team},
  journal={arXiv preprint},
  year={2024}
}

@article{deepseekmath2024,
  title={DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author={DeepSeek Team},
  journal={arXiv preprint arXiv:2402.03300},
  year={2024}
}
```

## References

- [verl Documentation](https://verl.readthedocs.io/)
- [GRPO Paper (DeepSeekMath)](https://arxiv.org/abs/2402.03300)
- [HybridFlow Paper](https://arxiv.org/abs/2409.19256)
