# VLM Adaptive Reasoning

A two-stage training pipeline for teaching vision-language models to **adaptively** choose the appropriate reasoning mode based on question complexity.

## Overview

This project implements a novel training approach:

1. **Stage 1 (SFT)**: Supervised fine-tuning on three reasoning modes
2. **Stage 2 (GRPO)**: Reinforcement learning to select modes adaptively

### Three Reasoning Modes

| Mode | Format | Use Case | Example |
|------|--------|----------|---------|
| **Type 1** | `<answer>` | Simple, direct questions | Q: "What color?" → A: "Red" |
| **Type 2** | `<perception> + <answer>` | Needs description | Q: "What's happening?" → A: "\<perception>Person riding bike\</perception>\<answer>Riding a bike\</answer>" |
| **Type 3** | `<perception> + <reasoning> + <answer>` | Complex problems | Q: "Triangle area?" → A: Full geometric reasoning |

### Training Pipeline

```
┌─────────────────────┐
│  Dataset            │
│  (All Type 3)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Stage 1: SFT       │  ← Learns all three modes
│  (LLaMA-Factory)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Stage 2: GRPO      │  ← Learns adaptive (Learn to use three modes)
│  (verl + Adaptive   │
│   Reward Function)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Adaptive VLM       │  ← Intelligently chooses mode
│                     │
└─────────────────────┘
```

## Project Structure

```
vlm-adaptive-resoning/
├── dataset/                              # Dataset classification
│
├── LLaMA-Factory/                       # Stage 1: SFT training
│   └── data/
│       └── vlm_adaptive_reasoning/
│
├── verl/                                # Stage 2: GRPO training
│
├── reward_functions/                    # Adaptive reward function
│   └── adaptive_reasoning_reward.py
│
├── train_configs/                       # Training configurations
│   ├── qwen2_5vl_3b_full_sft_*.yaml    # SFT configs
│   └── run_qwen2_5vl_3b_grpo_adaptive.sh  # GRPO config
│
├── grpo_data/                           # GRPO training data
│
├── prepare_sft_data.py                  # SFT data preparation
├── prepare_grpo_data.py                 # GRPO data preparation
├── train-sft.sh                         # Stage 1 launcher
├── train-grpo.sh                        # Stage 2 launcher
│
└── Documentation
    ├── README.md (this file)
    ├── SETUP_SUMMARY.md                 # Stage 1 setup
    ├── TRAINING_README.md               # Stage 1 guide
    ├── GRPO_SETUP_SUMMARY.md           # Stage 2 setup
    └── GRPO_README.md                   # Stage 2 guide
```

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPUs (4x A100 recommended)
- ~200GB disk space

### Stage 1: Supervised Fine-Tuning

Teach the model all three reasoning modes.

```bash
# 1. Setup and prepare data
./quickstart.sh

# 2. Train on all types
./train-sft.sh --type all --gpus 4

# Or train on specific types
./train-sft.sh --type type1 --gpus 2
./train-sft.sh --type type2 --gpus 4
./train-sft.sh --type type3 --gpus 4
```

**Time**: ~6-10 hours on 4x A100

**Output**: `LLaMA-Factory/saves/qwen2_5vl-3b/full/sft_all/`

See `TRAINING_README.md` for details.

### Stage 2: GRPO with Adaptive Reward

Teach the model to choose modes intelligently.

```bash
# 1. Prepare GRPO data
python prepare_grpo_data.py

# 2. Install verl
cd verl && pip install -e . && cd ..

# 3. Train with adaptive reward
./train-grpo.sh \
    --sft_model LLaMA-Factory/saves/qwen2_5vl-3b/full/sft_all \
    --gpus 4
```

**Time**: ~8-12 hours on 4x A100

**Output**: `saves/qwen2_5vl-3b/grpo/adaptive_reasoning/`

See `GRPO_README.md` for details.

## Key Innovation: Adaptive Reward Function

The reward function encourages efficiency:

```python
Reward Design:
  If correct answer:
    Base reward: +1.0
    Has <perception>: -0.2
    Has <reasoning>: -0.3
  If incorrect:
    Reward: 0.0
```

**Examples:**

| Response Type | Correctness | Reward | Learning Signal |
|---------------|-------------|--------|-----------------|
| Direct "Red" | ✓ Correct | 1.0 | ⭐⭐⭐ Best! |
| Perception + "Red" | ✓ Correct | 0.8 | ⭐⭐ Good |
| Full reasoning + "Red" | ✓ Correct | 0.5 | ⭐ OK (but inefficient) |
| Any type + "Blue" | ✗ Wrong | 0.0 | ✗ Avoid! |

This teaches: *"Get it right with minimal reasoning"*

## Expected Results

### Stage 1 Behavior

Model responds with the mode it was trained on:

```python
# Train on Type 3 → Always Type 3
Q: "What color is the car?"
A: "<perception>The car is red</perception>
    <reasoning>Based on observation...</reasoning>
    <answer>Red</answer>"
```

### Stage 2 Behavior (After GRPO)

Model **adapts** based on question complexity:

```python
# Simple question → Type 1
Q: "What color is the car?"
A: "Red"

# Medium question → Type 2
Q: "What is the person doing?"
A: "<perception>A person riding a bicycle</perception>
    <answer>Riding a bicycle</answer>"

# Complex question → Type 3
Q: "What is the area of the triangle?"
A: "<perception>Triangle with base 8, height 6</perception>
    <reasoning>Area = (1/2) × base × height = (1/2) × 8 × 6 = 24</reasoning>
    <answer>24</answer>"
```

## Model

**Base Model**: Qwen/Qwen2.5-VL-3B-Instruct

**Training Method**:
- Stage 1: Full fine-tuning (freeze vision tower)
- Stage 2: GRPO with group sampling

**Datasets**:
- OK-VQA
- Geometry3K
- ScienceQA
- GQA
- CLEVR
- MathVista
- VCR
- TextbookQA

## Monitoring

### Stage 1 (SFT)

```bash
tensorboard --logdir=LLaMA-Factory/saves/qwen2_5vl-3b/full/
```

### Stage 2 (GRPO)

```bash
tensorboard --logdir=saves/qwen2_5vl-3b/grpo/
```

**Key Metrics**:
- `reward/mean`: Should increase to 0.7-0.9
- `reward/type1_ratio`: % using direct answers
- `reward/type3_ratio`: % using full reasoning

## Hardware Requirements

### Minimum

- Stage 1: 4x A100 40GB
- Stage 2: 4x A100 40GB

### Recommended

- Stage 1: 4x A100 80GB
- Stage 2: 4x A100 80GB

### Storage

- ~100GB for datasets
- ~50GB per checkpoint
- ~200GB total recommended

## Citation

```bibtex
@software{vlm_adaptive_reasoning,
  title={VLM Adaptive Reasoning: Two-Stage Training for Adaptive Reasoning Mode Selection},
  author={Your Name},
  year={2025},
  url={https://github.com/RunRiotComeOn/vlm-adaptive-resoning}
}

@article{verl2024,
  title={verl: Volcano Engine Reinforcement Learning for LLMs},
  author={ByteDance Seed Team},
  year={2024}
}

@article{deepseekmath2024,
  title={DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author={DeepSeek Team},
  journal={arXiv preprint arXiv:2402.03300},
  year={2024}
}
```

## License

This project uses:
- LLaMA-Factory (Apache 2.0)
- verl (Apache 2.0)

## Acknowledgments

- **LLaMA-Factory**: For the excellent SFT training framework
- **verl**: For the GRPO training infrastructure
- **Qwen Team**: For the Qwen2.5-VL base model
- **Dataset Providers**: OK-VQA, Geometry3K, ScienceQA, GQA, CLEVR, MathVista, VCR, TextbookQA

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Support

For issues or questions:
1. Check documentation in respective README files
2. Review TensorBoard logs
3. Open an issue on GitHub

---

**Happy Training! 🚀**
