#!/bin/bash
set -e

echo "========================================================================"
echo "  Curriculum SFT Training: 2-Phase Strategy"
echo "========================================================================"
echo ""
echo "Training Strategy:"
echo "  Phase 1: Train on Type 3 ONLY (cold_start_9k) - 9,044 samples"
echo "           Model learns ALL special tokens solidly"
echo "           Format: <perception>...</perception><reasoning>...</reasoning><answer>...</answer>"
echo "           Learning rate: 5.0e-5, Epochs: 1"
echo ""
echo "  Phase 2: Continue on MIXED data - ~6,531 samples"
echo "           cold_start Type 1 (2,470): <answer> only"
echo "           cold_start Type 2 (1,061): <perception> + <answer>"
echo "           vlm_ar Type 3 formatted (3,000): <perception> + <reasoning> + <answer>"
echo "           Model learns to selectively omit tokens"
echo "           Learning rate: 1.0e-5 (lower), Epochs: 1"
echo ""
echo "Special Tokens: <perception>, </perception>, <reasoning>, </reasoning>, <answer>, </answer>"
echo "========================================================================"
echo ""

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate environment
echo "Activating llamafactory environment..."
conda activate llamafactory

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CLI: $(which llamafactory-cli)"
echo ""

# ============================================================================
# Step 0: Add special tokens to tokenizer (only needed once)
# ============================================================================
echo "========================================================================"
echo "Stage 0: Preparing Model with Reasoning Tokens"
echo "========================================================================"
MODEL_WITH_TOKENS="/nas03/yixuh/vlm-adaptive-resoning/models/qwen2.5-vl-3b-with-reasoning-tokens"
if [ ! -d "$MODEL_WITH_TOKENS" ]; then
    echo "Adding special tokens: <perception>, <reasoning>, <answer>"
    python /nas03/yixuh/vlm-adaptive-resoning/add_special_tokens.py \
        --model_path Qwen/Qwen2.5-VL-3B-Instruct \
        --output_path $MODEL_WITH_TOKENS
    echo "Tokenizer with reasoning tokens created"
else
    echo "Tokenizer already exists at: $MODEL_WITH_TOKENS"
fi
echo ""

# Copy model files (excluding tokenizer) to use with new tokenizer
echo "Preparing model configuration..."
python3 -c "
from transformers import AutoConfig
import os
import shutil

model_dir = '$MODEL_WITH_TOKENS'
base_model = 'Qwen/Qwen2.5-VL-3B-Instruct'

# Only copy model files if they don't exist
if not os.path.exists(os.path.join(model_dir, 'config.json')):
    print('Copying model configuration and weights...')

    # Load and save config
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    config.save_pretrained(model_dir)

    # Create symbolic links for model weights to save space
    from huggingface_hub import snapshot_download
    base_path = snapshot_download(base_model)

    for file in os.listdir(base_path):
        if file.endswith('.safetensors') or file.endswith('.bin'):
            src = os.path.join(base_path, file)
            dst = os.path.join(model_dir, file)
            if not os.path.exists(dst):
                os.symlink(src, dst)
                print(f'  Linked: {file}')

    # Copy other necessary files
    for file in ['config.json', 'generation_config.json', 'preprocessor_config.json',
                 'merges.txt', 'vocab.json', 'processor_config.json', 'model.safetensors.index.json']:
        src = os.path.join(base_path, file)
        dst = os.path.join(model_dir, file)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f'  Copied: {file}')

    print('Model files prepared')
else:
    print('Model files already exist')
"
echo ""

# ============================================================================
# Step 0.5: Reformat vlm_adaptive_reasoning type3 data (if not done)
# ============================================================================
FORMATTED_DATA="/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/data/vlm_adaptive_reasoning/train_type3_formatted.json"
if [ ! -f "$FORMATTED_DATA" ]; then
    echo "Reformatting vlm_adaptive_reasoning type3 data..."
    python /nas03/yixuh/vlm-adaptive-resoning/reformat_vlm_ar_type3.py
    echo ""
else
    echo "Formatted data already exists: $FORMATTED_DATA"
    echo ""
fi

# Navigate to LLaMA-Factory directory
cd /nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory

# Set GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# ============================================================================
# Phase 1: Train on Type 3 ONLY (all special tokens)
# ============================================================================
echo "========================================================================"
echo "Phase 1: Training on cold_start Type 3 ONLY"
echo "========================================================================"
echo "Dataset: vision_sr1_cold_type3 (9,044 samples)"
echo "Format: <perception>...</perception><reasoning>...</reasoning><answer>...</answer>"
echo "Goal: Model learns ALL special tokens solidly"
echo "Learning rate: 5.0e-5"
echo "Epochs: 1"
echo ""

FORCE_TORCHRUN=1 llamafactory-cli train ../train_configs/qwen2_5vl_3b_curriculum_phase1.yaml

echo ""
echo "Phase 1 completed!"
echo ""

# Navigate to LLaMA-Factory directory (in case it changed)
cd /nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory

# Set GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ============================================================================
# Phase 2: Continue training on MIXED data (lower learning rate)
# ============================================================================
echo "========================================================================"
echo "Phase 2: Training on MIXED data (learning to selectively omit tokens)"
echo "========================================================================"
echo "Datasets:"
echo "  - cold_start Type 1: 2,470 samples (<answer> only)"
echo "  - cold_start Type 2: 1,061 samples (<perception> + <answer>)"
echo "  - vlm_ar Type 3 formatted: 5,928 samples (<perception> + <reasoning> + <answer>)"
echo "  - Total: ~6,531 samples"
echo "Goal: Model learns WHEN to use which tokens"
echo "Learning rate: 1.0e-5 (reduced)"
echo "Epochs: 1"
echo ""

# Check Phase 1 model exists
PHASE1_DIR="/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/saves/qwen2_5vl-3b/curriculum/phase1"
if [ -d "$PHASE1_DIR" ]; then
    echo "Continuing from Phase 1 checkpoint: $PHASE1_DIR"
    FORCE_TORCHRUN=1 llamafactory-cli train ../train_configs/qwen2_5vl_3b_curriculum_phase2.yaml
else
    echo "ERROR: Phase 1 checkpoint not found at $PHASE1_DIR"
    exit 1
fi

echo ""
echo "========================================================================"
echo "Curriculum SFT Training Completed!"
echo "========================================================================"
echo ""
echo "Training Summary:"
echo "  Phase 1: Type 3 only (9,044 samples) - Learned all special tokens"
echo "  Phase 2: Mixed data (7,531 samples) - Learned selective token usage"
echo ""
echo "Final Model Location:"
echo "  /nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/saves/qwen2_5vl-3b/curriculum/phase2"
echo ""
echo "Special Tokens (added to tokenizer):"
echo "  <perception> / </perception> - Visual understanding"
echo "  <reasoning> / </reasoning>   - Step-by-step reasoning"
echo "  <answer> / </answer>         - Final answer"
echo ""
echo "To use the model for generation:"
echo "  skip_special_tokens=False to see tags"
echo "  skip_special_tokens=True for clean output"
echo "========================================================================"
