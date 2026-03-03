#!/bin/bash
set -e

echo "========================================================================"
echo "  Phase Training: Progressive Data Integration"
echo "========================================================================"
echo ""
echo "Training Strategy:"
echo "  Phase 1: 1/3 of Type 3 (perception + reasoning + answer) - 3,014 samples"
echo "  Phase 2: 1/3 of Type 3 + 1/2 of Type 2 (add perception + answer) - 3,544 samples"
echo "  Phase 3: 1/3 of Type 3 + 1/2 of Type 2 + all Type 1 (add simple answer) - 6,017 samples"
echo ""
echo "This approach gradually introduces data of increasing complexity diversity."
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
    echo "✓ Tokenizer with reasoning tokens created"
else
    echo "✓ Tokenizer already exists at: $MODEL_WITH_TOKENS"
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

    print('✓ Model files prepared')
else:
    print('✓ Model files already exist')
"
echo ""

# Navigate to LLaMA-Factory directory
cd /nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory

# Set GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# ============================================================================
# Phase 1: Train on 1/3 of Type 3
# ============================================================================
echo "========================================================================"
echo "Phase 1: Training on 1/3 of Type 3 (Perception + Reasoning + Answer)"
echo "========================================================================"
echo "Samples: 3,014"
echo "Format: <perception>...</perception><reasoning>...</reasoning><answer>...</answer>"
echo "Learning rate: 5.0e-5"
echo "Epochs: 1"
echo ""

FORCE_TORCHRUN=1 llamafactory-cli train ../train_configs/qwen2_5vl_3b_phase1.yaml

echo ""
echo "✓ Phase 1 completed!"
echo ""

# Navigate to LLaMA-Factory directory
cd /nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory

# Set GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# # ============================================================================
# # Phase 2: Continue training on 1/3 Type 3 + 1/2 Type 2
# # ============================================================================
# echo "========================================================================"
# echo "Phase 2: Adding 1/3 Type 3 + 1/2 Type 2 (Perception + Answer)"
# echo "========================================================================"
# echo "Cumulative samples: 3,544 (1/3 Type 3: 3,014 + 1/2 Type 2: 530)"
# echo "New format added: <perception>...</perception><answer>...</answer>"
# echo "Learning rate: 3.0e-5 (reduced for fine-tuning)"
# echo "Epochs: 1"
# echo ""

# # Check Phase 1 model exists
# PHASE1_DIR="/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/saves/qwen2_5vl-3b/phase/phase1"
# if [ -d "$PHASE1_DIR" ]; then
#     echo "Continuing from Phase 1 checkpoint: $PHASE1_DIR"
#     echo "(model_name_or_path is set in the yaml config)"

#     FORCE_TORCHRUN=1 llamafactory-cli train ../train_configs/qwen2_5vl_3b_phase2.yaml
# else
#     echo "ERROR: Phase 1 checkpoint not found at $PHASE1_DIR"
#     exit 1
# fi

# echo ""
# echo "✓ Phase 2 completed!"
# echo ""

# # Navigate to LLaMA-Factory directory
# cd /nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory

# # Set GPUs
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
# echo ""

# # ============================================================================
# # Phase 3: Continue training on 1/3 Type 3 + 1/2 Type 2 + all Type 1
# # ============================================================================
# echo "========================================================================"
# echo "Phase 3: Adding 1/3 Type 3 + 1/2 Type 2 + all Type 1 (All formats)"
# echo "========================================================================"
# echo "Total samples: 6,017 (1/3 Type 3: 3,014 + 1/2 Type 2: 530 + Type 1: 2,473)"
# echo "New format added: <answer>...</answer>"
# echo "Learning rate: 1.0e-5 (further reduced for final tuning)"
# echo "Epochs: 1"
# echo ""

# # Check Phase 2 model exists (model_name_or_path is already set in the yaml config)
# PHASE2_DIR="/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/saves/qwen2_5vl-3b/phase/phase2"
# if [ -d "$PHASE2_DIR" ]; then
#     echo "Using Phase 2 model as starting point: $PHASE2_DIR"
#     echo "(model_name_or_path is set in the yaml config)"

#     FORCE_TORCHRUN=1 llamafactory-cli train ../train_configs/qwen2_5vl_3b_phase3.yaml
# else
#     echo "ERROR: Phase 2 model not found at $PHASE2_DIR"
#     exit 1
# fi

# echo ""
# echo "========================================================================"
# echo "✓ Phase Training Completed Successfully!"
# echo "========================================================================"
# echo ""
# echo "Training Summary:"
# echo "  Phase 1: 1/3 Type 3 (3,014 samples) - Complex reasoning"
# echo "  Phase 2: 1/3 Type 3 + 1/2 Type 2 (3,544 samples) - Added simpler perception"
# echo "  Phase 3: 1/3 Type 3 + 1/2 Type 2 + Type 1 (6,017 samples) - Added simple answers"
# echo ""
# echo "Final Model Location:"
# echo "  /nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/saves/qwen2_5vl-3b/phase/phase3"
# echo ""
# echo "To use the model for generation:"
# echo "  1. With tags visible: skip_special_tokens=False"
# echo "  2. Clean output: skip_special_tokens=True"
# echo ""
# echo "Special Tokens:"
# echo "  <perception> - Visual understanding"
# echo "  <reasoning> - Step-by-step reasoning"
# echo "  <answer> - Final answer"
# echo "========================================================================"
