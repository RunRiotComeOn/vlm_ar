#!/bin/bash
set -e

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate environment
echo "Activating llamafactory environment..."
conda activate llamafactory

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CLI: $(which llamafactory-cli)"

# Step 1: Add special tokens to tokenizer (only needed once)
echo "========================================="
echo "Step 1: Adding special tokens to tokenizer..."
echo "========================================="
MODEL_WITH_TOKENS="/nas03/yixuh/vlm-adaptive-resoning/models/qwen2.5-vl-3b-with-reasoning-tokens"
if [ ! -d "$MODEL_WITH_TOKENS" ]; then
    python /nas03/yixuh/vlm-adaptive-resoning/add_special_tokens.py \
        --model_path Qwen/Qwen2.5-VL-3B-Instruct \
        --output_path $MODEL_WITH_TOKENS
    echo "✓ Tokenizer with reasoning tokens created"
else
    echo "✓ Tokenizer already exists, skipping creation"
fi

# Step 2: Copy model files (excluding tokenizer) to use with new tokenizer
echo ""
echo "========================================="
echo "Step 2: Preparing model with new tokenizer..."
echo "========================================="
python3 -c "
from transformers import Qwen2VLForConditionalGeneration, AutoConfig
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

# Step 3: Navigate to LLaMA-Factory directory
cd /nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory

# Set GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo ""
echo "========================================="
echo "Step 3: Starting training..."
echo "========================================="
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Run training with new config
echo "Training with reasoning structure tokens..."
FORCE_TORCHRUN=1 llamafactory-cli train ../train_configs/qwen2_5vl_3b_with_tokens_type3.yaml

echo ""
echo "========================================="
echo "Training completed!"
echo "========================================="
echo ""
echo "Note: When generating outputs, use skip_special_tokens=True to hide the tags:"
echo "  output = model.generate(..., skip_special_tokens=True)"
echo "  # This will hide <perception>, <reasoning>, <answer> tags"
