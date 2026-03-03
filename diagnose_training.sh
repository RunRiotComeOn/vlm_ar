#!/bin/bash
# Script to diagnose training issues

echo "================================"
echo "GRPO Training Diagnostics"
echo "================================"
echo

echo "1. Checking running processes..."
ps aux | grep -E "python.*verl|vllm" | grep -v grep || echo "No training processes running"
echo

echo "2. Checking GPU memory usage..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo

echo "3. Checking disk space..."
df -h . | tail -1
echo

echo "4. Checking recent training outputs..."
if [ -d "saves/qwen2_5vl-3b/grpo" ]; then
    find saves/qwen2_5vl-3b/grpo -type f -newer /tmp -name "*.pt" -o -name "*.log" | head -5
else
    echo "No training output directory found"
fi
echo

echo "5. Checking for large log files..."
find . -maxdepth 3 -name "*.log" -size +10M -exec ls -lh {} \; 2>/dev/null | head -5
echo

echo "6. Checking data files..."
if [ -f "grpo_data/train.parquet" ]; then
    ls -lh grpo_data/*.parquet
    echo "Data files exist ✓"
else
    echo "ERROR: Data files not found!"
fi
echo

echo "7. Checking SFT model..."
if [ -d "LLaMA-Factory/saves/qwen2_5vl-3b/full/sft_9k" ]; then
    ls -lh LLaMA-Factory/saves/qwen2_5vl-3b/full/sft_9k/*.safetensors 2>/dev/null | wc -l
    echo "SFT model files found ✓"
else
    echo "ERROR: SFT model not found!"
fi
echo

echo "================================"
echo "Diagnostics complete"
echo "================================"
