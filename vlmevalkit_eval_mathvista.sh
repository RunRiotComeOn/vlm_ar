#!/bin/bash

echo "==================================="
echo "Run MathVista Evaluation (Local Verifier)"
echo "==================================="

# 使用本地 Qwen3-8B 作为 verifier，不需要 OpenAI API
export VERIFIER_PATH="/nas03/yixuh/models/Qwen3-8B"

echo "Verifier model: $VERIFIER_PATH"

# 工作目录
WORK_DIR="/nas03/yixuh/vlm-adaptive-resoning/VLMEvalKit/outputs/qwen2_5vl_3b/grpo/trial-v2"

# 如果工作目录不存在则自动创建
if [ ! -d "$WORK_DIR" ]; then
    echo "创建工作目录: $WORK_DIR"
    mkdir -p "$WORK_DIR"
    if [ $? -eq 0 ]; then
        echo "目录创建成功"
    else
        echo "错误：无法创建目录 $WORK_DIR"
        exit 1
    fi
else
    echo "工作目录已存在: $WORK_DIR"
fi

cd /nas03/yixuh/vlm-adaptive-resoning/VLMEvalKit

# Run MathVista evaluation with local verifier
python ./run.py \
    --config configs/qwen2_5vl_3b/grpo/trial-v2/mathvista_config.json \
    --work-dir "$WORK_DIR" \
    --mode eval \
    --use-verifier

echo ""
echo "==================================="
echo "MathVista Evaluation Complete!"
echo "==================================="
echo "Results saved to: $WORK_DIR"
echo ""

WORK_DIR="/nas03/yixuh/vlm-adaptive-resoning/VLMEvalKit/outputs/qwen2_5vl_3b/sft/vision_sr1_1_epoch_without_st"

# 如果工作目录不存在则自动创建
if [ ! -d "$WORK_DIR" ]; then
    echo "创建工作目录: $WORK_DIR"
    mkdir -p "$WORK_DIR"
    if [ $? -eq 0 ]; then
        echo "目录创建成功"
    else
        echo "错误：无法创建目录 $WORK_DIR"
        exit 1
    fi
else
    echo "工作目录已存在: $WORK_DIR"
fi

cd /nas03/yixuh/vlm-adaptive-resoning/VLMEvalKit

# Run MathVista evaluation with local verifier
python ./run.py \
    --config configs/qwen2_5vl_3b/sft/vision_sr1_1_epoch_without_st/mathvista_config.json \
    --work-dir "$WORK_DIR" \
    --mode eval \
    --use-verifier

echo ""
echo "==================================="
echo "MathVista Evaluation Complete!"
echo "==================================="
echo "Results saved to: $WORK_DIR"
echo ""

WORK_DIR="/nas03/yixuh/vlm-adaptive-resoning/VLMEvalKit/outputs/qwen2_5vl_3b/base"

# 如果工作目录不存在则自动创建
if [ ! -d "$WORK_DIR" ]; then
    echo "创建工作目录: $WORK_DIR"
    mkdir -p "$WORK_DIR"
    if [ $? -eq 0 ]; then
        echo "目录创建成功"
    else
        echo "错误：无法创建目录 $WORK_DIR"
        exit 1
    fi
else
    echo "工作目录已存在: $WORK_DIR"
fi

cd /nas03/yixuh/vlm-adaptive-resoning/VLMEvalKit

# Run MathVista evaluation with local verifier
python ./run.py \
    --config configs/qwen2_5vl_3b/base/mathvista_config.json \
    --work-dir "$WORK_DIR" \
    --mode eval \
    --use-verifier

echo ""
echo "==================================="
echo "MathVista Evaluation Complete!"
echo "==================================="
echo "Results saved to: $WORK_DIR"
echo ""
