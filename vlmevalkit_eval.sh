#!/bin/bash

echo "==================================="
echo "Run Evaluation"
echo "==================================="

# # OpenAI API 配置 (使用 ChatAnywhere)
# export OPENAI_API_KEY="sk-your-chatanywhere-key"  # 替换为你的 API Key
# export OPENAI_API_BASE="https://api.chatanywhere.org/v1"

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

Run evaluation
python ./run.py \
    --config configs/qwen2_5vl_3b/grpo/trial-v2/mmmu_config.json \
    --work-dir "$WORK_DIR" \
    --mode all
    # --reuse

python ./run.py \
    --config configs/qwen2_5vl_3b/grpo/trial-v2/ocrbench_config.json \
    --work-dir "$WORK_DIR" \
    --mode all

# python ./run.py \
#     --config configs/qwen2_5vl_3b/sft/vision_sr1_2_epoch/mathvista_config.json \
#     --work-dir "$WORK_DIR" \
#     --mode all

echo ""
echo "==================================="
echo "Evaluation Complete!"
echo "==================================="
echo "Results saved to: $WORK_DIR"
echo ""