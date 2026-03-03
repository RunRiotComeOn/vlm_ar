#!/bin/bash

echo "==================================="
echo "Update Config File for VLMEvalKit"
echo "==================================="

# 目标配置目录
CONFIG_DIR="/nas03/yixuh/vlm-adaptive-resoning/VLMEvalKit/configs/qwen2_5vl_3b/grpo/trial-v2"

# 如果目录不存在则自动创建
if [ ! -d "$CONFIG_DIR" ]; then
    echo "创建配置目录: $CONFIG_DIR"
    mkdir -p "$CONFIG_DIR"
    if [ $? -eq 0 ]; then
        echo "目录创建成功"
    else
        echo "错误：无法创建目录 $CONFIG_DIR"
        exit 1
    fi
else
    echo "配置目录已存在: $CONFIG_DIR"
fi

# 创建/覆盖配置文件（`mmmu_config.json`）
cat > "$CONFIG_DIR/mmmu_config.json" << 'EOF'
{
  "model": {
    "vision_sr1_1_epoch_without_st": {
      "class": "Qwen2VLChat",
      "model_path": "/nas03/yixuh/vlm-adaptive-resoning/saves/qwen2_5vl-3b/grpo/adaptive_reward_v2_full/global_step_220/actor_merged",
      "min_pixels": 35840,
      "max_pixels": 458752,
      "use_custom_prompt": false,
      "use_vllm": true,
      "max_new_tokens": 4096,
      "temperature": 0.01,
      "gpu_utils": 0.8,
      "attn_implementation": "eager"
    }
  },
  "data": {
    "MMMU_DEV_VAL": {
      "class": "ImageMCQDataset",
      "dataset": "MMMU_DEV_VAL"
    }
  }
}
EOF

# 如果创建成功，给出提示
if [ $? -eq 0 ]; then
    echo ""
    echo "配置文件已成功创建/更新："
    echo "  → $CONFIG_DIR/mmmu_config.json"
    echo ""
else
    echo "错误：配置文件写入失败！"
    exit 1
fi

# 创建/覆盖配置文件（`ocrbench_config.json`）
cat > "$CONFIG_DIR/ocrbench_config.json" << 'EOF'
{
  "model": {
    "vision_sr1_1_epoch_without_st": {
      "class": "Qwen2VLChat",
      "model_path": "/nas03/yixuh/vlm-adaptive-resoning/saves/qwen2_5vl-3b/grpo/adaptive_reward_v2_full/global_step_220/actor_merged",
      "min_pixels": 35840,
      "max_pixels": 458752,
      "use_custom_prompt": false,
      "use_vllm": true,
      "max_new_tokens": 4096,
      "temperature": 0.01,
      "gpu_utils": 0.8,
      "attn_implementation": "eager"
    }
  },
  "data": {
    "OCRBench": {
      "class": "OCRBench",
      "dataset": "OCRBench"
    }
  }
}
EOF

# 如果创建成功，给出提示
if [ $? -eq 0 ]; then
    echo ""
    echo "配置文件已成功创建/更新："
    echo "  → $CONFIG_DIR/ocrbench_config.json"
    echo ""
else
    echo "错误：配置文件写入失败！"
    exit 1
fi

# 创建/覆盖配置文件（`mathvista_config.json`）
cat > "$CONFIG_DIR/mathvista_config.json" << 'EOF'
{
  "model": {
    "vision_sr1_1_epoch_without_st": {
      "class": "Qwen2VLChat",
      "model_path": "/nas03/yixuh/vlm-adaptive-resoning/saves/qwen2_5vl-3b/grpo/adaptive_reward_v2_full/global_step_220/actor_merged",
      "min_pixels": 35840,
      "max_pixels": 458752,
      "use_custom_prompt": false,
      "use_vllm": true,
      "max_new_tokens": 4096,
      "temperature": 0.01,
      "gpu_utils": 0.8,
      "attn_implementation": "eager"
    }
  },
  "data": {
    "MathVista_MINI": {
      "class": "MathVista",
      "dataset": "MathVista_MINI"
    }
  }
}
EOF

# 如果创建成功，给出提示
if [ $? -eq 0 ]; then
    echo ""
    echo "配置文件已成功创建/更新："
    echo "  → $CONFIG_DIR/mathvista_config.json"
    echo ""
else
    echo "错误：配置文件写入失败！"
    exit 1
fi

# 可选：列出其他常用数据集配置（注释形式，方便以后快速切换）
cat << 'COMMENT'

# 其他常用数据集配置示例（可根据需要替换上面的 "data" 部分）：

# MMMU
# "data": {
#   "MMMU_DEV_VAL": {
#     "class": "ImageMCQDataset",
#     "dataset": "MMMU_DEV_VAL"
#   }
# }

# OCRBench
# "data": {
#   "OCRBench": {
#     "class": "OCRBench",
#     "dataset": "OCRBench"
#   }
# }

# MathVista (mini)
# "data": {
#   "MathVista_MINI": {
#     "class": "MathVista",
#     "dataset": "MathVista_MINI"
#   }
# }

# OCRVQA
# "data": {
#   "OCRVQA_TEST": {
#     "class": "ImageVQADataset",
#     "dataset": "OCRVQA_TEST"
#   }
# }

COMMENT

echo "Done!"