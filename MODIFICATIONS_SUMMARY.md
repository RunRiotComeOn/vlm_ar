# SFT 训练 Special Tokens 修改总结

## 问题描述

你原来的训练中，`<perception>`, `</perception>`, `<reasoning>`, `</reasoning>`, `<answer>`, `</answer>` 这些标签被当作普通文本处理，会被 tokenizer 切分成多个 tokens。

## 解决方案

将这些标签添加为 **special tokens**，这样它们会被作为单个 token 处理，并且在生成时可以选择性隐藏。

## 主要修改

### 1. 添加 Special Tokens 到 Tokenizer

**文件**: `add_special_tokens.py`

- 从 `Qwen/Qwen2.5-VL-7B-Instruct` 加载 tokenizer
- 添加 6 个新的 special tokens
- 保存到 `/nas03/yixuh/vlm-adaptive-resoning/models/qwen2.5-vl-7b-with-reasoning-tokens`

**结果**:
- Vocab size: 151665 → 151671 (+6)
- 新增 token IDs: 151665-151670

### 2. 修改训练配置

**原文件**: `train_configs/qwen2_5vl_7b_full_sft_all.yaml`
```yaml
model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct
output_dir: saves/qwen2_5vl-7b/full/sft_9k
```

**新文件**: `train_configs/qwen2_5vl_7b_full_sft_all_with_tokens.yaml`
```yaml
model_name_or_path: /nas03/yixuh/vlm-adaptive-resoning/models/qwen2.5-vl-7b-with-reasoning-tokens
output_dir: saves/qwen2_5vl-7b/full/sft_9k_with_tokens
```

**关键变化**:
- ✓ 使用扩展了 special tokens 的 tokenizer
- ✓ 使用绝对路径（避免 cd 导致的路径问题）
- ✓ 使用不同的输出目录

### 3. 修改训练脚本

**原文件**: `train-sft.sh`
**新文件**: `train-sft-with-tokens-simple.sh`

**关键变化**:
- ✓ 添加 tokenizer 验证步骤
- ✓ 检查所有必需文件
- ✓ 使用新的配置文件

### 4. 生成时选择性隐藏 Tokens

**文件**: `generate_with_selective_tokens.py`

**关键功能**:
```python
# 保留所有 special tokens
output = processor.batch_decode(ids, skip_special_tokens=False)[0]

# 只移除 reasoning tokens
for token in ['<perception>', '</perception>', '<reasoning>',
              '</reasoning>', '<answer>', '</answer>']:
    output = output.replace(token, '')
```

**为什么不用 `skip_special_tokens=True`**:
- 那样会隐藏**所有** special tokens，包括：
  - Qwen 系统 tokens: `<|im_start|>`, `<|im_end|>`
  - Vision tokens: `<|vision_start|>`, `<|image_pad|>` 等
- 我们只想隐藏 reasoning structure tokens

## 创建的文件

### 核心文件
1. **add_special_tokens.py** - 添加 special tokens 到 tokenizer
2. **train_configs/qwen2_5vl_7b_full_sft_all_with_tokens.yaml** - 新的训练配置
3. **train-sft-with-tokens-simple.sh** - 新的训练脚本
4. **generate_with_selective_tokens.py** - 选择性隐藏 tokens 的生成脚本

### 验证和测试文件
5. **test_special_tokens.py** - 验证 special tokens 设置
6. **verify_model_loading.py** - 验证模型加载（可选）

### 文档文件
7. **SPECIAL_TOKENS_GUIDE.md** - 完整使用指南
8. **QUICK_START_WITH_SPECIAL_TOKENS.md** - 快速开始指南
9. **MODIFICATIONS_SUMMARY.md** - 本文件（修改总结）

## 使用流程

### 训练

```bash
# 一键启动
bash train-sft-with-tokens-simple.sh
```

脚本会：
1. 验证 tokenizer（包含 6 个新 tokens）
2. 检查必需文件
3. 启动训练

### 生成（隐藏 reasoning tokens）

```bash
# 默认：隐藏 reasoning tokens
python generate_with_selective_tokens.py \
    --model_path /nas03/yixuh/vlm-adaptive-resoning/saves/qwen2_5vl-7b/full/sft_9k_with_tokens \
    --image test.jpg \
    --question "What is this?"

# 输出：干净的文本，没有 <perception> 等标签
```

### 生成（保留 reasoning structure）

```bash
# 调试模式：保留 reasoning tokens
python generate_with_selective_tokens.py \
    --model_path /nas03/yixuh/vlm-adaptive-resoning/saves/qwen2_5vl-7b/full/sft_9k_with_tokens \
    --image test.jpg \
    --question "What is this?" \
    --show_reasoning

# 输出：带有 <perception>, <reasoning>, <answer> 标签
```

### 生成（提取组件）

```bash
# 分析模式：提取各个组件
python generate_with_selective_tokens.py \
    --model_path /nas03/yixuh/vlm-adaptive-resoning/saves/qwen2_5vl-7b/full/sft_9k_with_tokens \
    --image test.jpg \
    --question "What is this?" \
    --show_components

# 输出：
# [PERCEPTION]
# ...
# [REASONING]
# ...
# [ANSWER]
# ...
```

## 关键优势

### 1. Token 效率提升

**不使用 special tokens**:
```
"<perception>Test</perception>"
→ ['<', 'perception', '>', 'Test', '</', 'perception', '>']
→ 7 tokens
```

**使用 special tokens**:
```
"<perception>Test</perception>"
→ ['<perception>', 'Test', '</perception>']
→ 3 tokens (-57% tokens!)
```

### 2. 灵活的输出控制

- ✓ 默认：隐藏 reasoning tokens（给用户看干净输出）
- ✓ 调试：显示 reasoning structure（分析模型推理过程）
- ✓ 分析：提取各组件（评估各部分质量）

### 3. 模型训练更有效

- ✓ 模型更容易学习结构化输出
- ✓ 减少 token 数量，提高训练效率
- ✓ Special tokens 有独立的 embeddings

## 数据文件

✓ **无需修改** - 你的训练数据（`train_all.json`）已经包含这些标签，tokenizer 会自动识别

## 注意事项

### 1. Vocabulary Size 变化

- **原始模型**: vocab_size = 151665
- **新模型**: vocab_size = 151671 (+6)

⚠️ **重要**: 训练后的模型只能使用相同的 tokenizer（vocab_size 必须匹配）

### 2. Embedding Layer 自动调整

- LLaMA-Factory 会自动检测 tokenizer vocab size
- 自动扩展 model embedding layer
- 新增的 6 个 tokens 会被随机初始化

### 3. 路径必须使用绝对路径

- ✗ 错误: `./models/qwen2.5-vl-7b-with-reasoning-tokens`
- ✓ 正确: `/nas03/yixuh/vlm-adaptive-resoning/models/qwen2.5-vl-7b-with-reasoning-tokens`

原因：训练脚本会 `cd` 到 LLaMA-Factory 目录，相对路径会失效

## 验证清单

### 训练前验证

```bash
# 1. 验证 special tokens
python test_special_tokens.py

# 预期输出：
# ✓ <perception>         -> ['<perception>']
# ✓ </perception>        -> ['</perception>']
# ...
# ✓ All special tokens are working correctly!

# 2. 验证文件完整性
ls -l /nas03/yixuh/vlm-adaptive-resoning/models/qwen2.5-vl-7b-with-reasoning-tokens/

# 必需文件：
# - tokenizer.json
# - tokenizer_config.json
# - config.json
# - model.safetensors.index.json
# - model-*-of-*.safetensors (5 个文件)
```

### 训练后验证

```bash
# 1. 检查模型输出目录
ls -l saves/qwen2_5vl-7b/full/sft_9k_with_tokens/

# 2. 测试生成
python generate_with_selective_tokens.py \
    --model_path saves/qwen2_5vl-7b/full/sft_9k_with_tokens \
    --image test_image.jpg \
    --question "Test question"
```

## 与原方案对比

| 特性 | 原方案 | 新方案 (Special Tokens) |
|------|--------|------------------------|
| Token 数量 | 多（标签被切分） | 少（标签是单个 token） |
| 训练效率 | 较低 | 较高 |
| 输出控制 | 困难（需复杂的后处理） | 简单（一行代码隐藏） |
| 结构学习 | 较难 | 容易（独立 embeddings） |
| 生成质量 | 可能不稳定 | 更稳定（明确的结构） |

## 快速参考

### 文件路径
```bash
# Tokenizer with special tokens
/nas03/yixuh/vlm-adaptive-resoning/models/qwen2.5-vl-7b-with-reasoning-tokens

# Training config
/nas03/yixuh/vlm-adaptive-resoning/train_configs/qwen2_5vl_7b_full_sft_all_with_tokens.yaml

# Training script
/nas03/yixuh/vlm-adaptive-resoning/train-sft-with-tokens-simple.sh

# Generation script
/nas03/yixuh/vlm-adaptive-resoning/generate_with_selective_tokens.py
```

### 常用命令
```bash
# 训练
bash train-sft-with-tokens-simple.sh

# 生成（隐藏 tokens）
python generate_with_selective_tokens.py --image IMG --question Q

# 生成（显示 tokens）
python generate_with_selective_tokens.py --image IMG --question Q --show_reasoning

# 验证 tokens
python test_special_tokens.py
```

## 问题排查

### 问题 1: "Repo id must be in the form..."
**原因**: 使用了相对路径
**解决**: 在配置文件中使用绝对路径

### 问题 2: "No file named pytorch_model.bin..."
**原因**: 缺少模型文件或索引文件
**解决**:
```bash
cp /path/to/cache/model.safetensors.index.json /path/to/model/
```

### 问题 3: Vocab size 不匹配
**原因**: Tokenizer 和模型不匹配
**解决**: 确保训练和推理使用相同的 tokenizer

## 总结

✅ **完成的工作**:
1. 添加 6 个 reasoning structure special tokens
2. 创建新的训练配置和脚本
3. 创建选择性隐藏 tokens 的生成脚本
4. 提供完整的文档和验证工具

🚀 **下一步**:
1. 运行 `bash train-sft-with-tokens-simple.sh` 开始训练
2. 训练完成后，使用 `generate_with_selective_tokens.py` 生成输出
3. 生成的输出会自动隐藏 reasoning tokens，给用户呈现干净的结果！
