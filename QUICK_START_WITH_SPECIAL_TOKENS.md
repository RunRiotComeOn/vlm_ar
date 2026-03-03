# 快速开始：使用 Special Tokens 训练

## 问题回顾

1. ✓ **已解决**：需要将 `<perception>`, `</perception>`, `<reasoning>`, `</reasoning>`, `<answer>`, `</answer>` 添加为 special tokens
2. ✓ **已解决**：生成时只隐藏这些 reasoning tokens，保留其他 special tokens

## 一键启动训练

```bash
chmod +x /nas03/yixuh/vlm-adaptive-resoning/train-sft-with-tokens-simple.sh
bash /nas03/yixuh/vlm-adaptive-resoning/train-sft-with-tokens-simple.sh
```

这个脚本会：
1. 验证 tokenizer 已正确设置（包含 6 个新的 reasoning tokens）
2. 检查所有必需文件
3. 启动训练

## 关键修改点

### 1. 训练配置文件

**文件**: `/nas03/yixuh/vlm-adaptive-resoning/train_configs/qwen2_5vl_7b_full_sft_all_with_tokens.yaml`

```yaml
### Model Configuration
model_name_or_path: /nas03/yixuh/vlm-adaptive-resoning/models/qwen2.5-vl-7b-with-reasoning-tokens
# ↑ 使用添加了 special tokens 的模型（绝对路径）

### Output Configuration
output_dir: saves/qwen2_5vl-7b/full/sft_9k_with_tokens
# ↑ 使用不同的输出目录
```

### 2. Tokenizer 设置

已添加的 Special Tokens（vocab size: 151665 → 151671）：
- `<perception>` (ID: 151665)
- `</perception>` (ID: 151666)
- `<reasoning>` (ID: 151667)
- `</reasoning>` (ID: 151668)
- `<answer>` (ID: 151669)
- `</answer>` (ID: 151670)

### 3. 数据文件

✓ **无需修改** - 你的训练数据已经包含这些标签

## 生成时隐藏 Reasoning Tokens

训练完成后，使用这个脚本生成：

```bash
python /nas03/yixuh/vlm-adaptive-resoning/generate_with_selective_tokens.py \
    --model_path /nas03/yixuh/vlm-adaptive-resoning/saves/qwen2_5vl-7b/full/sft_9k_with_tokens \
    --image path/to/image.jpg \
    --question "Your question here"
```

### 示例输出

**默认模式（隐藏 reasoning tokens）**:
```
The image shows a surfboard.
The surfboard was invented in 1926.
1926
```

**显示 reasoning structure**:
```bash
# 添加 --show_reasoning 参数
python generate_with_selective_tokens.py --show_reasoning ...
```

输出：
```
<perception>
The image shows a surfboard.
</perception>

<reasoning>
The surfboard was invented in 1926.
</reasoning>

<answer>
1926
</answer>
```

**提取各个组件**:
```bash
# 添加 --show_components 参数
python generate_with_selective_tokens.py --show_components ...
```

输出：
```
[PERCEPTION]
The image shows a surfboard.

[REASONING]
The surfboard was invented in 1926.

[ANSWER]
1926
```

## 关键优势

### 1. Token 效率
```
# 不使用 special tokens:
"<perception>Test</perception>" → 13 tokens

# 使用 special tokens:
"<perception>Test</perception>" → 9 tokens (-31% tokens!)
```

### 2. 灵活控制输出

```python
# 方法 1: 隐藏所有 reasoning tokens（默认）
output = generate_with_selective_tokens(..., hide_reasoning_tokens=True)
# 输出: 干净的文本

# 方法 2: 保留结构用于调试
output = generate_with_selective_tokens(..., hide_reasoning_tokens=False)
# 输出: 带有 <perception> 等标签

# 方法 3: 提取各组件用于分析
components = extract_reasoning_components(output)
# 返回: {'perception': '...', 'reasoning': '...', 'answer': '...'}
```

## 验证 Special Tokens

在训练前，验证 tokenizer 设置：

```bash
python /nas03/yixuh/vlm-adaptive-resoning/test_special_tokens.py
```

预期输出：
```
Special Token Verification
============================================================
✓ <perception>         -> ['<perception>']
✓ </perception>        -> ['</perception>']
✓ <reasoning>          -> ['<reasoning>']
✓ </reasoning>         -> ['</reasoning>']
✓ <answer>             -> ['<answer>']
✓ </answer>            -> ['</answer>']

✓ All special tokens are working correctly!
```

## 训练后的模型目录结构

```
saves/qwen2_5vl-7b/full/sft_9k_with_tokens/
├── config.json                 # 模型配置
├── tokenizer.json              # Tokenizer（包含新的 special tokens）
├── tokenizer_config.json       # Tokenizer 配置
├── special_tokens_map.json     # Special tokens 映射
├── model-*.safetensors         # 模型权重
└── training_args.bin           # 训练参数
```

## 常见问题

### Q1: 训练时会自动调整 embedding layer 吗？

**A**: 是的！LLaMA-Factory 会自动检测 tokenizer vocab size 并调整模型的 embedding layer。新增的 6 个 tokens 会被随机初始化。

### Q2: 为什么不使用 `skip_special_tokens=True`？

**A**: 因为那样会隐藏**所有** special tokens，包括：
- 我们的 reasoning tokens: `<perception>`, `<reasoning>`, `<answer>`
- Qwen 的系统 tokens: `<|im_start|>`, `<|im_end|>`
- Vision tokens: `<|vision_start|>`, `<|image_pad|>` 等

我们只想隐藏 reasoning tokens，所以使用 `skip_special_tokens=False` + 手动移除。

### Q3: 可以在已有模型上继续训练吗？

**A**: 可以，但需要注意：
- 如果从 checkpoint 继续训练，确保使用相同的 tokenizer（vocab size 必须匹配）
- 设置 `resume_from_checkpoint: path/to/checkpoint` 在 YAML 配置中

### Q4: 如何在其他代码中使用这个模型？

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch

# 加载模型（自动包含扩展的 vocabulary）
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/nas03/yixuh/vlm-adaptive-resoning/saves/qwen2_5vl-7b/full/sft_9k_with_tokens",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(
    "/nas03/yixuh/vlm-adaptive-resoning/saves/qwen2_5vl-7b/full/sft_9k_with_tokens"
)

# 生成
output_ids = model.generate(...)

# 方法 1: 保留所有 tokens，然后手动移除 reasoning tokens
output = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
for token in ['<perception>', '</perception>', '<reasoning>', '</reasoning>', '<answer>', '</answer>']:
    output = output.replace(token, '')

# 方法 2: 使用我们提供的函数
from generate_with_selective_tokens import remove_reasoning_tokens
output = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
clean_output = remove_reasoning_tokens(output)
```

## 总结

✓ **已完成**：
1. 添加 6 个 reasoning structure special tokens
2. 创建训练配置使用扩展的 tokenizer
3. 创建生成脚本支持选择性隐藏 tokens

✓ **下一步**：
1. 运行 `bash train-sft-with-tokens-simple.sh` 开始训练
2. 训练完成后，使用 `generate_with_selective_tokens.py` 生成
3. 在生成时，reasoning tokens 会被自动隐藏！
