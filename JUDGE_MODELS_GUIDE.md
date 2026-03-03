# VLMEvalKit Judge 模型配置指南

## 修改内容总结

已修改以下文件以支持更多 judge 模型：

### 已修改的文件

1. **VLMEvalKit/vlmeval/dataset/image_mcq.py** (主要文件，包含 MMMU)
   - 修改了 6 处 assert 限制
   - 支持 MMMU、CCBench、MUIRBench、HRBench、AffordanceDataset、TopViewRS 等数据集

2. **VLMEvalKit/vlmeval/dataset/mvbench.py** - MVBench 数据集
3. **VLMEvalKit/vlmeval/dataset/tamperbench.py** - TamperBench 数据集
4. **VLMEvalKit/vlmeval/dataset/mlvu.py** - MLVU 数据集
5. **VLMEvalKit/vlmeval/dataset/longvideobench.py** - LongVideoBench 数据集
6. **VLMEvalKit/vlmeval/dataset/video_holmes.py** - Video Holmes 数据集
7. **VLMEvalKit/vlmeval/dataset/videomme.py** - VideoMME 数据集
8. **VLMEvalKit/vlmeval/dataset/worldsense.py** - WorldSense 数据集
9. **VLMEvalKit/vlmeval/dataset/text_mcq.py** - 文本 MCQ 数据集
10. **VLMEvalKit/vlmeval/dataset/EgoExoBench/egoexobench.py** - EgoExoBench 数据集

### 修改前后对比

**修改前：**
```python
assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
```

**修改后：**
```python
assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125', 'gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo']
name_str_map = {
    'chatgpt-0125': 'openai',
    'gpt-4-0125': 'gpt4',
    'gpt-4o-mini': 'gpt4omini',
    'gpt-4o': 'gpt4o',
    'gpt-4-turbo': 'gpt4turbo'
}
```

## 现在支持的 Judge 模型

| 模型名称 | 实际模型 | 成本 | 推荐场景 |
|---------|---------|------|---------|
| `chatgpt-0125` | gpt-3.5-turbo-0125 | $ | 默认选择，便宜 |
| `gpt-4-0125` | gpt-4-0125-preview | $$$ | 需要最高准确度 |
| `gpt-4-turbo` | gpt-4-1106-preview | $$$ | GPT-4 快速版本 |
| `gpt-4o` | gpt-4o-2024-05-13 | $$ | GPT-4 优化版本 |
| `gpt-4o-mini` | gpt-4o-mini-2024-07-18 | $ | **推荐！性价比最高** |
| `exact_matching` | 无 (规则匹配) | 免费 | 仅 MCQ，准确率可能降低 |

## 使用方法

### 1. MMMU 数据集

```bash
# 使用 gpt-4o-mini (推荐)
python run.py \
    --data MMMU_DEV_VAL \
    --model Qwen2_5-VL-3B-Instruct \
    --judge gpt-4o-mini

# 使用 chatgpt-0125 (更便宜)
python run.py \
    --data MMMU_DEV_VAL \
    --model Qwen2_5-VL-3B-Instruct \
    --judge chatgpt-0125

# 不使用 GPT (免费但可能不准确)
python run.py \
    --data MMMU_DEV_VAL \
    --model Qwen2_5-VL-3B-Instruct \
    --judge exact_matching
```

### 2. MathVista 数据集

**注意：MathVista 必须使用 GPT，不支持 exact_matching**

```bash
# 使用 gpt-4o-mini (推荐)
python run.py \
    --data MathVista_MINI \
    --model Qwen2_5-VL-3B-Instruct \
    --judge gpt-4o-mini

# 使用 gpt-4o (更准确)
python run.py \
    --data MathVista_MINI \
    --model Qwen2_5-VL-3B-Instruct \
    --judge gpt-4o
```

### 3. 配置 API

使用 ChatAnywhere 或其他代理：

```bash
export OPENAI_API_KEY="sk-your-api-key"
export OPENAI_API_BASE="https://api.chatanywhere.org/v1"

# 然后运行评估
python run.py --data MMMU_DEV_VAL --model your_model --judge gpt-4o-mini
```

## 不同数据集的默认 Judge 模型

根据 `run.py` 的配置，不同数据集的默认 judge 模型：

| 数据集类型 | 默认 Judge | 说明 |
|-----------|-----------|------|
| MMMU | chatgpt-0125 | 可改用 gpt-4o-mini |
| MathVista | gpt-4o-mini | 必须使用 GPT |
| WeMath | gpt-4o-mini | 数学推理 |
| MMVet | gpt-4-turbo | 需要强推理 |
| LLaVABench | gpt-4-turbo | 需要强推理 |
| MMLongBench | gpt-4o | 长文本理解 |

## 成本优化建议

1. **开发测试阶段**：使用 `exact_matching`（MMMU）或 `chatgpt-0125`（其他）
2. **正式评估阶段**：使用 `gpt-4o-mini`（性价比最高）
3. **论文发表阶段**：使用 `gpt-4o` 或 `gpt-4-turbo`（最准确）

## 备份文件

所有修改的原始文件都已备份为 `.bak` 后缀：
- `image_mcq.py.bak`
- `mvbench.py.bak`
- 等等...

如需恢复原始版本：
```bash
cd /nas03/yixuh/vlm-adaptive-resoning/VLMEvalKit/vlmeval/dataset
mv image_mcq.py.bak image_mcq.py
```

## 故障排查

### 问题1：运行时报错 "assert model in [...]"

**原因**：还有数据集文件未修改

**解决**：
1. 查看错误信息中的文件名
2. 手动修改该文件中的 assert 语句
3. 或者联系我帮你修改

### 问题2：API 调用失败

**检查清单**：
```bash
# 1. 检查环境变量
echo $OPENAI_API_KEY
echo $OPENAI_API_BASE

# 2. 测试 API
python test_openai_api.py

# 3. 检查网络
curl -I https://api.chatanywhere.org
```

### 问题3：MathVista 不支持 exact_matching

这是正常的！MathVista 必须使用 GPT API，因为它需要：
- 提取数值答案
- 理解列表格式
- 处理多种答案类型

最便宜的选择是 `chatgpt-0125` 或 `gpt-4o-mini`。

## 相关文件

- `fix_judge_models.sh` - 批量修复脚本
- `test_openai_api.py` - API 测试脚本
- `vlmevalkit_eval.sh` - 评估脚本（已配置 API）

## 总结

现在你可以在 MMMU、MathVista 等数据集上使用以下任意 judge 模型：
- ✅ chatgpt-0125
- ✅ gpt-4-0125
- ✅ gpt-4-turbo
- ✅ gpt-4o
- ✅ **gpt-4o-mini (推荐)**
- ✅ exact_matching (仅 MCQ 数据集)

修改已完成，可以直接使用！
