# Vision-SR1-Cold-9K 快速开始指南

## 当前状态

✅ **已完成:**
- JSON 数据文件处理和格式转换（9364 个样本）
- LLaMA-Factory dataset_info.json 配置
- 数据分类（全部为 type3: perception + reasoning + answer）

❌ **待完成:**
- 下载图像文件到 `LLaMA-Factory/data/cold_start_9k/images/`

## 快速下载图像

### 推荐方法：使用 Git 克隆

```bash
cd /nas03/yixuh/vlm-adaptive-resoning
bash download_images_git.sh
```

### 验证安装

下载完成后验证：

```bash
python verify_cold_start_data.py
```

期望输出：
```
✓ JSON files: OK
✓ Images: OK
✓ Configuration: OK
```

## 使用数据集进行训练

### 在 LLaMA-Factory 中使用

```bash
cd /nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory

# 使用全部数据（9364 个样本）
llamafactory-cli train \\
  --stage sft \\
  --model_name_or_path <your_vision_model> \\
  --dataset vision_sr1_cold_all \\
  --template qwen2_vl \\
  --finetuning_type lora \\
  --output_dir ./saves/cold_start_9k \\
  --per_device_train_batch_size 4 \\
  --gradient_accumulation_steps 4 \\
  --lr_scheduler_type cosine \\
  --logging_steps 10 \\
  --save_steps 500 \\
  --learning_rate 5e-5 \\
  --num_train_epochs 3 \\
  --plot_loss \\
  --fp16

# 仅使用 type3 数据（与 all 相同，因为所有数据都是 type3）
llamafactory-cli train \\
  --dataset vision_sr1_cold_type3 \\
  ... # 其他参数同上
```

### 可用的数据集名称

- `vision_sr1_cold_all`: 全部 9364 个样本
- `vision_sr1_cold_type3`: type3 样本（9364 个，与 all 相同）
- `vision_sr1_cold_type2`: type2 样本（0 个）
- `vision_sr1_cold_type1`: type1 样本（0 个）

## 数据格式示例

每个样本包含：

```json
{
  "messages": [
    {
      "content": "Question: ... <image>",
      "role": "user"
    },
    {
      "content": "<perception>...</perception><reasoning>...</reasoning><answer>...</answer>",
      "role": "assistant"
    }
  ],
  "images": ["cold_start_9k/images/XXX.jpg"]
}
```

## 文件位置

- **数据文件**: `/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/data/cold_start_9k/`
  - `train_all.json` (9364 样本)
  - `train_type3.json` (9364 样本)
  - `train_type2.json` (0 样本)
  - `train_type1.json` (0 样本)

- **图像目录**: `/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/data/cold_start_9k/images/`
  - 需要 9364 张图像

- **配置文件**: `/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/data/dataset_info.json`

## 脚本说明

| 脚本 | 用途 |
|------|------|
| `process_cold_start_data.py` | 处理原始数据，转换格式 |
| `download_images_git.sh` | 使用 git 下载图像（推荐） |
| `download_cold_start_images_retry.py` | 使用 Python API 下载图像（带重试） |
| `verify_cold_start_data.py` | 验证数据完整性 |

## 故障排查

### 图像下载失败

如果遇到 HuggingFace API 限流：

1. **使用 Git 方法**（推荐）:
   ```bash
   bash download_images_git.sh
   ```

2. **等待并重试 Python 方法**:
   ```bash
   # 等待 5-10 分钟
   python download_cold_start_images_retry.py
   ```

3. **手动克隆**:
   ```bash
   git clone https://huggingface.co/datasets/LMMs-Lab-Turtle/Vision-SR1-Cold-9K temp_download
   cp temp_download/cold_start/*.jpg LLaMA-Factory/data/cold_start_9k/images/
   # 或者
   cp temp_download/images/*.jpg LLaMA-Factory/data/cold_start_9k/images/
   ```

### 验证失败

运行验证脚本查看详细错误：
```bash
python verify_cold_start_data.py
```

## 下一步

1. 下载图像（如果还没下载）
2. 运行验证脚本确认一切正常
3. 开始训练你的视觉语言模型！

## 更多信息

详细文档请参阅: `COLD_START_9K_README.md`
