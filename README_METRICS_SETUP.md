# 🎉 train-grpo-v2.sh 已配置完成！

## ✅ 已完成的工作

`train-grpo-v2.sh` 已经修改完毕，现在支持完整的 TensorBoard Metrics 追踪功能！

### 主要改动

1. **使用 Adaptive Reasoning Reward Manager**
   - 替代了原来的自定义 reward function
   - 支持批量 metrics 计算和记录
   - 所有参数可在脚本中配置

2. **添加详细的输出信息**
   - 训练开始时显示 metrics 说明
   - 训练结束时显示 TensorBoard 使用指南
   - 列出所有可用的 metrics

3. **配置所有 Reward Function 参数**
   - Format bonuses (Type 1/2/3)
   - Length penalty settings
   - Diversity scaling parameters
   - Answer normalization options

## 🎯 可用的 TensorBoard Metrics

### 格式分布
- `format/type1_ratio`
- `format/type2_ratio`
- `format/type3_ratio`

### 每种类型的质量
- `format/type{1,2,3}_correct_rate`
- `format/type{1,2,3}_avg_length`

### Reward 组件
- `reward/base_mean`
- `reward/format_bonus_mean`
- `reward/length_scalar_mean`
- `reward/diversity_scalar_mean`
- `reward/total_mean`

### 总体统计
- `accuracy/overall`

### 窗口统计（检测收敛）
- `format/window_type{1,2,3}_ratio`
- `format/window_size`

## 🚀 使用方法

### 1. 运行验证（推荐）

```bash
# 创建并运行验证脚本
cat > verify_setup.sh << 'VERIFY_EOF'
#!/bin/bash
echo "🔍 Verifying setup..."
fail_count=0

tests=(
    "reward_functions/adaptive_reasoning_reward.py:Reward function"
    "verl/verl/workers/reward_manager/adaptive_reasoning.py:Reward manager"
    "train-grpo-v2.sh:Training script"
    "grpo_data/train.parquet:Training data"
    "LLaMA-Factory/saves/qwen2_5vl-3b/full/sft_9k:SFT model"
)

for test in "${tests[@]}"; do
    IFS=':' read -r path name <<< "$test"
    echo -n "Checking $name... "
    if [ -e "$path" ]; then
        echo "✓"
    else
        echo "✗ Missing!"
        fail_count=$((fail_count + 1))
    fi
done

if [ $fail_count -eq 0 ]; then
    echo "✅ All checks passed!"
else
    echo "❌ $fail_count check(s) failed!"
fi
VERIFY_EOF

chmod +x verify_setup.sh
./verify_setup.sh
```

### 2. 启动训练

```bash
./train-grpo-v2.sh --sft_model LLaMA-Factory/saves/qwen2_5vl-3b/full/sft_9k
```

### 3. 查看 TensorBoard

在另一个终端：

```bash
tensorboard --logdir saves/qwen2_5vl-3b/grpo/adaptive_reasoning_9k_v4
```

访问：http://localhost:6006

## 📊 监控指南

### 训练初期
重点关注：
- `format/type{1,2,3}_ratio` - 格式是否分布均匀？
- `accuracy/overall` - 准确率趋势如何？

### 训练中期
重点关注：
- `format/type{1,2,3}_correct_rate` - 哪种类型准确率最高？
- `reward/diversity_scalar_mean` - 是否接近 1.0（均匀分布）？
- `reward/length_scalar_mean` - 响应长度是否合理？

### 训练后期
重点关注：
- `format/window_type{1,2,3}_ratio` - 是否收敛到单一格式？
- 对比各类型的表现，调整超参数

## ⚙️ 参数调整

如需调整参数，编辑 `train-grpo-v2.sh` 中的对应行：

```bash
# 例如，增加 Type 3 的奖励
+reward_model.type3_format_bonus=0.3 \  # 原来是 0.2

# 例如，允许更长的响应
+reward_model.length_threshold=200 \    # 原来是 150
+reward_model.ideal_length=200.0 \      # 原来是 150
```

## 📚 相关文档

按推荐阅读顺序：

1. **QUICK_START_METRICS.md** - 快速开始（必读）
2. **TRAIN_GRPO_V2_CHANGES.md** - 脚本修改说明
3. **VERIFICATION_CHECKLIST.md** - 详细验证步骤
4. **METRICS_IMPLEMENTATION_SUMMARY.md** - 完整技术细节
5. **REWARD_METRICS_README.md** - Reward function 文档

## 🆚 对比：train-grpo.sh vs train-grpo-v2.sh

| 特性 | train-grpo.sh | train-grpo-v2.sh |
|------|---------------|------------------|
| TensorBoard Metrics | ❌ | ✅ 15+ metrics |
| Format Distribution | ❌ | ✅ Type 1/2/3 比例 |
| Per-Type Accuracy | ❌ | ✅ 各类型准确率 |
| Reward Breakdown | ❌ | ✅ 4 个组件 |
| 参数可配置性 | ❌ | ✅ 11 个参数 |
| 输出信息 | 基础 | 详细 |
| 适用场景 | 快速实验 | 深入分析 |

## ✨ 关键优势

使用 `train-grpo-v2.sh` 后，你能够：

✅ **实时监控** 模型在使用什么格式  
✅ **深入理解** 为什么某些样本获得高/低 reward  
✅ **数据驱动** 调整超参数而不是瞎猜  
✅ **及时发现** 格式收敛等训练问题  
✅ **持续优化** 训练策略以获得更好的格式分布  

## 🎓 示例分析场景

### 场景 1：模型只使用 Type 1
**现象：** `format/type1_ratio` 持续 > 0.8  
**分析：** Type 1 奖励过高或其他类型奖励不足  
**调整：** 增加 `type3_format_bonus`

### 场景 2：响应过长
**现象：** `reward/length_scalar_mean` < 0.6  
**分析：** 模型生成过长响应被惩罚  
**调整：** 增加 `ideal_length` 或 `length_threshold`

### 场景 3：Type 3 准确率低
**现象：** `format/type3_correct_rate` < `format/type1_correct_rate`  
**分析：** 模型在生成推理时容易出错  
**调整：** 可能需要更多 SFT 数据或调整 temperature

## 🔧 故障排除

### 训练启动失败

**检查 1：** Reward manager 是否正确加载？
```bash
python -c "from verl.workers.reward_manager import get_reward_manager_cls; print(get_reward_manager_cls('adaptive_reasoning'))"
```

**检查 2：** 训练数据是否存在？
```bash
ls -lh grpo_data/*.parquet
```

### Metrics 没有出现在 TensorBoard

**检查 1：** ray_trainer.py 是否包含修改？
```bash
grep "batch_metrics" verl/verl/trainer/ppo/ray_trainer.py
```

**检查 2：** 查看训练日志，确认 metrics 被计算

## 📞 获取帮助

如遇问题：
1. 查看 `VERIFICATION_CHECKLIST.md` 进行诊断
2. 查看 `QUICK_START_METRICS.md` 中的故障排除部分
3. 检查训练日志中的错误信息

---

## 🎯 准备开始？

**第一次使用？**
1. 阅读 `QUICK_START_METRICS.md`
2. 运行验证脚本
3. 启动训练
4. 打开 TensorBoard 观察 metrics

**已经熟悉？**
```bash
./train-grpo-v2.sh --sft_model LLaMA-Factory/saves/qwen2_5vl-3b/full/sft_9k
```

**祝训练顺利！** 🚀
