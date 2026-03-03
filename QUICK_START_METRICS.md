# 快速开始：使用带 TensorBoard Metrics 的 Reward Function

## 🎯 目标

在 GRPO 训练中实时监控响应质量指标：
- Type 1/2/3 格式分布
- 每种类型的准确率
- 平均响应长度
- Reward 组件分解

## ✅ 已完成的修改

1. ✅ 创建了支持 metrics 的 reward function (V4)
2. ✅ 创建了自定义 Reward Manager
3. ✅ 修改了 verl 的 ray_trainer.py 以支持 batch metrics
4. ✅ 注册了新的 reward manager

## 🚀 使用方法

### 修改训练脚本

编辑 `train-grpo.sh`，将：

```bash
custom_reward_function.path=../reward_functions/adaptive_reasoning_reward.py \
custom_reward_function.name=create_reward_function \
```

**替换为：**

```bash
reward_model.reward_manager=adaptive_reasoning \
```

### 可选：自定义参数

如果需要调整 reward function 参数，可以添加：

```bash
reward_model.reward_manager=adaptive_reasoning \
+reward_model.type1_format_bonus=0.0 \
+reward_model.type2_format_bonus=0.1 \
+reward_model.type3_format_bonus=0.2 \
+reward_model.length_threshold=150 \
+reward_model.ideal_length=150.0 \
+reward_model.enable_diversity_scaling=True \
+reward_model.diversity_weight=0.3 \
```

## 📊 TensorBoard 中的 Metrics

### 启动训练

```bash
./train-grpo.sh --sft_model LLaMA-Factory/saves/qwen2_5vl-3b/full/sft_9k
```

### 查看 TensorBoard

在另一个终端中：

```bash
tensorboard --logdir saves/qwen2_5vl-3b/grpo/adaptive_reasoning_9k_v3
```

然后访问: `http://localhost:6006`

### Metrics 说明

**Format Distribution（格式分布）** - 在 `format/` 下：
- `type1_ratio`: Type 1（直接答案）占比
- `type2_ratio`: Type 2（感知 + 答案）占比
- `type3_ratio`: Type 3（完整推理）占比

**Per-Type Quality（每种类型的质量）** - 在 `format/` 下：
- `type1_correct_rate`: Type 1 准确率
- `type2_correct_rate`: Type 2 准确率
- `type3_correct_rate`: Type 3 准确率
- `type1_avg_length`: Type 1 平均长度
- `type2_avg_length`: Type 2 平均长度
- `type3_avg_length`: Type 3 平均长度

**Reward Components（Reward 组件）** - 在 `reward/` 下：
- `base_mean`: 平均基础 reward
- `format_bonus_mean`: 平均格式奖励
- `length_scalar_mean`: 平均长度系数
- `diversity_scalar_mean`: 平均多样性系数
- `total_mean`: 平均总 reward

**Overall（总体）** - 在 `accuracy/` 下：
- `overall`: 总体准确率

## 🔍 监控建议

### 1. 检查格式分布
观察 `format/type{1,2,3}_ratio`，确保：
- 不要过度偏向某一种类型
- 如果 Type 3 比例很低，可能需要增加 `type3_format_bonus`

### 2. 检查准确率
对比 `format/type{1,2,3}_correct_rate`：
- 如果某种类型准确率明显偏低，可能需要调整训练策略
- 理想情况下，Type 3 的准确率应该最高（因为包含推理过程）

### 3. 检查 Diversity Scaling
观察 `reward/diversity_scalar_mean`：
- 接近 1.0：格式分布均匀 ✅
- < 0.8 或 > 1.2：格式分布不均匀，diversity scaling 正在发挥作用

### 4. 检查长度惩罚
观察 `reward/length_scalar_mean`：
- = 1.0：大部分响应长度适中 ✅
- < 0.8：响应普遍过长，受到较重惩罚
- 如果希望允许更长的响应，可以增加 `ideal_length`

## 🛠️ 故障排除

### Metrics 没有出现在 TensorBoard

**检查 1**: 确认使用了新的 reward manager
```bash
grep "reward_manager=adaptive_reasoning" train-grpo.sh
```

**检查 2**: 查看训练日志
日志中应该出现：
```
[prompt] ...
[response] ...
[ground_truth] ...
```

**检查 3**: 确认 verl 修改生效
```bash
grep "batch_metrics" verl/verl/trainer/ppo/ray_trainer.py
```
应该能找到我们添加的代码。

### 训练失败：找不到 adaptive_reasoning_reward

**解决方案**: 确保文件路径正确
```bash
ls reward_functions/adaptive_reasoning_reward.py
ls verl/verl/workers/reward_manager/adaptive_reasoning.py
```

## 📝 下一步

1. 运行测试以验证 metrics 是否正常工作
2. 根据 TensorBoard 中的指标调整超参数
3. 比较不同配置下的格式分布和准确率

## 📚 详细文档

更多信息请参考：
- `REWARD_METRICS_README.md` - 完整的功能说明
- `reward_functions/adaptive_reasoning_reward.py` - 源代码和实现细节

## ✨ 关键优势

相比之前的版本，现在你可以：
- ✅ 实时看到模型使用哪种响应格式
- ✅ 监控每种格式的质量（准确率、长度）
- ✅ 理解 reward 组件如何影响模型行为
- ✅ 基于数据调整超参数，而不是瞎猜
- ✅ 发现格式收敛问题（diversity scaling 会自动调整）

---

**准备好了吗？** 现在就修改 `train-grpo.sh` 并启动训练！🚀
