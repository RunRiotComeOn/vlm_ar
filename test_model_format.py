#!/usr/bin/env python3
"""
测试模型是否学会了训练数据的输出格式
检查 <perception>、<reasoning>、<answer> 标签
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import json
from PIL import Image
import os
import re

def load_model(model_path):
    """加载模型"""
    print(f"加载模型: {model_path}")

    # 使用 AutoModel 自动识别架构
    from transformers import AutoModelForVision2Seq

    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"使用 AutoModelForVision2Seq 失败: {e}")
        print("尝试直接使用 Qwen2VLForConditionalGeneration...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            ignore_mismatched_sizes=True  # 忽略大小不匹配
        )

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    return model, processor


def test_model_format(model, processor, test_samples, output_dir="test_outputs"):
    """测试模型输出格式"""

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for idx, sample in enumerate(test_samples):
        print(f"\n{'='*80}")
        print(f"测试样本 {idx + 1}/{len(test_samples)}")
        print(f"{'='*80}")

        # 构建消息 - 已包含图像路径
        messages = sample['messages']

        # 处理输入 - processor 会自动处理图像
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 从消息中提取图像路径
        image_inputs = []
        for msg in messages:
            if isinstance(msg.get('content'), list):
                for item in msg['content']:
                    if item.get('type') == 'image':
                        img_path = item.get('image')
                        if img_path and os.path.exists(img_path):
                            image_inputs.append(Image.open(img_path))

        inputs = processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # 生成输出
        print("生成中...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # 解码输出 - 测试两种方式
        output_with_special = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )[0]

        output_without_special = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # 分析输出
        result = {
            'sample_idx': idx + 1,
            'question': messages[0]['content'][-1]['text'][:100] + "...",
            'output_with_special_tokens': output_with_special,
            'output_without_special_tokens': output_without_special,
            'analysis': {
                'with_special': analyze_format(output_with_special),
                'without_special': analyze_format(output_without_special)
            }
        }

        results.append(result)

        # 打印结果
        print(f"\n问题: {result['question']}")
        print(f"\n--- skip_special_tokens=False ---")
        print(f"输出长度: {len(output_with_special)} 字符")
        print(f"格式分析: {result['analysis']['with_special']}")
        print(f"输出前500字符:\n{output_with_special[:500]}")
        print(f"输出后200字符:\n...{output_with_special[-200:]}")

        print(f"\n--- skip_special_tokens=True ---")
        print(f"输出长度: {len(output_without_special)} 字符")
        print(f"格式分析: {result['analysis']['without_special']}")
        print(f"输出前500字符:\n{output_without_special[:500]}")
        print(f"输出后200字符:\n...{output_without_special[-200:]}")

        # 保存详细结果
        with open(f"{output_dir}/sample_{idx+1}_detail.txt", 'w') as f:
            f.write(f"Sample {idx+1}\n")
            f.write(f"Question: {result['question']}\n\n")
            f.write("="*80 + "\n")
            f.write("WITH special tokens (skip_special_tokens=False):\n")
            f.write("="*80 + "\n")
            f.write(output_with_special)
            f.write("\n\n" + "="*80 + "\n")
            f.write("WITHOUT special tokens (skip_special_tokens=True):\n")
            f.write("="*80 + "\n")
            f.write(output_without_special)

    # 保存总结
    save_summary(results, output_dir)

    return results


def analyze_format(text):
    """分析输出格式"""
    analysis = {
        'has_perception': '<perception>' in text.lower(),
        'has_reasoning': '<reasoning>' in text.lower(),
        'has_answer': '<answer>' in text.lower(),
        'perception_count': text.lower().count('<perception>'),
        'reasoning_count': text.lower().count('<reasoning>'),
        'answer_count': text.lower().count('<answer>'),
        'length': len(text)
    }

    # 检查格式是否正确
    if analysis['has_perception'] and analysis['has_reasoning'] and analysis['has_answer']:
        if analysis['answer_count'] == 1:
            analysis['format_correct'] = True
            analysis['format_status'] = "✓ 完全正确"
        else:
            analysis['format_correct'] = False
            analysis['format_status'] = f"✗ answer标签数量错误 ({analysis['answer_count']}个)"
    else:
        analysis['format_correct'] = False
        missing = []
        if not analysis['has_perception']:
            missing.append('perception')
        if not analysis['has_reasoning']:
            missing.append('reasoning')
        if not analysis['has_answer']:
            missing.append('answer')
        analysis['format_status'] = f"✗ 缺少标签: {', '.join(missing)}"

    return analysis


def save_summary(results, output_dir):
    """保存总结报告"""
    with open(f"{output_dir}/summary.txt", 'w') as f:
        f.write("="*80 + "\n")
        f.write("模型输出格式测试总结\n")
        f.write("="*80 + "\n\n")

        # 统计 skip_special_tokens=False 的结果
        f.write("=== skip_special_tokens=False 结果 ===\n\n")
        correct_with = sum(1 for r in results if r['analysis']['with_special']['format_correct'])
        f.write(f"格式正确的样本数: {correct_with}/{len(results)}\n")
        f.write(f"正确率: {correct_with/len(results)*100:.1f}%\n\n")

        for r in results:
            f.write(f"样本 {r['sample_idx']}: {r['analysis']['with_special']['format_status']}\n")

        # 统计 skip_special_tokens=True 的结果
        f.write("\n=== skip_special_tokens=True 结果 ===\n\n")
        correct_without = sum(1 for r in results if r['analysis']['without_special']['format_correct'])
        f.write(f"格式正确的样本数: {correct_without}/{len(results)}\n")
        f.write(f"正确率: {correct_without/len(results)*100:.1f}%\n\n")

        for r in results:
            f.write(f"样本 {r['sample_idx']}: {r['analysis']['without_special']['format_status']}\n")

        # 结论
        f.write("\n" + "="*80 + "\n")
        f.write("结论\n")
        f.write("="*80 + "\n\n")

        if correct_with > correct_without:
            f.write("✓ 模型学会了格式，但需要 skip_special_tokens=False 来正确解码\n")
        elif correct_without > 0:
            f.write("✓ 模型学会了格式，使用 skip_special_tokens=True 即可\n")
        else:
            f.write("✗ 模型没有学会训练数据的格式\n")
            f.write("   可能原因:\n")
            f.write("   1. 训练数据格式与实际训练时使用的格式不一致\n")
            f.write("   2. 训练配置中 special tokens 设置有问题\n")
            f.write("   3. 训练 epochs 不足\n")
            f.write("   4. tokenizer 配置问题\n")

    print(f"\n总结已保存到: {output_dir}/summary.txt")


def load_test_samples(train_data_path, num_samples=3):
    """从训练数据中加载测试样本"""
    print(f"从训练数据加载测试样本: {train_data_path}")

    with open(train_data_path, 'r') as f:
        data = json.load(f)

    # 选择前 num_samples 个样本
    test_samples = []
    for i in range(min(num_samples, len(data))):
        sample = data[i]

        # 构建 user 消息，将 <image> 占位符转换为实际的图像对象
        user_messages = []
        for msg in sample['messages']:
            if msg['role'] == 'user':
                content = msg['content']

                # 将字符串内容转换为消息格式
                # Qwen2-VL 期望的格式是 [{"type": "image"}, {"type": "text", "text": "..."}]
                message_content = []

                # 检查是否有 <image> 占位符
                if '<image>' in content:
                    # 添加图像
                    for img_path in sample.get('images', []):
                        full_path = os.path.join('/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/data', img_path)
                        message_content.append({"type": "image", "image": full_path})

                    # 移除 <image> 占位符并添加文本
                    text_content = content.replace('<image>', '').strip()
                    if text_content:
                        message_content.append({"type": "text", "text": text_content})
                else:
                    # 没有图像，只有文本
                    message_content.append({"type": "text", "text": content})

                user_messages.append({
                    "role": "user",
                    "content": message_content
                })

        test_sample = {
            'messages': user_messages,
            'expected_output': [msg['content'] for msg in sample['messages'] if msg['role'] == 'assistant'][0],
            'images': sample.get('images', [])
        }

        test_samples.append(test_sample)

    print(f"加载了 {len(test_samples)} 个测试样本")
    return test_samples


if __name__ == "__main__":
    # 配置
    MODEL_PATH = "/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/saves/qwen2_5vl-3b/curriculum/phase2"
    TRAIN_DATA_PATH = "/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/data/cold_start_9k/train_type2.json"
    OUTPUT_DIR = "/nas03/yixuh/vlm-adaptive-resoning/test_outputs/phase2_type3"
    NUM_TEST_SAMPLES = 5  # 减少样本数加快测试

    print("="*80)
    print("模型输出格式测试")
    print("="*80)
    print(f"模型路径: {MODEL_PATH}")
    print(f"训练数据: {TRAIN_DATA_PATH}")
    print(f"测试样本数: {NUM_TEST_SAMPLES}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*80)

    # 加载模型
    model, processor = load_model(MODEL_PATH)

    # 加载测试样本
    test_samples = load_test_samples(TRAIN_DATA_PATH, NUM_TEST_SAMPLES)

    # 测试模型
    results = test_model_format(model, processor, test_samples, OUTPUT_DIR)

    print("\n" + "="*80)
    print("测试完成！")
    print(f"详细结果保存在: {OUTPUT_DIR}")
    print("="*80)
