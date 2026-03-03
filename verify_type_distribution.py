#!/usr/bin/env python3
"""
Verify the type distribution of the SFT model under GRPO-like sampling conditions.

This script loads the SFT model and generates responses with the same
temperature/sampling parameters as GRPO training, then classifies them
into type1/2/3 to check the initial distribution.
"""

import re
import sys
import json
import random
import pandas as pd
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch


def get_response_type(response: str) -> int:
    """Classify response type based on tag presence."""
    has_perception = bool(re.search(r'<perception>.*?</perception>', response, re.DOTALL | re.IGNORECASE))
    has_reasoning = bool(re.search(r'<reasoning>.*?</reasoning>', response, re.DOTALL | re.IGNORECASE))

    if has_perception and has_reasoning:
        return 3
    elif has_perception:
        return 2
    else:
        return 1


def main():
    sft_model_path = "LLaMA-Factory/saves/qwen2_5vl-3b/type3_with_tokens"
    grpo_data_path = "grpo_data_new/train.parquet"
    num_samples = 20  # number of prompts to test
    num_generations = 8  # same as GRPO n=8
    temperature = 0.8  # same as GRPO
    top_p = 0.95  # same as GRPO
    max_new_tokens = 512  # shorter for quick test

    print(f"Loading tokenizer from {sft_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)

    # Check if special tokens are present
    print("\n=== Special Token Check ===")
    for tag in ['<perception>', '</perception>', '<reasoning>', '</reasoning>', '<answer>', '</answer>']:
        token_id = tokenizer.convert_tokens_to_ids(tag)
        is_special = tag in tokenizer.all_special_tokens
        print(f"  {tag}: token_id={token_id}, is_special={is_special}")

    # Test decoding with skip_special_tokens=True vs False
    test_ids = tokenizer.encode("<perception>test</perception>", add_special_tokens=False)
    decoded_true = tokenizer.decode(test_ids, skip_special_tokens=True)
    decoded_false = tokenizer.decode(test_ids, skip_special_tokens=False)
    print(f"\n  Decode test (skip_special_tokens=True):  '{decoded_true}'")
    print(f"  Decode test (skip_special_tokens=False): '{decoded_false}'")

    if decoded_true == decoded_false:
        print("  WARNING: No difference! Tags might not be treated as special tokens.")
    else:
        print(f"  IMPORTANT: Tags ARE stripped when skip_special_tokens=True!")
        print(f"  This means if vLLM decodes with skip_special_tokens=True, tags will be lost!")

    # Load GRPO data for prompts
    print(f"\nLoading GRPO data from {grpo_data_path}...")
    df = pd.read_parquet(grpo_data_path)

    # Sample some prompts (text-only for simplicity, skip image-heavy ones)
    sampled_indices = random.sample(range(len(df)), min(num_samples, len(df)))

    print(f"\nLoading model from {sft_model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        sft_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(sft_model_path, trust_remote_code=True)
    model.eval()

    type_counts = {1: 0, 2: 0, 3: 0}
    total = 0

    print(f"\n=== Generating {num_samples} prompts x {num_generations} samples each ===")
    print(f"Temperature: {temperature}, Top-p: {top_p}")
    print()

    for idx_i, idx in enumerate(sampled_indices):
        prompt_msgs = df['prompt'].iloc[idx]

        # Build messages for the model
        messages = []
        for msg in prompt_msgs:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Apply chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # For simplicity, skip image processing - just use text
        # This tests format generation ability, not visual understanding
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        for gen_i in range(num_generations):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                )

            # Decode response only (not prompt)
            response_ids = outputs[0][inputs['input_ids'].shape[1]:]

            # Test both decoding modes
            response_skip_true = tokenizer.decode(response_ids, skip_special_tokens=True)
            response_skip_false = tokenizer.decode(response_ids, skip_special_tokens=False)

            type_with_skip = get_response_type(response_skip_true)
            type_without_skip = get_response_type(response_skip_false)

            type_counts[type_without_skip] += 1
            total += 1

            # Print first few examples
            if idx_i < 3 and gen_i < 2:
                print(f"--- Prompt {idx_i}, Gen {gen_i} ---")
                print(f"  Q: {messages[-1]['content'][:100]}...")
                print(f"  Response (skip=False): {response_skip_false[:200]}...")
                print(f"  Type (skip=False): {type_without_skip}")
                if type_with_skip != type_without_skip:
                    print(f"  Response (skip=True): {response_skip_true[:200]}...")
                    print(f"  Type (skip=True):  {type_with_skip}  <-- DIFFERENT!")
                print()

    # Print results
    print("\n" + "=" * 60)
    print("=== TYPE DISTRIBUTION (skip_special_tokens=False) ===")
    print("=" * 60)
    for t in [1, 2, 3]:
        ratio = type_counts[t] / total if total > 0 else 0
        bar = "#" * int(ratio * 40)
        print(f"  Type {t}: {type_counts[t]:4d} / {total} = {ratio:.1%}  {bar}")
    print()

    if type_counts[3] / total > 0.6:
        print("CONCLUSION: SFT model strongly prefers type3 format.")
        print("  -> 3:3:3 ratio in GRPO is likely caused by something else (e.g., vLLM decoding).")
    elif type_counts[1] / total > 0.6:
        print("CONCLUSION: SFT model mostly generates type1 (no tags).")
        print("  -> SFT was too weak to teach the format.")
    else:
        print("CONCLUSION: SFT model output is mixed across types.")
        print("  -> This confirms the 3:3:3 ratio is from the model itself (weak SFT).")


if __name__ == "__main__":
    main()
