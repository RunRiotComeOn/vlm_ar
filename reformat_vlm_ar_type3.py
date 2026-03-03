"""
Reformat vlm_adaptive_reasoning/train_type3.json to match cold_start_9k format.

Changes:
- User message: "<image>Question text" -> "<image>\nQuestion: Question text"
- Saves reformatted data to vlm_adaptive_reasoning/train_type3_formatted.json
"""

import json
import re
import os

INPUT_PATH = "/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/data/vlm_adaptive_reasoning/train_type3.json"
OUTPUT_PATH = "/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/data/vlm_adaptive_reasoning/train_type3_formatted.json"

with open(INPUT_PATH, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples from {INPUT_PATH}")

reformatted = 0
for item in data:
    user_msg = item["messages"][0]["content"]
    # Pattern: "<image>Question text" -> "<image>\nQuestion: Question text"
    if user_msg.startswith("<image>") and not user_msg.startswith("<image>\nQuestion:"):
        question_text = user_msg[len("<image>"):]
        item["messages"][0]["content"] = f"<image>\nQuestion: {question_text}"
        reformatted += 1

print(f"Reformatted {reformatted}/{len(data)} user messages")

# Show a few examples
print("\nExamples after reformatting:")
for i in range(min(3, len(data))):
    print(f"  [{i}] {data[i]['messages'][0]['content'][:120]}")

with open(OUTPUT_PATH, "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\nSaved to {OUTPUT_PATH}")
