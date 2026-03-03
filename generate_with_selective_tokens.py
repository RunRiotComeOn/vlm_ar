#!/usr/bin/env python3
"""
Generation script that selectively hides only reasoning structure tokens,
while keeping other special tokens (like <|im_start|>, <|im_end|>) intact.
"""

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import re

# Define our custom reasoning tokens that we want to hide
REASONING_TOKENS = [
    '<perception>',
    '</perception>',
    '<reasoning>',
    '</reasoning>',
    '<answer>',
    '</answer>'
]

def remove_reasoning_tokens(text):
    """
    Remove only reasoning structure tokens from text.
    Keeps all other special tokens intact.
    """
    for token in REASONING_TOKENS:
        text = text.replace(token, '')
    return text.strip()

def extract_reasoning_components(text):
    """
    Extract individual reasoning components from structured output.

    Returns:
        dict with 'perception', 'reasoning', 'answer' keys
    """
    components = {}

    # Extract perception
    perception_match = re.search(r'<perception>(.*?)</perception>', text, re.DOTALL)
    if perception_match:
        components['perception'] = perception_match.group(1).strip()

    # Extract reasoning
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', text, re.DOTALL)
    if reasoning_match:
        components['reasoning'] = reasoning_match.group(1).strip()

    # Extract answer
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        components['answer'] = answer_match.group(1).strip()

    return components

def generate_with_selective_hiding(
    model_path,
    image_path,
    question,
    hide_reasoning_tokens=True,
    show_components=False
):
    """
    Generate answer with selective token hiding.

    Args:
        model_path: Path to fine-tuned model
        image_path: Path to input image
        question: Question to ask
        hide_reasoning_tokens: If True, hide only reasoning structure tokens
        show_components: If True, also show extracted components separately

    Returns:
        Generated text
    """

    print(f"Loading model from: {model_path}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Prepare input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question}
            ]
        }
    ]

    # Prepare for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    print(f"\nGenerating answer...")
    print(f"Question: {question}")

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    # Trim input from generated ids
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode keeping ALL special tokens first
    output_full = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=False,  # Keep ALL special tokens
        clean_up_tokenization_spaces=False
    )[0]

    # Now selectively remove only our reasoning tokens
    if hide_reasoning_tokens:
        output_clean = remove_reasoning_tokens(output_full)
    else:
        output_clean = output_full

    # Show results
    print(f"\n{'='*60}")
    if hide_reasoning_tokens:
        print("Output (reasoning tokens hidden):")
    else:
        print("Output (with reasoning structure):")
    print(f"{'='*60}")
    print(output_clean)
    print(f"{'='*60}")

    # Optionally show components
    if show_components:
        print(f"\n{'='*60}")
        print("Extracted Components:")
        print(f"{'='*60}")
        components = extract_reasoning_components(output_full)

        if 'perception' in components:
            print(f"\n[PERCEPTION]")
            print(components['perception'])

        if 'reasoning' in components:
            print(f"\n[REASONING]")
            print(components['reasoning'])

        if 'answer' in components:
            print(f"\n[ANSWER]")
            print(components['answer'])

        print(f"\n{'='*60}")

    return output_clean

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate with selective hiding of reasoning tokens'
    )
    parser.add_argument('--model_path', type=str,
                        default='/nas03/yixuh/vlm-adaptive-resoning/saves/qwen2_5vl-7b/full/sft_9k_with_tokens',
                        help='Path to fine-tuned model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--question', type=str, required=True,
                        help='Question to ask')
    parser.add_argument('--show_reasoning', action='store_true',
                        help='Keep reasoning structure tokens visible')
    parser.add_argument('--show_components', action='store_true',
                        help='Show extracted reasoning components separately')

    args = parser.parse_args()

    output = generate_with_selective_hiding(
        args.model_path,
        args.image,
        args.question,
        hide_reasoning_tokens=not args.show_reasoning,
        show_components=args.show_components
    )
