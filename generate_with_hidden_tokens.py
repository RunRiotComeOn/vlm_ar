#!/usr/bin/env python3
"""
Example script showing how to generate text with reasoning structure,
but hide the special tokens in the final output.
"""

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

def generate_with_hidden_tokens(
    model_path,
    image_path,
    question,
    show_reasoning=False
):
    """
    Generate answer with reasoning structure.

    Args:
        model_path: Path to fine-tuned model
        image_path: Path to input image
        question: Question to ask
        show_reasoning: If True, show reasoning structure; if False, hide it

    Returns:
        Generated text (with or without reasoning structure tags)
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

    # Decode with reasoning structure (internal processing)
    output_with_structure = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=False,  # Keep structure for analysis
        clean_up_tokenization_spaces=False
    )[0]

    # Decode for user-facing output (clean)
    output_clean = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,  # Hide special tokens
        clean_up_tokenization_spaces=True
    )[0]

    print(f"\n{'='*60}")
    if show_reasoning:
        print("Output WITH reasoning structure:")
        print(f"{'='*60}")
        print(output_with_structure)
    else:
        print("Output WITHOUT reasoning structure (clean):")
        print(f"{'='*60}")
        print(output_clean)
    print(f"{'='*60}")

    return output_clean if not show_reasoning else output_with_structure


def extract_reasoning_components(output_with_structure):
    """
    Extract individual components from structured output.

    Args:
        output_with_structure: Generated text with special tokens

    Returns:
        dict with 'perception', 'reasoning', 'answer' keys
    """
    import re

    components = {}

    # Extract perception
    perception_match = re.search(r'<perception>(.*?)</perception>', output_with_structure, re.DOTALL)
    if perception_match:
        components['perception'] = perception_match.group(1).strip()

    # Extract reasoning
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', output_with_structure, re.DOTALL)
    if reasoning_match:
        components['reasoning'] = reasoning_match.group(1).strip()

    # Extract answer
    answer_match = re.search(r'<answer>(.*?)</answer>', output_with_structure, re.DOTALL)
    if answer_match:
        components['answer'] = answer_match.group(1).strip()

    return components


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate with reasoning structure')
    parser.add_argument('--model_path', type=str,
                        default='/nas03/yixuh/vlm-adaptive-resoning/saves/qwen2_5vl-7b/full/sft_9k_with_tokens',
                        help='Path to fine-tuned model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--question', type=str, required=True,
                        help='Question to ask')
    parser.add_argument('--show_reasoning', action='store_true',
                        help='Show reasoning structure in output')

    args = parser.parse_args()

    # Example 1: Generate clean output (hide reasoning structure)
    print("\n" + "="*60)
    print("Example 1: Clean output (for end users)")
    print("="*60)
    clean_output = generate_with_hidden_tokens(
        args.model_path,
        args.image,
        args.question,
        show_reasoning=False
    )

    # Example 2: Generate with structure (for analysis/debugging)
    print("\n" + "="*60)
    print("Example 2: Structured output (for debugging)")
    print("="*60)
    structured_output = generate_with_hidden_tokens(
        args.model_path,
        args.image,
        args.question,
        show_reasoning=True
    )

    # Example 3: Extract components
    print("\n" + "="*60)
    print("Example 3: Extracted components")
    print("="*60)
    components = extract_reasoning_components(structured_output)
    for key, value in components.items():
        print(f"\n[{key.upper()}]")
        print(value)
