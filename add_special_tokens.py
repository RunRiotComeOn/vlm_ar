#!/usr/bin/env python3
"""
Add special tokens for adaptive reasoning to Qwen2.5-VL tokenizer.
This should be run BEFORE training to properly add reasoning structure tokens.
"""

from transformers import AutoTokenizer, AutoConfig
import os

def add_reasoning_tokens(model_path, output_path=None):
    """
    Add <perception>, <reasoning>, <answer> tokens as special tokens.

    Args:
        model_path: Path to the base model
        output_path: Path to save modified tokenizer (default: same as model_path)
    """
    if output_path is None:
        output_path = model_path

    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Define new special tokens for reasoning structure
    new_special_tokens = [
        '<perception>',
        '</perception>',
        '<reasoning>',
        '</reasoning>',
        '<answer>',
        '</answer>'
    ]

    print(f"\nBefore adding tokens:")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Additional special tokens: {tokenizer.additional_special_tokens}")

    # Add new special tokens
    num_added = tokenizer.add_special_tokens({
        'additional_special_tokens': tokenizer.additional_special_tokens + new_special_tokens
    })

    print(f"\nAfter adding tokens:")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Tokens added: {num_added}")
    print(f"  New special tokens: {new_special_tokens}")
    print(f"  All additional special tokens: {tokenizer.additional_special_tokens}")

    # Verify the tokens are added correctly
    print(f"\nVerification:")
    for token in new_special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  {token}: ID={token_id}")

    # Save the modified tokenizer
    print(f"\nSaving modified tokenizer to {output_path}")
    tokenizer.save_pretrained(output_path)

    print(f"\n✓ Successfully added reasoning structure tokens!")
    print(f"\nNext steps:")
    print(f"1. Update your training config to use the modified tokenizer")
    print(f"2. Run training with the updated tokenizer")
    print(f"3. When generating, use skip_special_tokens=True to hide these tags")

    return tokenizer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Add reasoning structure tokens to Qwen2.5-VL tokenizer')
    parser.add_argument('--model_path', type=str,
                        default='Qwen/Qwen2.5-VL-7B-Instruct',
                        help='Path to base model')
    parser.add_argument('--output_path', type=str,
                        default='/nas03/yixuh/vlm-adaptive-resoning/models/qwen2.5-vl-7b-with-reasoning-tokens',
                        help='Path to save modified tokenizer')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    add_reasoning_tokens(args.model_path, args.output_path)
