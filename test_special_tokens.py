#!/usr/bin/env python3
"""
Test script to verify special tokens are working correctly.
"""

from transformers import AutoTokenizer

def test_tokenizer(tokenizer_path):
    """Test tokenizer with reasoning structure tokens."""

    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    print(f"\n{'='*60}")
    print(f"Tokenizer Information")
    print(f"{'='*60}")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Additional special tokens: {tokenizer.additional_special_tokens}")

    # Test text with reasoning structure
    test_text = """<perception>
The image shows a man walking towards the ocean while carrying a surfboard.
</perception>

<reasoning>
The surfboard was invented in 1926 by Tom Blake.
</reasoning>

<answer>
1926
</answer>"""

    print(f"\n{'='*60}")
    print(f"Test Text")
    print(f"{'='*60}")
    print(test_text)

    # Tokenize
    tokens = tokenizer.tokenize(test_text)
    token_ids = tokenizer.encode(test_text, add_special_tokens=False)

    print(f"\n{'='*60}")
    print(f"Tokenization Results")
    print(f"{'='*60}")
    print(f"Number of tokens: {len(tokens)}")
    print(f"\nFirst 20 tokens:")
    for i, (token, token_id) in enumerate(zip(tokens[:20], token_ids[:20])):
        print(f"  {i:2d}. {token:20s} (ID: {token_id})")

    # Verify special tokens are treated as single tokens
    special_tokens = ['<perception>', '</perception>', '<reasoning>', '</reasoning>', '<answer>', '</answer>']

    print(f"\n{'='*60}")
    print(f"Special Token Verification")
    print(f"{'='*60}")

    all_correct = True
    for special_token in special_tokens:
        token_list = tokenizer.tokenize(special_token)
        is_single = len(token_list) == 1 and token_list[0] == special_token

        status = "✓" if is_single else "✗"
        print(f"{status} {special_token:20s} -> {token_list}")

        if not is_single:
            all_correct = False

    # Test decoding with skip_special_tokens
    print(f"\n{'='*60}")
    print(f"Decoding Test")
    print(f"{'='*60}")

    encoded = tokenizer.encode(test_text, add_special_tokens=False)

    # Decode with special tokens
    decoded_with = tokenizer.decode(encoded, skip_special_tokens=False)
    print(f"Decoded WITH special tokens:")
    print(decoded_with)

    # Decode without special tokens (for generation output)
    decoded_without = tokenizer.decode(encoded, skip_special_tokens=True)
    print(f"\nDecoded WITHOUT special tokens (clean output):")
    print(decoded_without)

    print(f"\n{'='*60}")
    if all_correct:
        print("✓ All special tokens are working correctly!")
    else:
        print("✗ Some special tokens are NOT being treated as single tokens!")
        print("  Make sure you ran add_special_tokens.py first")
    print(f"{'='*60}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test reasoning structure special tokens')
    parser.add_argument('--tokenizer_path', type=str,
                        default='/nas03/yixuh/vlm-adaptive-resoning/models/qwen2.5-vl-7b-with-reasoning-tokens',
                        help='Path to tokenizer with special tokens')

    args = parser.parse_args()

    test_tokenizer(args.tokenizer_path)
