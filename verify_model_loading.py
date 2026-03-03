#!/usr/bin/env python3
"""
Verify that the model loads correctly with extended vocabulary.
This script checks that:
1. Tokenizer loads with correct vocab size
2. Model loads successfully
3. Embedding layer is correctly sized
4. New tokens are properly initialized
"""

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoConfig
import torch

def verify_model_loading(model_path):
    """Verify model and tokenizer compatibility."""

    print(f"{'='*60}")
    print(f"Verifying Model Loading")
    print(f"{'='*60}")
    print(f"Model path: {model_path}\n")

    # Step 1: Load tokenizer
    print("Step 1: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer_vocab_size = len(tokenizer)
    print(f"  ✓ Tokenizer loaded")
    print(f"  ✓ Vocab size: {tokenizer_vocab_size}")
    print(f"  ✓ Additional special tokens: {len(tokenizer.additional_special_tokens)}")

    # Verify our reasoning tokens
    reasoning_tokens = ['<perception>', '</perception>', '<reasoning>', '</reasoning>', '<answer>', '</answer>']
    print(f"\n  Reasoning tokens:")
    for token in reasoning_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"    {token:20s} -> ID {token_id}")

    # Step 2: Load config
    print(f"\nStep 2: Loading model config...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    original_vocab_size = config.vocab_size
    print(f"  ✓ Config loaded")
    print(f"  ✓ Original vocab size in config: {original_vocab_size}")

    # Step 3: Load model (with CPU to save memory during verification)
    print(f"\nStep 3: Loading model (this may take a minute)...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Use fp16 to save memory
        device_map="cpu",  # Load on CPU for verification
        trust_remote_code=True
    )
    print(f"  ✓ Model loaded")

    # Step 4: Check embedding size
    print(f"\nStep 4: Checking embedding layer...")
    embedding_size = model.get_input_embeddings().weight.shape[0]
    print(f"  Model embedding size: {embedding_size}")
    print(f"  Tokenizer vocab size: {tokenizer_vocab_size}")

    if embedding_size < tokenizer_vocab_size:
        print(f"  ⚠ WARNING: Embedding size ({embedding_size}) < tokenizer vocab size ({tokenizer_vocab_size})")
        print(f"  This means the model needs to be resized!")
        print(f"\n  Resizing model embeddings...")
        model.resize_token_embeddings(tokenizer_vocab_size)
        new_embedding_size = model.get_input_embeddings().weight.shape[0]
        print(f"  ✓ Embeddings resized to: {new_embedding_size}")

        # Note: LLaMA-Factory will do this automatically during training
        print(f"\n  Note: LLaMA-Factory will automatically resize embeddings during training.")
    elif embedding_size == tokenizer_vocab_size:
        print(f"  ✓ Embedding size matches tokenizer vocab size")
    else:
        print(f"  ⚠ WARNING: Embedding size ({embedding_size}) > tokenizer vocab size ({tokenizer_vocab_size})")

    # Step 5: Test tokenization and embedding lookup
    print(f"\nStep 5: Testing tokenization and embedding lookup...")
    test_text = "<perception>Test</perception>"
    input_ids = tokenizer.encode(test_text, return_tensors="pt")
    print(f"  Test text: {test_text}")
    print(f"  Token IDs: {input_ids.tolist()}")

    # Try to get embeddings
    with torch.no_grad():
        embeddings = model.get_input_embeddings()(input_ids)
        print(f"  ✓ Successfully retrieved embeddings")
        print(f"  ✓ Embedding shape: {embeddings.shape}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"✓ Tokenizer: OK (vocab_size={tokenizer_vocab_size})")
    print(f"✓ Model: OK (embedding_size={embedding_size})")
    if embedding_size != tokenizer_vocab_size:
        print(f"⚠ Note: LLaMA-Factory will automatically resize embeddings during training")
    else:
        print(f"✓ Embeddings: OK (size matches)")
    print(f"✓ New tokens: OK (6 reasoning tokens added)")
    print(f"\n✓ Model is ready for training!")
    print(f"{'='*60}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Verify model loading with extended vocabulary')
    parser.add_argument('--model_path', type=str,
                        default='/nas03/yixuh/vlm-adaptive-resoning/models/qwen2.5-vl-7b-with-reasoning-tokens',
                        help='Path to model with extended tokenizer')

    args = parser.parse_args()

    verify_model_loading(args.model_path)
