#!/usr/bin/env python3
"""
Minimal training script for hybrid diffusion adapters

This script demonstrates the basic training loop without multi-GPU complexity.
Use this for initial testing and validation.
"""

import torch
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import HybridGPTOSS20B
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def test_forward_pass():
    """Test that forward pass works"""
    print("\n" + "="*60)
    print("TEST 1: Forward Pass")
    print("="*60)

    # Load small model for testing
    print("Loading GPT-2 (small for testing)...")
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Create hybrid model
    print("Creating hybrid model...")
    model = HybridGPTOSS20B(
        base_model=base_model,
        freeze_base=True,
        adapter_dim=128,
        num_timesteps=1000,
        vocab_size=len(tokenizer),
    )

    if torch.cuda.is_available():
        model = model.cuda()
        print("✓ Model moved to GPU")

    # Test data
    text = "The quick brown fox jumps over the lazy dog."
    tokens = tokenizer(text, return_tensors="pt").input_ids

    if torch.cuda.is_available():
        tokens = tokens.cuda()

    # Test forward pass
    print("\nTesting forward pass...")
    timesteps = torch.randint(0, 1000, (tokens.size(0),))
    if torch.cuda.is_available():
        timesteps = timesteps.cuda()

    logits = model(tokens, timestep=timesteps, mode="diffusion")
    print(f"✓ Output shape: {logits.shape}")
    print(f"  Expected: ({tokens.size(0)}, {tokens.size(1)}, {len(tokenizer)})")

    return model, tokenizer


def test_loss_computation(model, tokenizer):
    """Test that loss computation works"""
    print("\n" + "="*60)
    print("TEST 2: Loss Computation")
    print("="*60)

    text = "Machine learning is transforming the world."
    tokens = tokenizer(text, return_tensors="pt").input_ids

    if torch.cuda.is_available():
        tokens = tokens.cuda()

    # Compute loss
    print("Computing loss...")
    loss = model.diffusion.loss(model, tokens)
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"  (Initial loss should be high, ~6-8)")

    return loss


def test_training_step(model, tokenizer):
    """Test a single training step"""
    print("\n" + "="*60)
    print("TEST 3: Training Step")
    print("="*60)

    # Create optimizer (only for adapters)
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=1e-4,
    )

    # Sample text
    text = "Artificial intelligence will change everything."
    tokens = tokenizer(text, return_tensors="pt", padding="max_length", max_length=64).input_ids

    if torch.cuda.is_available():
        tokens = tokens.cuda()

    # Training step
    print("Performing training step...")
    model.train()

    # Forward pass
    loss = model.diffusion.loss(model, tokens)
    initial_loss = loss.item()

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check that parameters updated
    print(f"✓ Backward pass completed")
    print(f"  Loss: {initial_loss:.4f}")

    # Do a few more steps to see if loss decreases
    print("\nRunning 10 training steps...")
    for step in range(10):
        loss = model.diffusion.loss(model, tokens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print(f"  Step {step}: Loss = {loss.item():.4f}")

    final_loss = loss.item()
    print(f"\n  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss:   {final_loss:.4f}")

    if final_loss < initial_loss:
        print("  ✓ Loss decreased! Training is working.")
    else:
        print("  ⚠️  Loss did not decrease (may need more steps)")


def test_generation(model, tokenizer):
    """Test generation modes"""
    print("\n" + "="*60)
    print("TEST 4: Generation")
    print("="*60)

    model.eval()

    # Test autoregressive generation
    print("Testing autoregressive generation...")
    prompt = "Once upon a time"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    with torch.no_grad():
        generated = model.generate_autoregressive(
            input_ids,
            max_new_tokens=20,
            temperature=0.8,
        )

    output = tokenizer.decode(generated[0])
    print(f"  Prompt: {prompt}")
    print(f"  Generated: {output}")

    # Test editing
    print("\nTesting diffusion editing...")
    text = "The cat sat on the mat"
    tokens = tokenizer(text, return_tensors="pt").input_ids

    # Edit positions 3 and 6 (random positions)
    edit_mask = torch.zeros_like(tokens, dtype=torch.bool)
    edit_mask[0, min(3, tokens.size(1)-1)] = True
    edit_mask[0, min(6, tokens.size(1)-1)] = True

    if torch.cuda.is_available():
        tokens = tokens.cuda()
        edit_mask = edit_mask.cuda()

    with torch.no_grad():
        edited = model.edit_text(
            tokens,
            edit_mask,
            num_steps=10,  # Fewer steps for testing
            temperature=0.8,
        )

    original = tokenizer.decode(tokens[0])
    edited_text = tokenizer.decode(edited[0])

    print(f"  Original: {original}")
    print(f"  Edited:   {edited_text}")


def main():
    """Run all tests"""
    print("="*60)
    print("MINIMAL TRAINING SCRIPT TEST")
    print("="*60)

    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  CUDA not available, using CPU (will be slow)")

    try:
        # Run tests
        model, tokenizer = test_forward_pass()
        test_loss_computation(model, tokenizer)
        test_training_step(model, tokenizer)
        test_generation(model, tokenizer)

        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✅")
        print("="*60)
        print("\nNext steps:")
        print("  1. Run full training: python scripts/train.py")
        print("  2. Monitor with TensorBoard")
        print("  3. Test on real dataset")

        return 0

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
