#!/usr/bin/env python3
"""
Demo script for hybrid diffusion-transformer

Showcases all capabilities:
1. Autoregressive generation (original GPT)
2. Smart editing (diffusion-based parallel editing)
3. Bidirectional infilling (fill in blanks)
4. Parallel generation (generate multiple tokens at once)
"""

import torch
import argparse
import sys
import os
import json
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import HybridGPTOSS20B
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    checkpoint_dir = Path(checkpoint_path)

    # Load config
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"✓ Loaded config from {config_path}")
    else:
        # Default config
        config = {
            'base_model': 'gpt2',
            'adapter_dim': 128,
            'num_timesteps': 1000,
            'vocab_size': 50257,
        }
        print(f"⚠️  No config found, using defaults")

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
    except:
        tokenizer = GPT2Tokenizer.from_pretrained(config['base_model'])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            config['base_model'],
            torch_dtype=torch.float16,
        )
    except:
        base_model = GPT2LMHeadModel.from_pretrained(config['base_model'])

    # Create hybrid model
    model = HybridGPTOSS20B(
        base_model=base_model,
        freeze_base=True,
        adapter_dim=config['adapter_dim'],
        num_timesteps=config['num_timesteps'],
        vocab_size=config['vocab_size'],
        mask_token_id=config['vocab_size'],
    )

    # Load weights
    model_path = checkpoint_dir / "model.pt"
    if model_path.exists():
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"✓ Loaded model weights from {model_path}")
    else:
        print(f"⚠️  No model weights found at {model_path}, using random initialization")

    model = model.to(device)
    model.eval()

    return model, tokenizer


def demo_autoregressive(model, tokenizer, prompt, max_tokens=50, temperature=0.8, device='cuda'):
    """Demo 1: Autoregressive generation (original GPT behavior)"""
    print("\n" + "="*70)
    print("DEMO 1: AUTOREGRESSIVE GENERATION")
    print("="*70)
    print(f"Prompt: {prompt}")
    print(f"Generating {max_tokens} tokens...")

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    start_time = time.time()

    with torch.no_grad():
        generated = model.generate_autoregressive(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
        )

    elapsed = time.time() - start_time

    output = tokenizer.decode(generated[0], skip_special_tokens=True)

    print(f"\n✓ Generated in {elapsed:.2f}s ({max_tokens/elapsed:.1f} tokens/sec)")
    print(f"\nOutput:\n{output}")
    print("="*70)


def demo_editing(model, tokenizer, text, edit_positions, num_steps=50, temperature=0.8, device='cuda'):
    """Demo 2: Smart editing with diffusion"""
    print("\n" + "="*70)
    print("DEMO 2: SMART EDITING")
    print("="*70)
    print(f"Original text: {text}")
    print(f"Editing positions: {edit_positions}")
    print(f"Denoising steps: {num_steps}")

    tokens = tokenizer(text, return_tensors="pt").input_ids.to(device)

    # Create edit mask
    edit_mask = torch.zeros_like(tokens, dtype=torch.bool)
    for pos in edit_positions:
        if pos < tokens.size(1):
            edit_mask[0, pos] = True

    # Show which tokens are being edited
    token_list = tokenizer.convert_ids_to_tokens(tokens[0])
    print(f"\nTokens being edited:")
    for pos in edit_positions:
        if pos < len(token_list):
            print(f"  Position {pos}: '{token_list[pos]}'")

    start_time = time.time()

    with torch.no_grad():
        edited_tokens = model.edit_text(
            tokens,
            edit_mask,
            num_steps=num_steps,
            temperature=temperature,
        )

    elapsed = time.time() - start_time

    original = tokenizer.decode(tokens[0], skip_special_tokens=True)
    edited = tokenizer.decode(edited_tokens[0], skip_special_tokens=True)

    print(f"\n✓ Edited in {elapsed:.2f}s")
    print(f"\nOriginal: {original}")
    print(f"Edited:   {edited}")
    print("="*70)


def demo_infilling(model, tokenizer, text_with_masks, num_steps=50, temperature=0.7, device='cuda'):
    """Demo 3: Bidirectional infilling"""
    print("\n" + "="*70)
    print("DEMO 3: BIDIRECTIONAL INFILLING")
    print("="*70)
    print(f"Text with blanks: {text_with_masks}")
    print(f"Denoising steps: {num_steps}")

    # Replace [MASK] with pad token (we'll use this as mask)
    # In practice, you might want a special mask token
    text_processed = text_with_masks.replace("[MASK]", tokenizer.pad_token)

    tokens = tokenizer(text_processed, return_tensors="pt").input_ids.to(device)

    # Create mask for pad tokens
    mask_positions = (tokens == tokenizer.pad_token_id)
    num_masks = mask_positions.sum().item()

    print(f"Number of positions to fill: {num_masks}")

    start_time = time.time()

    with torch.no_grad():
        # Use edit_text with mask positions
        filled_tokens = model.edit_text(
            tokens,
            mask_positions,
            num_steps=num_steps,
            temperature=temperature,
        )

    elapsed = time.time() - start_time

    filled = tokenizer.decode(filled_tokens[0], skip_special_tokens=True)

    print(f"\n✓ Filled in {elapsed:.2f}s")
    print(f"\nResult: {filled}")
    print("="*70)


def demo_parallel_generation(model, tokenizer, prompt, length=100, num_steps=50, temperature=0.8, device='cuda'):
    """Demo 4: Parallel generation (generate multiple tokens at once)"""
    print("\n" + "="*70)
    print("DEMO 4: PARALLEL GENERATION")
    print("="*70)
    print(f"Prompt: {prompt}")
    print(f"Generating {length} tokens in {num_steps} denoising steps")
    print(f"Expected speedup: ~{length/num_steps:.1f}× over autoregressive")

    # Encode prompt
    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_tokens.size(1)

    # Create full sequence with masks for positions to generate
    full_tokens = torch.cat([
        prompt_tokens,
        torch.full((1, length), tokenizer.pad_token_id, device=device)
    ], dim=1)

    # Mask positions after prompt
    edit_mask = torch.zeros_like(full_tokens, dtype=torch.bool)
    edit_mask[0, prompt_len:] = True

    print(f"Total length: {full_tokens.size(1)} (prompt: {prompt_len}, generate: {length})")

    start_time = time.time()

    with torch.no_grad():
        generated_tokens = model.edit_text(
            full_tokens,
            edit_mask,
            num_steps=num_steps,
            temperature=temperature,
        )

    elapsed = time.time() - start_time

    output = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    print(f"\n✓ Generated in {elapsed:.2f}s ({length/elapsed:.1f} tokens/sec)")
    print(f"\nOutput:\n{output}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Demo hybrid diffusion-transformer capabilities")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--demo", type=str, default="all",
                       choices=["all", "autoregressive", "editing", "infilling", "parallel"],
                       help="Which demo to run")

    # Demo-specific arguments
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt for generation")
    parser.add_argument("--text", type=str, default="The quick brown fox jumps over the lazy dog",
                       help="Text for editing")
    parser.add_argument("--edit-positions", type=int, nargs='+', default=[3, 8],
                       help="Positions to edit (space-separated)")
    parser.add_argument("--masked-text", type=str,
                       default="The capital of France is [MASK], and the capital of Spain is [MASK].",
                       help="Text with [MASK] for infilling")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens for autoregressive")
    parser.add_argument("--parallel-length", type=int, default=100, help="Length for parallel generation")
    parser.add_argument("--num-steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")

    args = parser.parse_args()

    print("="*70)
    print("HYBRID DIFFUSION-TRANSFORMER DEMO")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(args.checkpoint, device=args.device)
    print("✓ Model loaded successfully")

    # Run demos
    if args.demo in ["all", "autoregressive"]:
        demo_autoregressive(
            model, tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            device=args.device
        )

    if args.demo in ["all", "editing"]:
        demo_editing(
            model, tokenizer,
            text=args.text,
            edit_positions=args.edit_positions,
            num_steps=args.num_steps,
            temperature=args.temperature,
            device=args.device
        )

    if args.demo in ["all", "infilling"]:
        demo_infilling(
            model, tokenizer,
            text_with_masks=args.masked_text,
            num_steps=args.num_steps,
            temperature=args.temperature,
            device=args.device
        )

    if args.demo in ["all", "parallel"]:
        demo_parallel_generation(
            model, tokenizer,
            prompt=args.prompt,
            length=args.parallel_length,
            num_steps=args.num_steps,
            temperature=args.temperature,
            device=args.device
        )

    print("\n✓ All demos complete!")


if __name__ == "__main__":
    main()
