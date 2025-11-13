#!/usr/bin/env python3
"""
Test script to verify hybrid diffusion-transformer setup.

This script tests all core components to ensure everything is working correctly.
"""

import torch
import sys
from pathlib import Path

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import TimeAdapter, MultiLayerTimeAdapter, DiscreteTokenDiffusion


def test_gpu_availability():
    """Test GPU availability"""
    print("=" * 60)
    print("Testing GPU Availability")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False

    num_gpus = torch.cuda.device_count()
    print(f"✓ Found {num_gpus} GPU(s)")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f"  GPU {i}: {props.name}, {memory_gb:.1f}GB VRAM")

    return num_gpus >= 2  # Need at least 2 for training


def test_time_adapter():
    """Test TimeAdapter module"""
    print("\n" + "=" * 60)
    print("Testing TimeAdapter")
    print("=" * 60)

    try:
        # Create adapter
        adapter = TimeAdapter(hidden_dim=2880, adapter_dim=128)
        print(f"✓ Created TimeAdapter with {adapter.count_parameters():,} parameters")

        # Test forward pass
        batch_size, seq_len, hidden_dim = 4, 512, 2880
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        timesteps = torch.randint(0, 1000, (batch_size,))

        if torch.cuda.is_available():
            adapter = adapter.cuda()
            hidden_states = hidden_states.cuda()
            timesteps = timesteps.cuda()

        output = adapter(hidden_states, timesteps)
        assert output.shape == hidden_states.shape
        print(f"✓ Forward pass: {hidden_states.shape} -> {output.shape}")

        # Test multi-layer adapter
        multi_adapter = MultiLayerTimeAdapter(num_layers=24, hidden_dim=2880, adapter_dim=128)
        print(f"✓ Created MultiLayerTimeAdapter with {multi_adapter.count_parameters():,} parameters")
        print(f"  ({multi_adapter.count_parameters() / 20e9 * 100:.3f}% of 20B model)")

        return True

    except Exception as e:
        print(f"❌ TimeAdapter test failed: {e}")
        return False


def test_discrete_diffusion():
    """Test DiscreteTokenDiffusion module"""
    print("\n" + "=" * 60)
    print("Testing DiscreteTokenDiffusion")
    print("=" * 60)

    try:
        # Create diffusion process
        diffusion = DiscreteTokenDiffusion(
            vocab_size=199036,
            num_timesteps=1000,
            mask_token_id=199036,
            noise_strategy="mask"
        )
        print(f"✓ Created DiscreteTokenDiffusion with {diffusion.num_timesteps} steps")

        # Test forward process
        batch_size, seq_len = 4, 128
        x_0 = torch.randint(0, 199036, (batch_size, seq_len))
        t = torch.randint(0, 1000, (batch_size,))

        if torch.cuda.is_available():
            diffusion = diffusion.cuda()
            x_0 = x_0.cuda()
            t = t.cuda()

        x_t, corruption_mask = diffusion.q_sample(x_0, t)
        assert x_t.shape == x_0.shape
        assert corruption_mask.shape == x_0.shape
        print(f"✓ Forward process: {x_0.shape} -> {x_t.shape}")

        # Test noise schedule
        print("\n  Corruption rates at different timesteps:")
        for t_val in [0, 250, 500, 750, 999]:
            t_tensor = torch.full((1,), t_val, device=x_0.device)
            x_t_test, mask_test = diffusion.q_sample(x_0[:1], t_tensor)
            corruption_rate = mask_test[0].float().mean().item()
            print(f"    t={t_val:4d}: {corruption_rate:6.1%} corrupted")

        return True

    except Exception as e:
        print(f"❌ DiscreteTokenDiffusion test failed: {e}")
        return False


def test_memory_usage():
    """Test memory usage for training"""
    print("\n" + "=" * 60)
    print("Testing Memory Usage")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("⚠️  Skipping memory test (no CUDA)")
        return True

    try:
        # Create mock model components
        from transformers import GPT2Config, GPT2LMHeadModel

        print("Creating mock 20B model (scaled down for testing)...")
        config = GPT2Config(
            vocab_size=199036,
            n_positions=2048,
            n_embd=2880,
            n_layer=4,  # Scaled down from 24 for testing
            n_head=64,
        )
        base_model = GPT2LMHeadModel(config).cuda().half()  # FP16
        base_params = sum(p.numel() for p in base_model.parameters())
        print(f"  Base model: {base_params:,} parameters (scaled down)")

        # Create adapters
        adapter = MultiLayerTimeAdapter(num_layers=4, hidden_dim=2880, adapter_dim=128).cuda()
        adapter_params = adapter.count_parameters()
        print(f"  Adapters: {adapter_params:,} parameters")

        # Measure memory
        torch.cuda.reset_peak_memory_stats()
        batch_size, seq_len = 4, 512

        # Simulate forward pass
        input_ids = torch.randint(0, 199036, (batch_size, seq_len)).cuda()
        with torch.cuda.amp.autocast():
            outputs = base_model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            timesteps = torch.randint(0, 1000, (batch_size,)).cuda()
            adapted = adapter.adapters[0](hidden_states, timesteps)

        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n  Peak memory (batch_size={batch_size}): {peak_memory:.2f} GB")

        # Estimate full model memory
        full_model_factor = 24 / 4  # Scaling factor for full 24 layers
        estimated_full = peak_memory * full_model_factor * 1.2  # Add 20% buffer
        print(f"  Estimated full model (24 layers): {estimated_full:.2f} GB")

        if estimated_full < 45:
            print(f"  ✓ Should fit in 48GB L40S GPU")
        else:
            print(f"  ⚠️  May need optimization for 48GB GPU")

        return True

    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("HYBRID DIFFUSION-TRANSFORMER SETUP TEST")
    print("=" * 60)

    results = {}

    # Run tests
    results["GPU"] = test_gpu_availability()
    results["TimeAdapter"] = test_time_adapter()
    results["DiscreteTokenDiffusion"] = test_discrete_diffusion()
    results["Memory"] = test_memory_usage()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:25s}: {status}")
        all_passed = all_passed and passed

    print("=" * 60)

    if all_passed:
        print("\n🎉 All tests passed! Ready to train.")
        print("\nNext steps:")
        print("  1. Convert GPT-OSS-20B TensorRT model to PyTorch")
        print("  2. Run training: python scripts/train.py")
        print("  3. Test demos: python scripts/demo.py")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix issues before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
