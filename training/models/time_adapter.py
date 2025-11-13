"""
Time-Step Conditioning Adapters for Hybrid Diffusion-Transformer

Lightweight LoRA-style adapters that inject timestep information into
frozen GPT-OSS-20B transformer layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal timestep embeddings (from Attention is All You Need)"""

    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps):
        """
        Args:
            timesteps: (batch_size,) tensor of timesteps in [0, num_timesteps-1]

        Returns:
            embeddings: (batch_size, dim) tensor of sinusoidal embeddings
        """
        half_dim = self.dim // 2
        # Compute frequency factors
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=timesteps.device) / half_dim
        )
        # Compute arguments for sin/cos
        args = timesteps.unsqueeze(-1).float() * freqs.unsqueeze(0)
        # Concatenate sin and cos embeddings
        embeddings = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        return embeddings


class TimeAdapter(nn.Module):
    """
    Time-conditioned adapter module using FiLM (Feature-wise Linear Modulation).

    Architecture:
        1. Sinusoidal timestep embedding
        2. LoRA-style adapter (low-rank bottleneck)
        3. FiLM conditioning (scale + shift modulation)

    Parameters:
        - hidden_dim: Hidden dimension of transformer (2880 for OSS20B)
        - adapter_dim: Bottleneck dimension (default: 128)
        - dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, hidden_dim=2880, adapter_dim=128, dropout=0.1, num_timesteps=1000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.adapter_dim = adapter_dim

        # Timestep embedding (sinusoidal)
        self.time_embed = SinusoidalEmbedding(adapter_dim)

        # LoRA-style adapter (down-project → activation → up-project)
        self.adapter_down = nn.Linear(hidden_dim, adapter_dim, bias=False)
        self.adapter_up = nn.Linear(adapter_dim, hidden_dim, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # FiLM conditioning layers (scale and shift)
        self.time_scale = nn.Linear(adapter_dim, hidden_dim)
        self.time_shift = nn.Linear(adapter_dim, hidden_dim)

        # Initialize with small weights (important for stability)
        nn.init.normal_(self.adapter_down.weight, std=0.02)
        nn.init.normal_(self.adapter_up.weight, std=0.02)
        nn.init.zeros_(self.time_scale.weight)
        nn.init.zeros_(self.time_scale.bias)
        nn.init.zeros_(self.time_shift.weight)
        nn.init.zeros_(self.time_shift.bias)

    def forward(self, hidden_states, timestep):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim) transformer hidden states
            timestep: (batch,) tensor of timesteps

        Returns:
            adapted_states: (batch, seq_len, hidden_dim) time-conditioned states
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Get timestep embeddings
        time_emb = self.time_embed(timestep)  # (batch, adapter_dim)

        # Compute FiLM modulation parameters
        scale = self.time_scale(time_emb)  # (batch, hidden_dim)
        shift = self.time_shift(time_emb)  # (batch, hidden_dim)

        # Apply FiLM conditioning: scale and shift
        # Broadcast to (batch, seq_len, hidden_dim)
        scale = scale.unsqueeze(1)  # (batch, 1, hidden_dim)
        shift = shift.unsqueeze(1)  # (batch, 1, hidden_dim)

        modulated = hidden_states * (1 + scale) + shift

        # Apply LoRA-style adapter
        adapter_out = self.adapter_down(modulated)  # (batch, seq_len, adapter_dim)
        adapter_out = self.activation(adapter_out)
        adapter_out = self.dropout(adapter_out)
        adapter_out = self.adapter_up(adapter_out)  # (batch, seq_len, hidden_dim)

        # Residual connection
        adapted_states = modulated + adapter_out

        return adapted_states

    def count_parameters(self):
        """Count trainable parameters in this adapter"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiLayerTimeAdapter(nn.Module):
    """
    Stack of time adapters for all transformer layers.

    This module manages adapters for all 24 layers of OSS20B.
    """

    def __init__(self, num_layers=24, hidden_dim=2880, adapter_dim=128, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers

        # Create adapter for each layer
        self.adapters = nn.ModuleList([
            TimeAdapter(hidden_dim, adapter_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states_list, timestep):
        """
        Apply time adaptation to hidden states from each layer.

        Args:
            hidden_states_list: List of (batch, seq_len, hidden_dim) tensors,
                               one for each transformer layer
            timestep: (batch,) tensor of timesteps

        Returns:
            adapted_list: List of adapted hidden states
        """
        adapted_list = []
        for layer_idx, hidden_states in enumerate(hidden_states_list):
            adapted = self.adapters[layer_idx](hidden_states, timestep)
            adapted_list.append(adapted)

        return adapted_list

    def count_parameters(self):
        """Count total trainable parameters across all adapters"""
        return sum(adapter.count_parameters() for adapter in self.adapters)


# Test the adapter
if __name__ == "__main__":
    print("Testing TimeAdapter...")

    # Test single adapter
    adapter = TimeAdapter(hidden_dim=2880, adapter_dim=128)
    print(f"Single adapter parameters: {adapter.count_parameters():,}")

    # Test forward pass
    batch_size = 4
    seq_len = 512
    hidden_dim = 2880

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    timesteps = torch.randint(0, 1000, (batch_size,))

    output = adapter(hidden_states, timesteps)
    assert output.shape == hidden_states.shape
    print(f"✓ Single adapter forward pass: {hidden_states.shape} -> {output.shape}")

    # Test multi-layer adapter
    multi_adapter = MultiLayerTimeAdapter(num_layers=24, hidden_dim=2880, adapter_dim=128)
    print(f"\nMulti-layer adapter (24 layers) parameters: {multi_adapter.count_parameters():,}")
    print(f"Percentage of 20B model: {multi_adapter.count_parameters() / 20e9 * 100:.3f}%")

    # Test multi-layer forward pass
    hidden_states_list = [torch.randn(batch_size, seq_len, hidden_dim) for _ in range(24)]
    output_list = multi_adapter(hidden_states_list, timesteps)
    assert len(output_list) == 24
    assert all(out.shape == hidden_states.shape for out in output_list)
    print(f"✓ Multi-layer adapter forward pass: 24 layers processed")

    # Memory footprint
    param_memory = multi_adapter.count_parameters() * 2 / 1024**2  # FP16 in MB
    print(f"\nMemory footprint (FP16): {param_memory:.1f} MB")
    print(f"Estimated training memory (with gradients + optimizer): {param_memory * 4:.1f} MB")

    print("\n✅ All tests passed!")
