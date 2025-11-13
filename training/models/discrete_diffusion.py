"""
Discrete Token Diffusion Process

Implements forward (noising) and reverse (denoising) processes for discrete tokens.
Adapted for language modeling with masking and token replacement strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class DiscreteTokenDiffusion(nn.Module):
    """
    Discrete diffusion process for language modeling.

    Forward process: Gradually corrupt tokens by masking or random replacement
    Reverse process: Iteratively denoise to recover original tokens

    Noise schedule: Cosine schedule from improved DDPM (Nichol & Dhariwal 2021)
    """

    def __init__(
        self,
        vocab_size: int = 199036,
        num_timesteps: int = 1000,
        mask_token_id: int = 199036,  # Special [MASK] token
        noise_strategy: str = "mask",  # "mask", "random", or "hybrid"
        cosine_s: float = 0.008,  # Cosine schedule offset
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_timesteps = num_timesteps
        self.mask_token_id = mask_token_id
        self.noise_strategy = noise_strategy

        # Compute noise schedule (cosine schedule)
        betas = self._cosine_beta_schedule(num_timesteps, cosine_s)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Register as buffers (not trainable parameters)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        # Compute derived quantities
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine schedule from improved DDPM paper (Nichol & Dhariwal 2021).

        Args:
            timesteps: Number of diffusion steps
            s: Small offset to prevent beta from being too small near t=0

        Returns:
            betas: (timesteps,) tensor of beta values
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(
        self,
        x_0: torch.LongTensor,
        t: torch.LongTensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        """
        Forward diffusion: Add noise at timestep t.

        Args:
            x_0: (batch, seq_len) original token IDs
            t: (batch,) timesteps in [0, num_timesteps-1]
            mask: (batch, seq_len) optional mask of positions to corrupt
                  If None, all positions can be corrupted

        Returns:
            x_t: (batch, seq_len) corrupted token IDs
            corruption_mask: (batch, seq_len) bool mask of corrupted positions
        """
        batch_size, seq_len = x_0.shape
        device = x_0.device

        # Get corruption probability based on timestep
        # At t=0: ~0% corrupted, at t=T: ~100% corrupted
        corruption_probs = 1.0 - self.alphas_cumprod[t]  # (batch,)

        # Generate random corruption mask
        random_probs = torch.rand(batch_size, seq_len, device=device)
        corruption_mask = random_probs < corruption_probs.unsqueeze(-1)

        # Apply user-provided mask if specified
        if mask is not None:
            corruption_mask = corruption_mask & mask

        # Corrupt tokens based on strategy
        x_t = x_0.clone()

        if self.noise_strategy == "mask":
            # Replace with [MASK] token (BERT-style)
            x_t[corruption_mask] = self.mask_token_id

        elif self.noise_strategy == "random":
            # Replace with random tokens (more challenging)
            random_tokens = torch.randint(
                0, self.vocab_size, (batch_size, seq_len), device=device
            )
            x_t[corruption_mask] = random_tokens[corruption_mask]

        elif self.noise_strategy == "hybrid":
            # 50% mask, 50% random
            mask_or_random = torch.rand(batch_size, seq_len, device=device) < 0.5
            mask_positions = corruption_mask & mask_or_random
            random_positions = corruption_mask & ~mask_or_random

            x_t[mask_positions] = self.mask_token_id
            random_tokens = torch.randint(
                0, self.vocab_size, (batch_size, seq_len), device=device
            )
            x_t[random_positions] = random_tokens[random_positions]

        else:
            raise ValueError(f"Unknown noise strategy: {self.noise_strategy}")

        return x_t, corruption_mask

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.LongTensor,
        t: torch.LongTensor,
        edit_mask: Optional[torch.BoolTensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.LongTensor:
        """
        Reverse diffusion: Denoise at timestep t.

        Args:
            model: Hybrid model that takes (x_t, timestep, mode, edit_mask)
            x_t: (batch, seq_len) noised token IDs at timestep t
            t: (batch,) timesteps
            edit_mask: (batch, seq_len) optional mask of positions to update
                      If None, update all masked positions
            temperature: Sampling temperature
            top_k: Top-k sampling (optional)
            top_p: Nucleus sampling (optional)

        Returns:
            x_t_minus_1: (batch, seq_len) denoised tokens at timestep t-1
        """
        batch_size, seq_len = x_t.shape
        device = x_t.device

        # Get model predictions (logits over vocabulary)
        logits = model(x_t, timestep=t, mode="diffusion", edit_mask=edit_mask)
        # logits shape: (batch, seq_len, vocab_size)

        # Apply temperature
        logits = logits / temperature

        # Apply top-k filtering if specified
        if top_k is not None:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Apply nucleus (top-p) filtering if specified
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift to keep at least one token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter back to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        # Sample from predicted distribution
        probs = F.softmax(logits, dim=-1)  # (batch, seq_len, vocab_size)
        probs_2d = probs.view(-1, self.vocab_size)  # (batch*seq_len, vocab_size)
        sampled_tokens = torch.multinomial(probs_2d, 1).view(batch_size, seq_len)

        # Only update positions specified by edit_mask
        if edit_mask is not None:
            x_t_minus_1 = torch.where(edit_mask, sampled_tokens, x_t)
        else:
            # Update all [MASK] positions
            is_masked = (x_t == self.mask_token_id)
            x_t_minus_1 = torch.where(is_masked, sampled_tokens, x_t)

        return x_t_minus_1

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, int],
        context: Optional[torch.LongTensor] = None,
        edit_mask: Optional[torch.BoolTensor] = None,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        verbose: bool = False,
    ) -> torch.LongTensor:
        """
        Complete reverse diffusion loop.

        Args:
            model: Hybrid model
            shape: (batch_size, seq_len) for generated sequence
            context: Optional context tokens to condition on
            edit_mask: (batch, seq_len) positions to generate (True = generate)
            num_steps: Number of denoising steps (default: num_timesteps)
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            verbose: Print progress

        Returns:
            x_0: (batch, seq_len) generated tokens
        """
        batch_size, seq_len = shape
        device = next(model.parameters()).device

        if num_steps is None:
            num_steps = self.num_timesteps

        # Initialize with all [MASK] tokens
        x_t = torch.full((batch_size, seq_len), self.mask_token_id, dtype=torch.long, device=device)

        # If context is provided, initialize with context
        if context is not None:
            x_t = context.clone()
            # Only generate at masked positions
            if edit_mask is None:
                edit_mask = (context == self.mask_token_id)

        # Iterative denoising
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, dtype=torch.long, device=device)

        for i, t in enumerate(timesteps):
            if verbose and i % 10 == 0:
                print(f"Denoising step {i+1}/{num_steps}, t={t}")

            # Batch timestep
            t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)

            # Denoise
            x_t = self.p_sample(
                model, x_t, t_batch, edit_mask=edit_mask,
                temperature=temperature, top_k=top_k, top_p=top_p
            )

        return x_t

    def loss(
        self,
        model: nn.Module,
        x_0: torch.LongTensor,
        edit_mask: Optional[torch.BoolTensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute diffusion training loss (simplified ELBO).

        Args:
            model: Hybrid model
            x_0: (batch, seq_len) ground truth tokens
            edit_mask: (batch, seq_len) optional mask for loss computation
            reduction: "mean", "sum", or "none"

        Returns:
            loss: Scalar loss value (if reduction != "none")
        """
        batch_size, seq_len = x_0.shape
        device = x_0.device

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)

        # Add noise
        x_t, corruption_mask = self.q_sample(x_0, t, mask=edit_mask)

        # Predict original tokens
        logits = model(x_t, timestep=t, mode="diffusion", edit_mask=corruption_mask)
        # logits: (batch, seq_len, vocab_size)

        # Compute loss only on corrupted positions
        loss = F.cross_entropy(
            logits[corruption_mask].view(-1, self.vocab_size),
            x_0[corruption_mask].view(-1),
            reduction=reduction
        )

        return loss


# Test the diffusion process
if __name__ == "__main__":
    print("Testing DiscreteTokenDiffusion...")

    # Create diffusion process
    diffusion = DiscreteTokenDiffusion(
        vocab_size=199036,
        num_timesteps=1000,
        mask_token_id=199036,
        noise_strategy="mask"
    )

    # Test forward process (q_sample)
    batch_size = 4
    seq_len = 128
    x_0 = torch.randint(0, 199036, (batch_size, seq_len))
    t = torch.randint(0, 1000, (batch_size,))

    x_t, corruption_mask = diffusion.q_sample(x_0, t)
    assert x_t.shape == x_0.shape
    assert corruption_mask.shape == x_0.shape
    print(f"✓ Forward process: {x_0.shape} -> {x_t.shape}")
    print(f"  Corruption rate at t={t[0]}: {corruption_mask[0].float().mean():.1%}")

    # Test noise schedule
    print("\n✓ Noise schedule (cosine):")
    for t_val in [0, 250, 500, 750, 999]:
        t_tensor = torch.tensor([t_val])
        x_t, mask = diffusion.q_sample(x_0[:1], t_tensor)
        corruption_rate = mask[0].float().mean()
        print(f"  t={t_val:4d}: {corruption_rate:.1%} corrupted")

    # Test different noise strategies
    print("\n✓ Noise strategies:")
    for strategy in ["mask", "random", "hybrid"]:
        diffusion_strat = DiscreteTokenDiffusion(
            vocab_size=199036, num_timesteps=1000, noise_strategy=strategy
        )
        t_mid = torch.tensor([500])
        x_t, _ = diffusion_strat.q_sample(x_0[:1], t_mid)
        n_masked = (x_t[0] == 199036).sum()
        print(f"  {strategy:6s}: {n_masked}/{seq_len} [MASK] tokens")

    print("\n✅ All tests passed!")
