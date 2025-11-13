"""
Hybrid Transformer-Diffusion Model

Wraps GPT-OSS-20B with time adapters for dual-mode operation:
- Mode 1: Autoregressive generation (original GPT)
- Mode 2: Diffusion editing/infilling (new capability)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
from .time_adapter import MultiLayerTimeAdapter
from .discrete_diffusion import DiscreteTokenDiffusion


class HybridGPTOSS20B(nn.Module):
    """
    Hybrid model combining frozen GPT-OSS-20B with diffusion adapters.

    Architecture:
        - Base: Frozen GPT-OSS-20B (20B parameters)
        - Adapters: Trainable time-conditioned adapters (~35M parameters, 0.18%)
        - Modes: Autoregressive (causal) or Diffusion (bidirectional)
    """

    def __init__(
        self,
        base_model: nn.Module,  # Pretrained GPT-OSS-20B
        freeze_base: bool = True,
        adapter_dim: int = 128,
        adapter_dropout: float = 0.1,
        num_timesteps: int = 1000,
        vocab_size: int = 199036,
        mask_token_id: int = 199036,
    ):
        super().__init__()

        # Store base model
        self.base_model = base_model
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id

        # Freeze base model if specified
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            print(f"✓ Frozen base model parameters")

        # Get model config from base model
        self.config = base_model.config if hasattr(base_model, 'config') else None
        self.num_layers = getattr(self.config, 'num_hidden_layers', 24)
        self.hidden_dim = getattr(self.config, 'hidden_size', 2880)

        # Create time adapters
        self.time_adapters = MultiLayerTimeAdapter(
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            adapter_dim=adapter_dim,
            dropout=adapter_dropout
        )

        # Create diffusion process
        self.diffusion = DiscreteTokenDiffusion(
            vocab_size=vocab_size,
            num_timesteps=num_timesteps,
            mask_token_id=mask_token_id,
            noise_strategy="mask"
        )

        # Current mode
        self.mode = "autoregressive"

        print(f"✓ Initialized HybridGPTOSS20B")
        print(f"  Base model: {self.count_base_parameters():,} parameters (frozen: {freeze_base})")
        print(f"  Adapters: {self.time_adapters.count_parameters():,} parameters")
        print(f"  Total trainable: {self.count_trainable_parameters():,} parameters")

    def set_mode(self, mode: Literal["autoregressive", "diffusion"]):
        """Set generation mode"""
        assert mode in ["autoregressive", "diffusion"]
        self.mode = mode

    def get_attention_mask(
        self,
        seq_len: int,
        mode: str,
        edit_mask: Optional[torch.BoolTensor] = None,
        device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """
        Generate attention mask based on mode.

        Args:
            seq_len: Sequence length
            mode: "autoregressive" or "diffusion"
            edit_mask: (batch, seq_len) positions being edited (for diffusion mode)
            device: Device for mask tensor

        Returns:
            attention_mask: (seq_len, seq_len) or (batch, seq_len, seq_len)
        """
        if mode == "autoregressive":
            # Causal mask (lower triangular)
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device), diagonal=1
            ).bool()
            # Convert to attention mask (0 for attend, -inf for mask)
            attention_mask = mask.masked_fill(mask, float('-inf')).masked_fill(~mask, 0.0)

        elif mode == "diffusion":
            if edit_mask is None:
                # Full bidirectional attention
                attention_mask = torch.zeros(seq_len, seq_len, device=device)
            else:
                # Hybrid attention: known tokens use causal, edited tokens bidirectional
                batch_size = edit_mask.shape[0]
                attention_mask = torch.zeros(batch_size, seq_len, seq_len, device=device)

                for i in range(seq_len):
                    # If this position is known (not being edited), use causal mask
                    known_mask = ~edit_mask[:, i]  # (batch,)
                    # Mask future positions for known tokens
                    attention_mask[known_mask, i, i+1:] = float('-inf')

        else:
            raise ValueError(f"Unknown mode: {mode}")

        return attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor,
        timestep: Optional[torch.LongTensor] = None,
        mode: Optional[str] = None,
        edit_mask: Optional[torch.BoolTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ):
        """
        Forward pass with dual-mode support.

        Args:
            input_ids: (batch, seq_len) token IDs
            timestep: (batch,) timesteps for diffusion mode (required if mode="diffusion")
            mode: "autoregressive" or "diffusion" (overrides self.mode if provided)
            edit_mask: (batch, seq_len) positions being edited (for diffusion mode)
            attention_mask: Optional custom attention mask
            output_hidden_states: Return intermediate hidden states

        Returns:
            logits: (batch, seq_len, vocab_size) output logits
            hidden_states: List of hidden states (if output_hidden_states=True)
        """
        if mode is None:
            mode = self.mode

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Generate attention mask if not provided
        if attention_mask is None:
            attention_mask = self.get_attention_mask(seq_len, mode, edit_mask, device)

        # Get base model outputs
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # Always get hidden states for adapters
            return_dict=True
        )

        # Get hidden states from all layers
        # Note: base_outputs.hidden_states includes embedding layer + all transformer layers
        # We want only the transformer layer outputs (skip embedding)
        all_hidden_states = base_outputs.hidden_states[1:]  # Skip embedding layer

        # Apply time adapters in diffusion mode
        if mode == "diffusion":
            if timestep is None:
                raise ValueError("timestep is required for diffusion mode")

            # Apply time conditioning to each layer's hidden states
            adapted_hidden_states = self.time_adapters(all_hidden_states, timestep)

            # Use the last layer's adapted hidden states for prediction
            final_hidden = adapted_hidden_states[-1]
        else:
            # Autoregressive mode: use original hidden states
            final_hidden = all_hidden_states[-1]

        # Project to vocabulary logits
        if hasattr(self.base_model, 'lm_head'):
            logits = self.base_model.lm_head(final_hidden)
        elif hasattr(self.base_model, 'score'):
            logits = self.base_model.score(final_hidden)
        else:
            # Fallback: assume tied embeddings
            logits = F.linear(final_hidden, self.base_model.get_input_embeddings().weight)

        if output_hidden_states:
            return logits, adapted_hidden_states if mode == "diffusion" else all_hidden_states
        else:
            return logits

    @torch.no_grad()
    def generate_autoregressive(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Standard autoregressive generation (original GPT behavior).

        Args:
            input_ids: (batch, seq_len) prompt tokens
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            eos_token_id: End-of-sequence token ID

        Returns:
            generated_ids: (batch, seq_len + new_tokens) generated sequence
        """
        self.set_mode("autoregressive")

        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Forward pass (only last token logits needed)
            logits = self(generated, mode="autoregressive")
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply nucleus (top-p) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    @torch.no_grad()
    def edit_text(
        self,
        input_ids: torch.LongTensor,
        edit_mask: torch.BoolTensor,
        num_steps: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.LongTensor:
        """
        Diffusion-based text editing.

        Args:
            input_ids: (batch, seq_len) original tokens
            edit_mask: (batch, seq_len) positions to edit (True = edit)
            num_steps: Number of denoising steps
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            edited_ids: (batch, seq_len) edited tokens
        """
        self.set_mode("diffusion")

        # Use diffusion process for iterative editing
        edited_ids = self.diffusion.p_sample_loop(
            model=self,
            shape=input_ids.shape,
            context=input_ids,
            edit_mask=edit_mask,
            num_steps=num_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        return edited_ids

    @torch.no_grad()
    def infill(
        self,
        input_ids: torch.LongTensor,
        num_steps: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.LongTensor:
        """
        Bidirectional infilling (fill [MASK] tokens).

        Args:
            input_ids: (batch, seq_len) tokens with [MASK] positions
            num_steps: Number of denoising steps
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            filled_ids: (batch, seq_len) filled tokens
        """
        self.set_mode("diffusion")

        # Mask positions are edit positions
        mask_positions = (input_ids == self.mask_token_id)

        # Use diffusion editing
        filled_ids = self.edit_text(
            input_ids=input_ids,
            edit_mask=mask_positions,
            num_steps=num_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        return filled_ids

    def count_base_parameters(self) -> int:
        """Count parameters in base model"""
        return sum(p.numel() for p in self.base_model.parameters())

    def count_trainable_parameters(self) -> int:
        """Count trainable parameters (adapters only if base is frozen)"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_trainable_parameters(self):
        """Get trainable parameters for optimizer"""
        return [p for p in self.parameters() if p.requires_grad]


# Test with a mock base model
if __name__ == "__main__":
    print("Testing HybridGPTOSS20B...")

    # Create a mock GPT model for testing
    from transformers import GPT2Config, GPT2LMHeadModel

    print("\n1. Creating mock GPT-2 model (for testing)...")
    config = GPT2Config(
        vocab_size=199036,
        n_positions=2048,
        n_embd=2880,
        n_layer=24,
        n_head=64,
    )
    base_model = GPT2LMHeadModel(config)
    print(f"   Mock model parameters: {sum(p.numel() for p in base_model.parameters()):,}")

    print("\n2. Creating hybrid model...")
    hybrid_model = HybridGPTOSS20B(
        base_model=base_model,
        freeze_base=True,
        adapter_dim=128,
        num_timesteps=1000,
        vocab_size=199036,
    )

    print("\n3. Testing autoregressive mode...")
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, 199036, (batch_size, seq_len))

    hybrid_model.set_mode("autoregressive")
    logits_ar = hybrid_model(input_ids, mode="autoregressive")
    assert logits_ar.shape == (batch_size, seq_len, 199036)
    print(f"   ✓ Autoregressive: {input_ids.shape} -> {logits_ar.shape}")

    print("\n4. Testing diffusion mode...")
    timesteps = torch.randint(0, 1000, (batch_size,))
    edit_mask = torch.rand(batch_size, seq_len) < 0.3  # Edit 30% of tokens

    hybrid_model.set_mode("diffusion")
    logits_diff = hybrid_model(input_ids, timestep=timesteps, mode="diffusion", edit_mask=edit_mask)
    assert logits_diff.shape == (batch_size, seq_len, 199036)
    print(f"   ✓ Diffusion: {input_ids.shape} -> {logits_diff.shape}")

    print("\n5. Testing generation...")
    prompt = torch.randint(0, 199036, (1, 10))
    generated = hybrid_model.generate_autoregressive(prompt, max_new_tokens=20)
    assert generated.shape[1] == 30  # 10 prompt + 20 new
    print(f"   ✓ Generation: {prompt.shape} -> {generated.shape}")

    print("\n6. Testing editing...")
    text = torch.randint(0, 199036, (1, 50))
    edit_mask = torch.zeros(1, 50, dtype=torch.bool)
    edit_mask[0, 10:20] = True  # Edit positions 10-19

    edited = hybrid_model.edit_text(text, edit_mask, num_steps=5)
    assert edited.shape == text.shape
    # Check that non-edited positions are unchanged
    assert (edited[0, :10] == text[0, :10]).all()
    assert (edited[0, 20:] == text[0, 20:]).all()
    print(f"   ✓ Editing: {text.shape} -> {edited.shape} (edited positions 10-19)")

    print("\n✅ All tests passed!")
