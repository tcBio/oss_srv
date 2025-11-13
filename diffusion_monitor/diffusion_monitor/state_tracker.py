"""
State Tracker for Diffusion Models

Tracks intermediate states (activations, attention, gradients) during inference.
"""

from typing import Dict, List, Optional
import torch
import numpy as np


class StateTracker:
    """Tracks and stores intermediate states during diffusion inference"""

    def __init__(self, max_storage_mb: int = 1000):
        """
        Initialize state tracker

        Args:
            max_storage_mb: Maximum memory to use for storing states (MB)
        """
        self.max_storage_mb = max_storage_mb
        self.activations: Dict[str, List[torch.Tensor]] = {}
        self.attention: Dict[str, List[torch.Tensor]] = {}
        self.gradients: Dict[str, List[float]] = {}
        self.current_size_mb = 0.0

    def add_activation(self, layer_name: str, activation: torch.Tensor) -> None:
        """Store activation from a layer"""
        if not self._check_storage(activation):
            return

        if layer_name not in self.activations:
            self.activations[layer_name] = []

        self.activations[layer_name].append(activation)
        self.current_size_mb += self._tensor_size_mb(activation)

    def add_attention(self, layer_name: str, attention: torch.Tensor) -> None:
        """Store attention weights from a layer"""
        if not self._check_storage(attention):
            return

        if layer_name not in self.attention:
            self.attention[layer_name] = []

        self.attention[layer_name].append(attention)
        self.current_size_mb += self._tensor_size_mb(attention)

    def add_gradient(self, layer_name: str, grad_norm: float) -> None:
        """Store gradient norm (scalar, not full tensor)"""
        if layer_name not in self.gradients:
            self.gradients[layer_name] = []

        self.gradients[layer_name].append(grad_norm)

    def get_activations(self, layer_name: str) -> Optional[List[torch.Tensor]]:
        """Retrieve activations for a layer"""
        return self.activations.get(layer_name)

    def get_attention(self, layer_name: str) -> Optional[List[torch.Tensor]]:
        """Retrieve attention weights for a layer"""
        return self.attention.get(layer_name)

    def get_gradients(self, layer_name: str) -> Optional[List[float]]:
        """Retrieve gradient norms for a layer"""
        return self.gradients.get(layer_name)

    def get_all_layers(self) -> List[str]:
        """Get list of all tracked layers"""
        layers = set()
        layers.update(self.activations.keys())
        layers.update(self.attention.keys())
        layers.update(self.gradients.keys())
        return sorted(list(layers))

    def reset(self) -> None:
        """Clear all stored states"""
        self.activations.clear()
        self.attention.clear()
        self.gradients.clear()
        self.current_size_mb = 0.0

    def get_storage_info(self) -> Dict:
        """Get information about current storage usage"""
        return {
            "current_size_mb": self.current_size_mb,
            "max_size_mb": self.max_storage_mb,
            "utilization": self.current_size_mb / self.max_storage_mb,
            "num_activation_layers": len(self.activations),
            "num_attention_layers": len(self.attention),
            "num_gradient_layers": len(self.gradients),
            "total_layers": len(self.get_all_layers()),
        }

    def _check_storage(self, tensor: torch.Tensor) -> bool:
        """Check if we have room to store this tensor"""
        size_mb = self._tensor_size_mb(tensor)
        if self.current_size_mb + size_mb > self.max_storage_mb:
            print(
                f"Warning: Storage limit reached ({self.current_size_mb:.1f}/{self.max_storage_mb} MB). "
                f"Skipping storage."
            )
            return False
        return True

    @staticmethod
    def _tensor_size_mb(tensor: torch.Tensor) -> float:
        """Calculate tensor size in MB"""
        num_elements = tensor.numel()
        bytes_per_element = tensor.element_size()
        return (num_elements * bytes_per_element) / (1024 * 1024)

    def export_to_numpy(self) -> Dict:
        """Export all stored states as numpy arrays (for analysis)"""
        return {
            "activations": {
                layer: [t.numpy() for t in tensors]
                for layer, tensors in self.activations.items()
            },
            "attention": {
                layer: [t.numpy() for t in tensors]
                for layer, tensors in self.attention.items()
            },
            "gradients": self.gradients.copy(),
        }

    def compute_activation_statistics(self) -> Dict:
        """Compute statistics over stored activations"""
        stats = {}

        for layer_name, activations in self.activations.items():
            if not activations:
                continue

            # Compute stats across steps
            means = [act.mean().item() for act in activations]
            stds = [act.std().item() for act in activations]
            maxes = [act.max().item() for act in activations]
            mins = [act.min().item() for act in activations]

            stats[layer_name] = {
                "num_steps": len(activations),
                "mean_activation": {
                    "mean": np.mean(means),
                    "std": np.std(means),
                    "min": np.min(means),
                    "max": np.max(means),
                },
                "std_activation": {
                    "mean": np.mean(stds),
                    "std": np.std(stds),
                },
                "range": {
                    "min": np.min(mins),
                    "max": np.max(maxes),
                },
            }

        return stats

    def compute_attention_statistics(self) -> Dict:
        """Compute statistics over stored attention weights"""
        stats = {}

        for layer_name, attention_list in self.attention.items():
            if not attention_list:
                continue

            # Compute attention entropy and sparsity
            entropies = []
            sparsities = []

            for attn in attention_list:
                # attn: [heads, seq_len, seq_len]
                # Compute entropy per head
                attn_np = attn.numpy()
                for head in range(attn_np.shape[0]):
                    for pos in range(attn_np.shape[1]):
                        dist = attn_np[head, pos, :]
                        if dist.sum() > 0:
                            entropy = -np.sum(dist * np.log(dist + 1e-10))
                            entropies.append(entropy)

                            # Sparsity: fraction of weights < 0.01
                            sparsity = (dist < 0.01).sum() / len(dist)
                            sparsities.append(sparsity)

            stats[layer_name] = {
                "num_steps": len(attention_list),
                "attention_entropy": {
                    "mean": np.mean(entropies) if entropies else 0.0,
                    "std": np.std(entropies) if entropies else 0.0,
                },
                "attention_sparsity": {
                    "mean": np.mean(sparsities) if sparsities else 0.0,
                    "std": np.std(sparsities) if sparsities else 0.0,
                },
            }

        return stats
