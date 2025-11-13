"""
PyTorch Hooks for Diffusion Model Monitoring

Attaches hooks to diffusion models to capture intermediate states and metrics.
"""

import time
from typing import List, Dict, Optional, Callable, Any
import torch
import torch.nn as nn

from .metrics_collector import MetricsCollector, DiffusionMetrics, ConvergenceAnalyzer
from .state_tracker import StateTracker


class DiffusionMonitor:
    """
    Main monitoring class for diffusion models.

    Attaches hooks to model layers to capture metrics during inference.
    Supports multiple exporters (Prometheus, DataDog, etc.)
    """

    def __init__(
        self,
        model_name: str = "diffusion-model",
        exporters: Optional[List[Any]] = None,
        track_attention: bool = True,
        track_gradients: bool = False,
        store_activations: bool = False,
    ):
        """
        Initialize DiffusionMonitor

        Args:
            model_name: Name of the model (for metrics labeling)
            exporters: List of metric exporters (PrometheusExporter, etc.)
            track_attention: Whether to capture attention weights
            track_gradients: Whether to capture gradients (training mode)
            store_activations: Whether to store full activation tensors
        """
        self.model_name = model_name
        self.exporters = exporters or []
        self.track_attention = track_attention
        self.track_gradients = track_gradients
        self.store_activations = store_activations

        # Collectors
        self.metrics_collector = MetricsCollector(
            compute_attention_metrics=track_attention
        )
        self.state_tracker = StateTracker()

        # Storage
        self.metrics_history: List[DiffusionMetrics] = []
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        # State
        self.current_step = 0
        self.inference_start_time: Optional[float] = None
        self.step_start_time: Optional[float] = None

        # Model reference
        self.model: Optional[nn.Module] = None

    def attach(self, model: nn.Module) -> None:
        """
        Attach monitoring hooks to model

        Args:
            model: PyTorch model to monitor
        """
        self.model = model
        self.detach()  # Remove any existing hooks
        self.reset()

        # Find diffusion-specific layers
        # This is model-specific; adjust for your architecture
        for name, module in model.named_modules():
            # Hook denoising layers (common names in diffusion models)
            if any(
                keyword in name.lower()
                for keyword in [
                    "denoiser",
                    "diffusion",
                    "denoise_fn",
                    "unet",
                    "dit",  # Diffusion Transformer
                ]
            ):
                hook = module.register_forward_hook(self._create_forward_hook(name))
                self.hooks.append(hook)

            # Hook attention layers if requested
            if self.track_attention and "attn" in name.lower():
                hook = module.register_forward_hook(
                    self._create_attention_hook(name)
                )
                self.hooks.append(hook)

            # Hook gradient if in training mode
            if self.track_gradients and hasattr(module, "weight"):
                hook = module.register_full_backward_hook(
                    self._create_backward_hook(name)
                )
                self.hooks.append(hook)

        print(f"Attached {len(self.hooks)} hooks to {self.model_name}")

    def detach(self) -> None:
        """Remove all hooks from model"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def reset(self) -> None:
        """Reset state for new inference"""
        self.metrics_history.clear()
        self.metrics_collector.reset()
        self.state_tracker.reset()
        self.current_step = 0
        self.inference_start_time = None
        self.step_start_time = None

    def start_inference(self) -> None:
        """Mark start of inference"""
        self.inference_start_time = time.time()
        self.reset()

    def end_inference(self) -> Dict:
        """
        Mark end of inference and compute final metrics

        Returns:
            Dictionary with convergence analysis and summary
        """
        if self.inference_start_time is None:
            return {}

        total_time = (time.time() - self.inference_start_time) * 1000  # ms

        # Analyze convergence
        convergence_analysis = ConvergenceAnalyzer.analyze_convergence(
            self.metrics_history
        )

        # Detect anomalies
        anomalies = ConvergenceAnalyzer.detect_anomalies(self.metrics_history)

        summary = {
            "model": self.model_name,
            "total_steps": len(self.metrics_history),
            "total_time_ms": total_time,
            "convergence": convergence_analysis,
            "anomalies": anomalies,
        }

        # Export to all exporters
        for exporter in self.exporters:
            if hasattr(exporter, "record_inference"):
                exporter.record_inference(
                    model=self.model_name,
                    metrics=self.metrics_history,
                    summary=summary,
                )

        return summary

    def _create_forward_hook(self, layer_name: str) -> Callable:
        """Create forward hook for a layer"""

        def hook(module: nn.Module, input: Any, output: Any) -> None:
            # Check if this is a diffusion step output
            if not isinstance(output, torch.Tensor):
                return

            # Assume output is logits [batch, seq_len, vocab_size]
            if output.dim() != 3:
                return

            # Start step timer
            if self.step_start_time is None:
                self.step_start_time = time.time()

            # We'll collect metrics on the final denoising layer
            # For now, just track that we passed through this layer
            if self.store_activations:
                self.state_tracker.add_activation(layer_name, output.detach().cpu())

        return hook

    def _create_attention_hook(self, layer_name: str) -> Callable:
        """Create hook for attention layers"""

        def hook(module: nn.Module, input: Any, output: Any) -> None:
            if not isinstance(output, tuple):
                return

            # Extract attention weights (typically second element in output)
            if len(output) > 1 and isinstance(output[1], torch.Tensor):
                attn_weights = output[1]
                if self.store_activations:
                    self.state_tracker.add_attention(
                        layer_name, attn_weights.detach().cpu()
                    )

        return hook

    def _create_backward_hook(self, layer_name: str) -> Callable:
        """Create backward hook for gradients"""

        def hook(module: nn.Module, grad_input: Any, grad_output: Any) -> None:
            if grad_output and isinstance(grad_output[0], torch.Tensor):
                grad_norm = grad_output[0].norm().item()
                self.state_tracker.add_gradient(layer_name, grad_norm)

        return hook

    def record_step(
        self,
        step: int,
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> DiffusionMetrics:
        """
        Manually record a diffusion step (if hooks don't capture it)

        Args:
            step: Step number
            logits: Model output logits
            mask: Optional mask tensor
            attention_weights: Optional attention weights

        Returns:
            Collected metrics for this step
        """
        # Compute step latency
        if self.step_start_time is not None:
            latency_ms = (time.time() - self.step_start_time) * 1000
        else:
            latency_ms = 0.0

        # Collect metrics
        metrics = self.metrics_collector.collect_step_metrics(
            step=step,
            logits=logits,
            latency_ms=latency_ms,
            mask=mask,
            attention_weights=attention_weights,
        )

        # Store
        self.metrics_history.append(metrics)
        self.current_step = step

        # Reset step timer
        self.step_start_time = time.time()

        # Export to exporters
        for exporter in self.exporters:
            if hasattr(exporter, "record_step"):
                exporter.record_step(
                    model=self.model_name, step=step, metrics=metrics
                )

        return metrics

    def compute_convergence_metrics(self) -> Dict:
        """
        Compute convergence metrics from collected history

        Returns:
            Dictionary with convergence analysis
        """
        return ConvergenceAnalyzer.analyze_convergence(self.metrics_history)

    def get_denoising_trajectory(self) -> List[Dict]:
        """
        Get full denoising trajectory for visualization

        Returns:
            List of metric dictionaries per step
        """
        return [m.to_dict() for m in self.metrics_history]

    def get_step_metrics(self, step: int) -> Optional[DiffusionMetrics]:
        """Get metrics for a specific step"""
        if 0 <= step < len(self.metrics_history):
            return self.metrics_history[step]
        return None

    def add_exporter(self, exporter: Any) -> None:
        """Add a metrics exporter"""
        self.exporters.append(exporter)

    def __enter__(self):
        """Context manager entry"""
        self.start_inference()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.end_inference()
        return False


# Convenience function for quick monitoring
def monitor_diffusion_inference(
    model: nn.Module,
    model_name: str = "diffusion-model",
    exporters: Optional[List[Any]] = None,
) -> DiffusionMonitor:
    """
    Quick setup for monitoring diffusion inference

    Args:
        model: PyTorch diffusion model
        model_name: Model identifier
        exporters: List of metric exporters

    Returns:
        DiffusionMonitor instance (already attached)

    Example:
        >>> monitor = monitor_diffusion_inference(model, "llada-7b")
        >>> with monitor:
        ...     output = model.generate(prompt)
        >>> metrics = monitor.compute_convergence_metrics()
    """
    monitor = DiffusionMonitor(model_name=model_name, exporters=exporters)
    monitor.attach(model)
    return monitor
