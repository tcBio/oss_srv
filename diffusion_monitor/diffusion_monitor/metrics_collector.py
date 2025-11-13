"""
Metrics Collector for Diffusion Models

Extracts and computes metrics from diffusion model tensors.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import torch
import numpy as np
from scipy.stats import entropy


@dataclass
class DiffusionMetrics:
    """Container for per-step diffusion metrics"""

    # Step info
    step_number: int
    step_latency_ms: float
    batch_size: int

    # Token predictions
    predicted_tokens: List[int]
    token_confidences: List[float]
    avg_confidence: float
    min_confidence: float
    max_confidence: float

    # Mask info
    mask_coverage: float  # Fraction of tokens masked
    mask_positions: List[int]

    # Distribution metrics
    logit_variance: float
    logit_entropy: float
    attention_entropy: Optional[float] = None

    # Convergence indicators
    token_flip_count: Optional[int] = None  # vs. previous step
    token_flip_rate: Optional[float] = None
    confidence_delta: Optional[float] = None  # Change from previous step

    # Quality estimates
    perplexity: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    def to_flat_dict(self) -> Dict:
        """Convert to flat dictionary for metrics export"""
        return {
            "step_number": self.step_number,
            "step_latency_ms": self.step_latency_ms,
            "batch_size": self.batch_size,
            "avg_confidence": self.avg_confidence,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
            "mask_coverage": self.mask_coverage,
            "logit_variance": self.logit_variance,
            "logit_entropy": self.logit_entropy,
            "attention_entropy": self.attention_entropy or 0.0,
            "token_flip_count": self.token_flip_count or 0,
            "token_flip_rate": self.token_flip_rate or 0.0,
            "confidence_delta": self.confidence_delta or 0.0,
            "perplexity": self.perplexity or 0.0,
        }


class MetricsCollector:
    """Extracts metrics from model tensors during inference"""

    def __init__(self, compute_attention_metrics: bool = True):
        self.compute_attention_metrics = compute_attention_metrics
        self.previous_tokens: Optional[torch.Tensor] = None
        self.previous_confidence: Optional[float] = None

    def collect_step_metrics(
        self,
        step: int,
        logits: torch.Tensor,
        latency_ms: float,
        mask: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> DiffusionMetrics:
        """
        Collect metrics for a single diffusion step

        Args:
            step: Current diffusion step number
            logits: Model output logits [batch, seq_len, vocab_size]
            latency_ms: Step latency in milliseconds
            mask: Optional mask tensor indicating masked positions
            attention_weights: Optional attention weights [batch, heads, seq_len, seq_len]

        Returns:
            DiffusionMetrics object with computed metrics
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Token predictions
        predicted_tokens = logits.argmax(dim=-1)  # [batch, seq_len]
        probs = torch.softmax(logits, dim=-1)
        confidences = probs.max(dim=-1).values  # [batch, seq_len]

        # Convert to lists (use first batch item for simplicity)
        tokens_list = predicted_tokens[0].cpu().tolist()
        conf_list = confidences[0].cpu().tolist()

        # Aggregate confidence stats
        avg_conf = confidences.mean().item()
        min_conf = confidences.min().item()
        max_conf = confidences.max().item()

        # Mask metrics
        if mask is not None:
            mask_coverage = mask.float().mean().item()
            mask_positions = mask[0].nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
        else:
            mask_coverage = 0.0
            mask_positions = []

        # Distribution metrics
        logit_variance = logits.var().item()
        logit_entropy = self._compute_entropy(probs[0])

        # Attention metrics
        attention_entropy = None
        if attention_weights is not None and self.compute_attention_metrics:
            attention_entropy = self._compute_attention_entropy(attention_weights[0])

        # Convergence metrics (compare with previous step)
        token_flip_count = None
        token_flip_rate = None
        confidence_delta = None

        if self.previous_tokens is not None:
            flips = (predicted_tokens[0] != self.previous_tokens[0]).sum().item()
            token_flip_count = flips
            token_flip_rate = flips / seq_len

        if self.previous_confidence is not None:
            confidence_delta = avg_conf - self.previous_confidence

        # Perplexity estimate
        perplexity = self._compute_perplexity(probs[0])

        # Update state for next iteration
        self.previous_tokens = predicted_tokens.clone()
        self.previous_confidence = avg_conf

        return DiffusionMetrics(
            step_number=step,
            step_latency_ms=latency_ms,
            batch_size=batch_size,
            predicted_tokens=tokens_list,
            token_confidences=conf_list,
            avg_confidence=avg_conf,
            min_confidence=min_conf,
            max_confidence=max_conf,
            mask_coverage=mask_coverage,
            mask_positions=mask_positions,
            logit_variance=logit_variance,
            logit_entropy=logit_entropy,
            attention_entropy=attention_entropy,
            token_flip_count=token_flip_count,
            token_flip_rate=token_flip_rate,
            confidence_delta=confidence_delta,
            perplexity=perplexity,
        )

    def reset(self):
        """Reset state for new inference"""
        self.previous_tokens = None
        self.previous_confidence = None

    @staticmethod
    def _compute_entropy(probs: torch.Tensor) -> float:
        """Compute Shannon entropy of probability distribution"""
        # probs: [seq_len, vocab_size]
        probs_np = probs.detach().cpu().numpy()
        # Average entropy across positions
        entropies = [entropy(probs_np[i]) for i in range(len(probs_np))]
        return float(np.mean(entropies))

    @staticmethod
    def _compute_attention_entropy(attention: torch.Tensor) -> float:
        """Compute average entropy of attention distributions"""
        # attention: [heads, seq_len, seq_len]
        attention_np = attention.detach().cpu().numpy()
        entropies = []
        for head in range(attention_np.shape[0]):
            for pos in range(attention_np.shape[1]):
                att_dist = attention_np[head, pos, :]
                if att_dist.sum() > 0:  # Valid distribution
                    entropies.append(entropy(att_dist))
        return float(np.mean(entropies)) if entropies else 0.0

    @staticmethod
    def _compute_perplexity(probs: torch.Tensor) -> float:
        """Compute perplexity from probability distribution"""
        # probs: [seq_len, vocab_size]
        # Use max probability as proxy for model confidence
        max_probs = probs.max(dim=-1).values
        log_probs = torch.log(max_probs + 1e-10)
        avg_log_prob = log_probs.mean()
        perplexity = torch.exp(-avg_log_prob).item()
        return perplexity


class ConvergenceAnalyzer:
    """Analyzes convergence behavior from collected metrics"""

    @staticmethod
    def analyze_convergence(metrics_history: List[DiffusionMetrics]) -> Dict:
        """
        Analyze convergence from full step history

        Returns:
            Dictionary with convergence analysis:
            - converged: bool (whether model converged)
            - convergence_step: int (step where convergence occurred)
            - avg_flip_rate: float
            - final_confidence: float
            - efficiency: float (0-1, higher is better)
        """
        if not metrics_history:
            return {}

        total_steps = len(metrics_history)

        # Get flip rates (skip first step as it has no comparison)
        flip_rates = [
            m.token_flip_rate
            for m in metrics_history[1:]
            if m.token_flip_rate is not None
        ]

        # Detect convergence (flip rate < 5% for 2 consecutive steps)
        convergence_step = None
        convergence_threshold = 0.05

        for i in range(1, len(flip_rates) - 1):
            if (
                flip_rates[i] < convergence_threshold
                and flip_rates[i + 1] < convergence_threshold
            ):
                convergence_step = i + 2  # +2 because we skip first step and 0-indexed
                break

        # Compute efficiency: early convergence is better
        if convergence_step is not None:
            efficiency = 1.0 - (convergence_step / total_steps)
            converged = True
        else:
            efficiency = 0.0
            converged = False
            convergence_step = total_steps

        final_metrics = metrics_history[-1]

        return {
            "converged": converged,
            "convergence_step": convergence_step,
            "total_steps": total_steps,
            "avg_flip_rate": np.mean(flip_rates) if flip_rates else 0.0,
            "final_flip_rate": flip_rates[-1] if flip_rates else 0.0,
            "final_confidence": final_metrics.avg_confidence,
            "final_perplexity": final_metrics.perplexity or 0.0,
            "efficiency": efficiency,
            "step_latencies": [m.step_latency_ms for m in metrics_history],
            "avg_latency_ms": np.mean([m.step_latency_ms for m in metrics_history]),
            "total_time_ms": sum(m.step_latency_ms for m in metrics_history),
        }

    @staticmethod
    def detect_anomalies(metrics_history: List[DiffusionMetrics]) -> List[Dict]:
        """
        Detect anomalies in diffusion process

        Returns:
            List of anomaly detections with type and step
        """
        anomalies = []

        for i, metrics in enumerate(metrics_history):
            # Diverging confidence
            if metrics.confidence_delta is not None and metrics.confidence_delta < -0.1:
                anomalies.append(
                    {
                        "type": "confidence_drop",
                        "step": metrics.step_number,
                        "value": metrics.confidence_delta,
                        "severity": "warning",
                    }
                )

            # High flip rate late in process (should be converging)
            if (
                metrics.step_number > len(metrics_history) * 0.7
                and metrics.token_flip_rate is not None
                and metrics.token_flip_rate > 0.2
            ):
                anomalies.append(
                    {
                        "type": "slow_convergence",
                        "step": metrics.step_number,
                        "value": metrics.token_flip_rate,
                        "severity": "warning",
                    }
                )

            # Very high perplexity
            if metrics.perplexity is not None and metrics.perplexity > 100.0:
                anomalies.append(
                    {
                        "type": "high_perplexity",
                        "step": metrics.step_number,
                        "value": metrics.perplexity,
                        "severity": "error",
                    }
                )

            # Latency spike (>2x median)
            if i > 5:  # After warmup
                median_latency = np.median(
                    [m.step_latency_ms for m in metrics_history[:i]]
                )
                if metrics.step_latency_ms > 2 * median_latency:
                    anomalies.append(
                        {
                            "type": "latency_spike",
                            "step": metrics.step_number,
                            "value": metrics.step_latency_ms,
                            "baseline": median_latency,
                            "severity": "warning",
                        }
                    )

        return anomalies
