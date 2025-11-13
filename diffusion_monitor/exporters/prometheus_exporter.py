"""
Prometheus Exporter for Diffusion Metrics

Exposes metrics in Prometheus format via HTTP /metrics endpoint.
"""

import time
from typing import Dict, List, Optional
from prometheus_client import (
    start_http_server,
    Counter,
    Histogram,
    Gauge,
    Summary,
    Info,
)

from diffusion_monitor.metrics_collector import DiffusionMetrics


class PrometheusExporter:
    """
    Exports diffusion model metrics to Prometheus

    Starts an HTTP server that exposes metrics at /metrics endpoint.
    Metrics are updated in real-time as inference progresses.
    """

    def __init__(self, port: int = 8000, prefix: str = "diffusion"):
        """
        Initialize Prometheus exporter

        Args:
            port: Port for HTTP metrics server
            prefix: Metric name prefix
        """
        self.port = port
        self.prefix = prefix
        self.server_started = False

        # Initialize metrics
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize all Prometheus metrics"""

        # Step-level metrics
        self.step_latency = Histogram(
            f"{self.prefix}_step_latency_seconds",
            "Latency per diffusion step",
            ["model", "step"],
            buckets=(
                0.001,
                0.005,
                0.010,
                0.025,
                0.050,
                0.100,
                0.250,
                0.500,
                1.0,
                2.5,
                5.0,
            ),
        )

        self.step_confidence = Gauge(
            f"{self.prefix}_step_confidence",
            "Average confidence per step",
            ["model", "step"],
        )

        self.step_token_flips = Gauge(
            f"{self.prefix}_step_token_flips",
            "Number of token changes from previous step",
            ["model", "step"],
        )

        self.step_mask_coverage = Gauge(
            f"{self.prefix}_step_mask_coverage",
            "Fraction of tokens masked",
            ["model", "step"],
        )

        self.step_entropy = Gauge(
            f"{self.prefix}_step_entropy",
            "Logit entropy per step",
            ["model", "step"],
        )

        self.step_attention_entropy = Gauge(
            f"{self.prefix}_step_attention_entropy",
            "Attention entropy per step",
            ["model", "step"],
        )

        # Inference-level metrics
        self.inference_total = Counter(
            f"{self.prefix}_inference_total",
            "Total number of inference requests",
            ["model", "status"],
        )

        self.inference_duration = Histogram(
            f"{self.prefix}_inference_duration_seconds",
            "Total inference duration",
            ["model"],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
        )

        self.inference_steps = Histogram(
            f"{self.prefix}_inference_steps",
            "Number of diffusion steps per inference",
            ["model"],
            buckets=(5, 10, 20, 30, 50, 75, 100, 150, 200),
        )

        # Quality metrics
        self.final_perplexity = Gauge(
            f"{self.prefix}_final_perplexity",
            "Final output perplexity",
            ["model"],
        )

        self.final_confidence = Gauge(
            f"{self.prefix}_final_confidence",
            "Final average confidence",
            ["model"],
        )

        self.convergence_efficiency = Gauge(
            f"{self.prefix}_convergence_efficiency",
            "Convergence efficiency (0-1)",
            ["model"],
        )

        self.convergence_step = Gauge(
            f"{self.prefix}_convergence_step",
            "Step at which model converged",
            ["model"],
        )

        # Anomaly counters
        self.anomalies_total = Counter(
            f"{self.prefix}_anomalies_total",
            "Total anomalies detected",
            ["model", "type", "severity"],
        )

        # System info
        self.info = Info(
            f"{self.prefix}_monitor_info",
            "Information about the diffusion monitor",
        )
        self.info.info(
            {
                "version": "0.1.0",
                "exporter": "prometheus",
            }
        )

    def start(self) -> None:
        """Start Prometheus HTTP server"""
        if not self.server_started:
            start_http_server(self.port)
            self.server_started = True
            print(f"✓ Prometheus exporter running on http://localhost:{self.port}/metrics")

    def record_step(
        self, model: str, step: int, metrics: DiffusionMetrics
    ) -> None:
        """
        Record metrics for a single diffusion step

        Args:
            model: Model name
            step: Step number
            metrics: Collected metrics
        """
        # Start server if not started
        if not self.server_started:
            self.start()

        step_label = str(step)

        # Record step metrics
        self.step_latency.labels(model=model, step=step_label).observe(
            metrics.step_latency_ms / 1000.0  # Convert to seconds
        )

        self.step_confidence.labels(model=model, step=step_label).set(
            metrics.avg_confidence
        )

        if metrics.token_flip_count is not None:
            self.step_token_flips.labels(model=model, step=step_label).set(
                metrics.token_flip_count
            )

        self.step_mask_coverage.labels(model=model, step=step_label).set(
            metrics.mask_coverage
        )

        self.step_entropy.labels(model=model, step=step_label).set(
            metrics.logit_entropy
        )

        if metrics.attention_entropy is not None:
            self.step_attention_entropy.labels(model=model, step=step_label).set(
                metrics.attention_entropy
            )

    def record_inference(
        self, model: str, metrics: List[DiffusionMetrics], summary: Dict
    ) -> None:
        """
        Record full inference metrics

        Args:
            model: Model name
            metrics: List of per-step metrics
            summary: Inference summary dict
        """
        # Start server if not started
        if not self.server_started:
            self.start()

        # Record inference completion
        status = "success" if summary.get("convergence", {}).get("converged") else "incomplete"
        self.inference_total.labels(model=model, status=status).inc()

        # Record duration and steps
        total_time_s = summary.get("total_time_ms", 0) / 1000.0
        self.inference_duration.labels(model=model).observe(total_time_s)

        num_steps = len(metrics)
        self.inference_steps.labels(model=model).observe(num_steps)

        # Record convergence metrics
        convergence = summary.get("convergence", {})
        if convergence:
            if convergence.get("converged"):
                self.convergence_step.labels(model=model).set(
                    convergence.get("convergence_step", 0)
                )

            self.convergence_efficiency.labels(model=model).set(
                convergence.get("efficiency", 0.0)
            )

            # Final quality metrics
            final_perplexity = convergence.get("final_perplexity", 0.0)
            if final_perplexity > 0:
                self.final_perplexity.labels(model=model).set(final_perplexity)

            self.final_confidence.labels(model=model).set(
                convergence.get("final_confidence", 0.0)
            )

        # Record anomalies
        for anomaly in summary.get("anomalies", []):
            self.anomalies_total.labels(
                model=model,
                type=anomaly["type"],
                severity=anomaly["severity"],
            ).inc()

    def export_snapshot(self, model: str, metrics: List[DiffusionMetrics]) -> Dict:
        """
        Export current metrics as a dictionary (for manual inspection)

        Args:
            model: Model name
            metrics: List of collected metrics

        Returns:
            Dictionary with current metric values
        """
        if not metrics:
            return {}

        latest = metrics[-1]

        return {
            "model": model,
            "timestamp": time.time(),
            "current_step": latest.step_number,
            "total_steps": len(metrics),
            "latest_confidence": latest.avg_confidence,
            "latest_latency_ms": latest.step_latency_ms,
            "latest_flip_rate": latest.token_flip_rate or 0.0,
            "avg_latency_ms": sum(m.step_latency_ms for m in metrics) / len(metrics),
        }


def main():
    """Run standalone Prometheus exporter"""
    import argparse

    parser = argparse.ArgumentParser(description="Diffusion Model Prometheus Exporter")
    parser.add_argument(
        "--port", type=int, default=8000, help="HTTP server port (default: 8000)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="diffusion",
        help="Metric name prefix (default: diffusion)",
    )

    args = parser.parse_args()

    exporter = PrometheusExporter(port=args.port, prefix=args.prefix)
    exporter.start()

    print(f"Prometheus exporter running on http://localhost:{args.port}/metrics")
    print("Press Ctrl+C to exit")

    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
