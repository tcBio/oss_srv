"""
Basic Diffusion Model Monitoring Example

Demonstrates how to instrument a diffusion model with monitoring hooks.
"""

import time
import torch
import torch.nn as nn
from typing import List

# Import monitoring components
import sys
sys.path.insert(0, '..')

from diffusion_monitor import DiffusionMonitor
from exporters import PrometheusExporter


# Mock Diffusion Model (replace with actual LLaDA/Open-dLLM model)
class MockDiffusionModel(nn.Module):
    """
    Mock diffusion model for demonstration.
    Replace with actual model: LLaDA, Open-dLLM, etc.
    """

    def __init__(self, vocab_size=50000, hidden_size=768, seq_length=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        # Simple linear layer (replace with actual diffusion architecture)
        self.denoiser = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, step=0):
        """Forward pass for one diffusion step"""
        # Embed tokens
        hidden = self.embedding(input_ids)

        # Add noise based on step (mock diffusion process)
        noise_level = (step / 10.0) * 0.1
        hidden = hidden + torch.randn_like(hidden) * noise_level

        # Denoise
        logits = self.denoiser(hidden)

        return logits

    def generate(self, prompt_ids: torch.Tensor, num_steps: int = 20) -> torch.Tensor:
        """
        Generate text through iterative denoising

        Args:
            prompt_ids: Input token IDs [batch, seq_len]
            num_steps: Number of diffusion steps

        Returns:
            Generated token IDs
        """
        # Start with masked tokens
        batch_size, seq_len = prompt_ids.shape
        current_ids = prompt_ids.clone()

        # Mask some tokens (mock masking strategy)
        mask = torch.rand(batch_size, seq_len) > 0.5
        current_ids[mask] = 0  # Use 0 as mask token

        # Iteratively denoise
        for step in range(num_steps):
            # Get predictions
            with torch.no_grad():
                logits = self.forward(current_ids, step=step)

            # Sample tokens
            predicted_ids = logits.argmax(dim=-1)

            # Update masked positions (gradually unmask)
            unmask_prob = (step + 1) / num_steps
            unmask = (torch.rand(batch_size, seq_len) < unmask_prob) & mask
            current_ids[unmask] = predicted_ids[unmask]

        return current_ids


def run_monitored_inference(model: nn.Module, monitor: DiffusionMonitor):
    """Run inference with monitoring"""

    # Start monitoring
    monitor.start_inference()

    # Generate sample input
    prompt = "Once upon a time"
    vocab_size = 50000
    seq_length = 128

    # Mock tokenization (use actual tokenizer in production)
    prompt_ids = torch.randint(0, vocab_size, (1, seq_length))

    print(f"\n{'='*70}")
    print(f"Running monitored inference: {monitor.model_name}")
    print(f"{'='*70}\n")

    # Run diffusion generation with monitoring
    num_steps = 20
    current_ids = prompt_ids.clone()
    mask = torch.rand(1, seq_length) > 0.5
    current_ids[mask] = 0

    for step in range(num_steps):
        step_start = time.time()

        # Forward pass
        logits = model.forward(current_ids, step=step)

        # Calculate latency
        latency_ms = (time.time() - step_start) * 1000

        # Record step with monitor
        metrics = monitor.record_step(
            step=step,
            logits=logits,
            mask=mask if step == 0 else None,
        )

        # Print progress
        print(
            f"Step {step:2d}: "
            f"Latency={metrics.step_latency_ms:6.2f}ms, "
            f"Confidence={metrics.avg_confidence:.3f}, "
            f"Flips={metrics.token_flip_count or 0:3d}, "
            f"Entropy={metrics.logit_entropy:.2f}"
        )

        # Update tokens
        predicted_ids = logits.argmax(dim=-1)
        unmask_prob = (step + 1) / num_steps
        unmask = (torch.rand(1, seq_length) < unmask_prob) & mask
        current_ids[unmask] = predicted_ids[unmask]

        # Simulate variable latency
        time.sleep(0.01 + (step % 5) * 0.002)

    # End monitoring
    summary = monitor.end_inference()

    return current_ids, summary


def main():
    """Main example"""

    print("\n" + "="*70)
    print("Diffusion Model Monitoring - Basic Example")
    print("="*70)

    # 1. Initialize Prometheus exporter
    print("\n[1/5] Starting Prometheus exporter...")
    exporter = PrometheusExporter(port=8000)
    exporter.start()
    print("✓ Metrics available at: http://localhost:8000/metrics")

    # 2. Create mock diffusion model
    print("\n[2/5] Loading diffusion model...")
    model = MockDiffusionModel()
    model.eval()
    print("✓ Model loaded (mock)")

    # 3. Attach monitor
    print("\n[3/5] Attaching monitoring hooks...")
    monitor = DiffusionMonitor(
        model_name="mock-diffusion",
        exporters=[exporter],
        track_attention=True,
    )
    monitor.attach(model)
    print(f"✓ Monitor attached with {len(monitor.hooks)} hooks")

    # 4. Run inference
    print("\n[4/5] Running monitored inference...")
    output, summary = run_monitored_inference(model, monitor)

    # 5. Display results
    print(f"\n{'='*70}")
    print("Monitoring Results")
    print(f"{'='*70}\n")

    convergence = summary.get("convergence", {})
    print(f"Total Steps:          {convergence.get('total_steps', 0)}")
    print(f"Converged:            {convergence.get('converged', False)}")
    print(f"Convergence Step:     {convergence.get('convergence_step', 'N/A')}")
    print(f"Efficiency:           {convergence.get('efficiency', 0):.2%}")
    print(f"Avg Step Latency:     {convergence.get('avg_latency_ms', 0):.2f}ms")
    print(f"Total Time:           {convergence.get('total_time_ms', 0):.2f}ms")
    print(f"Final Confidence:     {convergence.get('final_confidence', 0):.3f}")
    print(f"Final Perplexity:     {convergence.get('final_perplexity', 0):.2f}")

    # Anomalies
    anomalies = summary.get("anomalies", [])
    if anomalies:
        print(f"\nAnomalies Detected:   {len(anomalies)}")
        for i, anomaly in enumerate(anomalies[:5], 1):
            print(f"  {i}. {anomaly['type']} at step {anomaly['step']} "
                  f"(severity: {anomaly['severity']})")
    else:
        print("\nAnomalies Detected:   None")

    print(f"\n{'='*70}")
    print("Next Steps:")
    print("  1. Open Prometheus: http://localhost:9090")
    print("  2. Query metrics: diffusion_step_latency_seconds")
    print("  3. Start Grafana: docker-compose up -d grafana")
    print("  4. Import dashboards from visualization/grafana_dashboards/")
    print(f"{'='*70}\n")

    # Keep exporter running
    print("Press Ctrl+C to exit...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        monitor.detach()


if __name__ == "__main__":
    main()
