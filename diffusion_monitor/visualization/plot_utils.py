"""
Plotting utilities for diffusion monitoring visualizations
"""

from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")


def plot_convergence(
    metrics_history: List[Dict],
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot convergence metrics over diffusion steps

    Args:
        metrics_history: List of metric dictionaries per step
        save_path: Optional path to save figure
        show: Whether to display figure

    Returns:
        Matplotlib figure
    """
    if not metrics_history:
        raise ValueError("metrics_history is empty")

    steps = [m["step_number"] for m in metrics_history]
    confidences = [m["avg_confidence"] for m in metrics_history]
    flip_rates = [m.get("token_flip_rate", 0) for m in metrics_history]
    perplexities = [m.get("perplexity", 0) for m in metrics_history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Diffusion Model Convergence Analysis", fontsize=16, fontweight="bold")

    # Confidence evolution
    ax1 = axes[0, 0]
    ax1.plot(steps, confidences, marker="o", linewidth=2, markersize=4, color="blue")
    ax1.set_xlabel("Diffusion Step")
    ax1.set_ylabel("Average Confidence")
    ax1.set_title("Confidence Evolution")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # Token flip rate
    ax2 = axes[0, 1]
    ax2.plot(steps[1:], flip_rates[1:], marker="s", linewidth=2, markersize=4, color="red")
    ax2.axhline(y=0.05, color="green", linestyle="--", label="Convergence threshold")
    ax2.set_xlabel("Diffusion Step")
    ax2.set_ylabel("Token Flip Rate")
    ax2.set_title("Token Stability")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # Perplexity
    ax3 = axes[1, 0]
    ax3.plot(steps, perplexities, marker="^", linewidth=2, markersize=4, color="purple")
    ax3.set_xlabel("Diffusion Step")
    ax3.set_ylabel("Perplexity")
    ax3.set_title("Quality Metric (Perplexity)")
    ax3.grid(True, alpha=0.3)

    # Step latency
    ax4 = axes[1, 1]
    latencies = [m["step_latency_ms"] for m in metrics_history]
    ax4.bar(steps, latencies, alpha=0.7, color="orange")
    ax4.axhline(
        y=np.mean(latencies),
        color="blue",
        linestyle="--",
        label=f"Mean: {np.mean(latencies):.2f}ms",
    )
    ax4.set_xlabel("Diffusion Step")
    ax4.set_ylabel("Latency (ms)")
    ax4.set_title("Step Latency")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_step_latencies(
    metrics_history: List[Dict],
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot detailed step latency analysis"""

    latencies = [m["step_latency_ms"] for m in metrics_history]
    steps = list(range(len(latencies)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Step Latency Analysis", fontsize=16, fontweight="bold")

    # Time series
    ax1 = axes[0]
    ax1.plot(steps, latencies, marker="o", linewidth=2, markersize=4)
    ax1.axhline(
        y=np.mean(latencies), color="red", linestyle="--", label=f"Mean: {np.mean(latencies):.2f}ms"
    )
    ax1.axhline(
        y=np.percentile(latencies, 95),
        color="orange",
        linestyle="--",
        label=f"P95: {np.percentile(latencies, 95):.2f}ms",
    )
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Latency per Step")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Distribution
    ax2 = axes[1]
    ax2.hist(latencies, bins=20, alpha=0.7, color="blue", edgecolor="black")
    ax2.axvline(x=np.mean(latencies), color="red", linestyle="--", linewidth=2, label="Mean")
    ax2.axvline(
        x=np.median(latencies), color="green", linestyle="--", linewidth=2, label="Median"
    )
    ax2.set_xlabel("Latency (ms)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Latency Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_confidence_evolution(
    metrics_history: List[Dict],
    show_tokens: bool = False,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot confidence evolution with optional per-token view

    Args:
        metrics_history: List of metric dictionaries
        show_tokens: If True, show per-token confidence (heatmap)
        save_path: Optional save path
        show: Whether to display

    Returns:
        Matplotlib figure
    """
    steps = [m["step_number"] for m in metrics_history]
    avg_confidences = [m["avg_confidence"] for m in metrics_history]
    min_confidences = [m["min_confidence"] for m in metrics_history]
    max_confidences = [m["max_confidence"] for m in metrics_history]

    if show_tokens:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(12, 5))
        axes = [axes]

    fig.suptitle("Confidence Evolution", fontsize=16, fontweight="bold")

    # Average confidence with min/max range
    ax = axes[0]
    ax.plot(steps, avg_confidences, marker="o", linewidth=2, label="Average", color="blue")
    ax.fill_between(
        steps,
        min_confidences,
        max_confidences,
        alpha=0.3,
        label="Min-Max Range",
        color="blue",
    )
    ax.set_xlabel("Diffusion Step")
    ax.set_ylabel("Confidence")
    ax.set_title("Average Confidence with Range")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Per-token heatmap (if requested)
    if show_tokens and len(axes) > 1:
        ax = axes[1]

        # Extract per-token confidences (limit to first 50 tokens for visibility)
        token_confidences = []
        for m in metrics_history:
            confs = m.get("token_confidences", [])[:50]
            if confs:
                token_confidences.append(confs)

        if token_confidences:
            # Convert to array
            conf_array = np.array(token_confidences)

            # Plot heatmap
            im = ax.imshow(
                conf_array.T,
                aspect="auto",
                cmap="viridis",
                interpolation="nearest",
                vmin=0,
                vmax=1,
            )
            ax.set_xlabel("Diffusion Step")
            ax.set_ylabel("Token Position")
            ax.set_title("Per-Token Confidence Heatmap")
            plt.colorbar(im, ax=ax, label="Confidence")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_denoising_animation(
    metrics_history: List[Dict],
    output_path: str = "denoising.mp4",
    fps: int = 2,
    tokenizer = None,
) -> str:
    """
    Create animation of denoising process

    Args:
        metrics_history: List of metric dictionaries
        output_path: Output video path
        fps: Frames per second
        tokenizer: Optional tokenizer to decode tokens

    Returns:
        Path to saved animation
    """
    print(f"Creating denoising animation with {len(metrics_history)} steps...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Diffusion Denoising Process", fontsize=16, fontweight="bold")

    def update(frame):
        """Update function for animation"""
        for ax in axes.flat:
            ax.clear()

        metrics = metrics_history[frame]
        step = metrics["step_number"]

        # Token predictions (as text or IDs)
        ax1 = axes[0, 0]
        tokens = metrics.get("predicted_tokens", [])[:20]  # First 20 tokens
        if tokenizer:
            # Decode if tokenizer provided
            try:
                text = tokenizer.decode(tokens)
                ax1.text(0.5, 0.5, text, ha="center", va="center", wrap=True, fontsize=10)
            except:
                ax1.text(0.5, 0.5, str(tokens), ha="center", va="center", fontsize=8)
        else:
            ax1.text(0.5, 0.5, str(tokens), ha="center", va="center", fontsize=8)
        ax1.set_title(f"Predicted Tokens (Step {step})")
        ax1.axis("off")

        # Confidence
        ax2 = axes[0, 1]
        confidences = [metrics_history[i]["avg_confidence"] for i in range(frame + 1)]
        steps_so_far = list(range(frame + 1))
        ax2.plot(steps_so_far, confidences, marker="o", linewidth=2, color="blue")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Confidence")
        ax2.set_title("Confidence Evolution")
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)

        # Token flips
        ax3 = axes[1, 0]
        flip_rates = [
            metrics_history[i].get("token_flip_rate", 0)
            for i in range(1, frame + 1)
        ]
        if flip_rates:
            steps_flip = list(range(1, frame + 1))
            ax3.plot(steps_flip, flip_rates, marker="s", linewidth=2, color="red")
            ax3.axhline(y=0.05, color="green", linestyle="--", alpha=0.5)
            ax3.set_xlabel("Step")
            ax3.set_ylabel("Token Flip Rate")
            ax3.set_title("Token Stability")
            ax3.set_ylim([0, 1])
            ax3.grid(True, alpha=0.3)

        # Metrics table
        ax4 = axes[1, 1]
        ax4.axis("off")
        table_data = [
            ["Step", str(step)],
            ["Confidence", f"{metrics['avg_confidence']:.3f}"],
            ["Latency", f"{metrics['step_latency_ms']:.2f}ms"],
            ["Entropy", f"{metrics['logit_entropy']:.2f}"],
            ["Token Flips", str(metrics.get("token_flip_count", "N/A"))],
        ]
        table = ax4.table(
            cellText=table_data,
            loc="center",
            cellLoc="left",
            colWidths=[0.4, 0.6],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        ax4.set_title("Current Metrics")

        return axes.flat

    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(metrics_history), interval=1000 / fps, blit=False
    )

    # Save
    try:
        anim.save(output_path, writer="ffmpeg", fps=fps, dpi=100)
        print(f"✓ Animation saved to: {output_path}")
    except Exception as e:
        print(f"✗ Failed to save animation: {e}")
        print("  Install ffmpeg: sudo apt install ffmpeg")
        # Try saving as GIF instead
        gif_path = output_path.replace(".mp4", ".gif")
        try:
            anim.save(gif_path, writer="pillow", fps=fps)
            print(f"✓ Animation saved as GIF: {gif_path}")
            output_path = gif_path
        except Exception as e2:
            print(f"✗ Failed to save as GIF: {e2}")

    plt.close(fig)
    return output_path


# Utility function for quick visualization
def visualize_inference(metrics_history: List[Dict], output_dir: str = "."):
    """Generate all visualizations for an inference run"""
    import os

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating visualizations in {output_dir}/...")

    # Convergence plot
    plot_convergence(
        metrics_history,
        save_path=os.path.join(output_dir, "convergence.png"),
        show=False,
    )

    # Latency analysis
    plot_step_latencies(
        metrics_history,
        save_path=os.path.join(output_dir, "latencies.png"),
        show=False,
    )

    # Confidence evolution
    plot_confidence_evolution(
        metrics_history,
        show_tokens=True,
        save_path=os.path.join(output_dir, "confidence.png"),
        show=False,
    )

    print(f"✓ Visualizations saved to {output_dir}/")
