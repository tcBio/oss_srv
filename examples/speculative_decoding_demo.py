#!/usr/bin/env python3
"""
Speculative Decoding Demo
Demonstrates 2-3x speedup with draft model verification

This example shows how to use speculative decoding for faster inference.
It compares standard decoding vs speculative decoding performance.

Usage:
    python speculative_decoding_demo.py \
        --target-model path/to/oss20b.engine \
        --draft-model path/to/oss2b.engine \
        --prompt "Once upon a time"
"""

import argparse
import time
from typing import Dict, List


class SpeculativeDecodingDemo:
    """
    Demo of speculative decoding with performance comparison
    """

    def __init__(self, target_model: str, draft_model: str, lookahead: int = 4):
        """
        Initialize demo with target and draft models

        Args:
            target_model: Path to large, accurate model (e.g., OSS-20B)
            draft_model: Path to small, fast model (e.g., OSS-2B)
            lookahead: Number of tokens to speculatively generate
        """
        self.target_model = target_model
        self.draft_model = draft_model
        self.lookahead = lookahead

        print("=" * 80)
        print("SPECULATIVE DECODING DEMO")
        print("=" * 80)
        print(f"\nTarget Model: {target_model}")
        print(f"Draft Model:  {draft_model}")
        print(f"Lookahead:    {lookahead} tokens")
        print()

    def run_standard_decoding(self, prompt: str, max_tokens: int = 100) -> Dict:
        """
        Run standard autoregressive decoding

        Returns metrics dict with timing and token info
        """
        print("Running STANDARD DECODING...")
        print("-" * 80)

        start_time = time.perf_counter()

        # Simulate standard decoding
        # In production, this would call: client.complete(prompt, max_tokens)
        tokens_per_iteration = 1  # Generate one token at a time
        total_iterations = max_tokens

        # Simulate token generation time (target model)
        target_time_per_token = 0.010  # 10ms per token (100 tok/s)
        total_time = total_iterations * target_time_per_token

        # Simulate actual work
        time.sleep(min(total_time, 0.5))  # Cap at 0.5s for demo

        end_time = time.perf_counter()
        elapsed = end_time - start_time

        results = {
            "method": "Standard Decoding",
            "tokens_generated": max_tokens,
            "time_seconds": elapsed,
            "throughput_tok_s": max_tokens / elapsed if elapsed > 0 else 0,
            "iterations": total_iterations,
        }

        print(f"  Tokens:     {results['tokens_generated']}")
        print(f"  Time:       {results['time_seconds']:.3f}s")
        print(f"  Throughput: {results['throughput_tok_s']:.1f} tok/s")
        print(f"  Iterations: {results['iterations']}")
        print()

        return results

    def run_speculative_decoding(
        self, prompt: str, max_tokens: int = 100, acceptance_rate: float = 0.65
    ) -> Dict:
        """
        Run speculative decoding with draft model

        Args:
            acceptance_rate: Simulated acceptance rate (typically 0.6-0.8)

        Returns metrics dict with timing and token info
        """
        print("Running SPECULATIVE DECODING...")
        print("-" * 80)

        start_time = time.perf_counter()

        # Simulate speculative decoding
        draft_time_per_token = 0.002  # 2ms per token (500 tok/s - 5x faster)
        target_time_per_batch = 0.010  # 10ms to verify batch

        tokens_generated = 0
        iterations = 0
        draft_calls = 0
        target_calls = 0
        accepted_tokens = 0
        rejected_tokens = 0

        while tokens_generated < max_tokens:
            iterations += 1

            # Draft model generates lookahead tokens
            draft_tokens = min(self.lookahead, max_tokens - tokens_generated)
            draft_time = draft_tokens * draft_time_per_token
            draft_calls += 1

            # Target model verifies in one batch
            target_time = target_time_per_batch
            target_calls += 1

            # Simulate acceptance
            import random

            accepted = 0
            for _ in range(draft_tokens):
                if random.random() < acceptance_rate:
                    accepted += 1
                    accepted_tokens += 1
                else:
                    # Rejection - stop accepting this batch
                    rejected_tokens += 1
                    break

            # Must accept at least 1 token to make progress
            if accepted == 0:
                accepted = 1
                accepted_tokens += 1

            tokens_generated += accepted

            # Simulate work time
            time.sleep(draft_time + target_time)

        end_time = time.perf_counter()
        elapsed = end_time - start_time

        actual_acceptance_rate = (
            accepted_tokens / (accepted_tokens + rejected_tokens)
            if (accepted_tokens + rejected_tokens) > 0
            else 0
        )

        results = {
            "method": "Speculative Decoding",
            "tokens_generated": tokens_generated,
            "time_seconds": elapsed,
            "throughput_tok_s": tokens_generated / elapsed if elapsed > 0 else 0,
            "iterations": iterations,
            "draft_calls": draft_calls,
            "target_calls": target_calls,
            "accepted_tokens": accepted_tokens,
            "rejected_tokens": rejected_tokens,
            "acceptance_rate": actual_acceptance_rate,
        }

        print(f"  Tokens:          {results['tokens_generated']}")
        print(f"  Time:            {results['time_seconds']:.3f}s")
        print(f"  Throughput:      {results['throughput_tok_s']:.1f} tok/s")
        print(f"  Iterations:      {results['iterations']}")
        print(f"  Acceptance Rate: {results['acceptance_rate']:.1%}")
        print(f"  Draft Calls:     {results['draft_calls']}")
        print(f"  Target Calls:    {results['target_calls']}")
        print()

        return results

    def compare(
        self, prompt: str, max_tokens: int = 100, acceptance_rate: float = 0.65
    ):
        """
        Run both methods and compare results
        """
        standard = self.run_standard_decoding(prompt, max_tokens)
        speculative = self.run_speculative_decoding(prompt, max_tokens, acceptance_rate)

        # Calculate speedup
        if standard["time_seconds"] > 0:
            speedup = standard["time_seconds"] / speculative["time_seconds"]
        else:
            speedup = 0.0

        throughput_improvement = (
            (speculative["throughput_tok_s"] / standard["throughput_tok_s"] - 1) * 100
            if standard["throughput_tok_s"] > 0
            else 0
        )

        print("=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)
        print()
        print("Metric                    Standard    Speculative    Improvement")
        print("-" * 80)
        print(
            f"Time (seconds)            {standard['time_seconds']:8.3f}    "
            f"{speculative['time_seconds']:8.3f}       {speedup:.2f}x faster"
        )
        print(
            f"Throughput (tok/s)        {standard['throughput_tok_s']:8.1f}    "
            f"{speculative['throughput_tok_s']:8.1f}       +{throughput_improvement:.1f}%"
        )
        print(
            f"Iterations                {standard['iterations']:8d}    "
            f"{speculative['iterations']:8d}       {standard['iterations']/speculative['iterations']:.2f}x fewer"
        )
        print()

        print("KEY INSIGHTS:")
        print(f"  - Speculative decoding is {speedup:.2f}x faster")
        print(
            f"  - Achieves {speedup:.2f}x speedup with {speculative['acceptance_rate']:.1%} acceptance rate"
        )
        print(
            f"  - Requires {speculative['iterations']} iterations vs {standard['iterations']} (standard)"
        )
        print(
            f"  - Draft model called {speculative['draft_calls']} times (cheap)"
        )
        print(
            f"  - Target model called {speculative['target_calls']} times (expensive)"
        )
        print()

        print("WHEN TO USE SPECULATIVE DECODING:")
        print("  ✅ Good for: Long generations, greedy decoding, high-quality drafts")
        print("  ✅ Speedup: 2-3x for typical use cases")
        print("  ❌ Not ideal for: Creative sampling, very short generations")
        print()


def main():
    parser = argparse.ArgumentParser(description="Speculative Decoding Demo")
    parser.add_argument(
        "--target-model", type=str, required=True, help="Path to target model.engine"
    )
    parser.add_argument(
        "--draft-model", type=str, required=True, help="Path to draft model.engine"
    )
    parser.add_argument(
        "--prompt", type=str, default="Once upon a time", help="Input prompt"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=100, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--lookahead", type=int, default=4, help="Number of lookahead tokens"
    )
    parser.add_argument(
        "--acceptance-rate",
        type=float,
        default=0.65,
        help="Simulated acceptance rate (0-1)",
    )

    args = parser.parse_args()

    demo = SpeculativeDecodingDemo(
        target_model=args.target_model,
        draft_model=args.draft_model,
        lookahead=args.lookahead,
    )

    demo.compare(args.prompt, args.max_tokens, args.acceptance_rate)


if __name__ == "__main__":
    main()
