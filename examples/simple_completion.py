#!/usr/bin/env python3
"""
Simple completion example using OSS_SRV Python SDK

Usage:
    python simple_completion.py --model path/to/model.engine
"""

import argparse
from oss_srv import InferenceClient, InferenceConfig


def main():
    parser = argparse.ArgumentParser(description='Simple OSS_SRV completion example')
    parser.add_argument('--model', type=str, required=True, help='Path to model.engine')
    parser.add_argument('--prompt', type=str, default='Once upon a time', help='Input prompt')
    parser.add_argument('--max-tokens', type=int, default=100, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    args = parser.parse_args()

    # Create client
    print(f"Loading model from: {args.model}")
    client = InferenceClient(model_path=args.model)

    # Generate completion
    print(f"\nPrompt: {args.prompt}")
    print("-" * 80)

    result = client.complete(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )

    if result.success:
        print(f"\nGenerated Text:\n{result.text}\n")
        print("-" * 80)
        print(f"Metrics:")
        print(f"  Inference Time: {result.inference_time_ms:.2f} ms")
        print(f"  Total Time: {result.total_time_ms:.2f} ms")
        print(f"  Throughput: {result.throughput_tokens_per_sec:.1f} tokens/sec")
    else:
        print(f"Error: {result.error}")


if __name__ == '__main__':
    main()
