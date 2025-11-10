#!/usr/bin/env python3
"""
Batch inference example using OSS_SRV Python SDK

Usage:
    python batch_inference.py --model path/to/model.engine
"""

import argparse
from oss_srv import BatchInferenceClient, BatchRequest


def main():
    parser = argparse.ArgumentParser(description='Batch OSS_SRV inference example')
    parser.add_argument('--model', type=str, required=True, help='Path to model.engine')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    args = parser.parse_args()

    # Create batch client
    print(f"Loading model from: {args.model}")
    client = BatchInferenceClient(model_path=args.model, batch_size=args.batch_size)

    # Prepare batch of requests
    prompts = [
        "Once upon a time in a galaxy far, far away",
        "The quick brown fox jumps over the lazy dog",
        "To be or not to be, that is the question",
        "In the beginning, there was nothing but darkness",
        "The future of artificial intelligence is",
        "Climate change is one of the biggest challenges",
        "Machine learning models have revolutionized",
        "The human brain is the most complex organ"
    ]

    requests = [
        BatchRequest(id=f"req_{i}", prompt=prompt, max_tokens=50)
        for i, prompt in enumerate(prompts)
    ]

    print(f"\nProcessing {len(requests)} requests in batches of {args.batch_size}...")
    print("=" * 80)

    # Process batch
    results = client.process_batch(requests)

    # Display results
    total_tokens = 0
    total_time = 0.0

    for batch_result in results:
        result = batch_result.result
        if result.success:
            print(f"\n[{batch_result.id}]")
            print(f"Prompt: {requests[int(batch_result.id.split('_')[1])].prompt}")
            print(f"Generated: {result.text}")
            print(f"Time: {result.total_time_ms:.2f}ms | Throughput: {result.throughput_tokens_per_sec:.1f} tok/s")
            print("-" * 80)

            total_time += result.total_time_ms
        else:
            print(f"[{batch_result.id}] Error: {result.error}")

    # Summary
    print(f"\n📊 SUMMARY")
    print(f"  Total requests: {len(results)}")
    print(f"  Successful: {sum(1 for r in results if r.result.success)}")
    print(f"  Total time: {total_time:.2f}ms")
    print(f"  Average time per request: {total_time/len(results):.2f}ms")


if __name__ == '__main__':
    main()
