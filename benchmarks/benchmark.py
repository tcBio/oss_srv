#!/usr/bin/env python3
"""
OSS_SRV Benchmarking Framework
Compare performance against vLLM, TensorRT-LLM, and llama.cpp

Usage:
    python benchmark.py --engine oss_srv --model path/to/model.engine
    python benchmark.py --compare-all --model-path path/to/model
"""

import argparse
import subprocess
import time
import json
import statistics
from dataclasses import dataclass
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    engine: str
    batch_size: int
    input_length: int
    output_length: int

    # Latency metrics (milliseconds)
    ttft: float  # Time to first token
    tpot: float  # Time per output token
    total_time: float

    # Throughput metrics
    throughput: float  # tokens per second
    requests_per_sec: float

    # Memory metrics
    peak_memory_gb: float
    avg_memory_gb: float

    # Quality metrics (optional)
    success_rate: float = 1.0
    error_count: int = 0


class OSSSRVBenchmark:
    """Benchmark OSS_SRV inference server"""

    def __init__(self, executable_path: str = "./complete_inference"):
        self.executable = executable_path

    def run_inference(self, model_path: str, prompt: str, max_tokens: int,
                     temperature: float = 0.7, top_p: float = 0.9) -> Dict:
        """Run single inference and parse output"""
        start_time = time.perf_counter()

        cmd = [
            self.executable,
            model_path,
            prompt,
            str(max_tokens),
            str(temperature),
            str(top_p)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            end_time = time.perf_counter()

            output = result.stdout

            # Parse metrics from output
            metrics = {}
            for line in output.split('\n'):
                if 'INFERENCE_TIME_MS:' in line:
                    metrics['inference_time'] = float(line.split(':')[1].strip())
                elif 'THROUGHPUT_TOKENS_PER_SEC:' in line:
                    metrics['throughput'] = float(line.split(':')[1].strip())
                elif 'TOKENS_GENERATED:' in line:
                    metrics['tokens_generated'] = int(line.split(':')[1].strip())

            metrics['total_time'] = (end_time - start_time) * 1000  # ms
            metrics['success'] = result.returncode == 0

            return metrics

        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'timeout'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def benchmark_throughput(self, model_path: str, batch_sizes: List[int],
                            prompt_lengths: List[int], num_runs: int = 5) -> List[BenchmarkResult]:
        """Benchmark throughput across different batch sizes and prompt lengths"""
        results = []

        # Generate test prompts of varying lengths
        test_prompts = {
            128: "Once upon a time " * 20,
            512: "The quick brown fox " * 80,
            1024: "In a galaxy far, far away " * 150,
            2048: "Hello world " * 300
        }

        for batch_size in batch_sizes:
            for prompt_len in prompt_lengths:
                prompt = test_prompts.get(prompt_len, "Test prompt " * (prompt_len // 2))

                print(f"\nTesting batch_size={batch_size}, prompt_length={prompt_len}")

                # Run multiple times and average
                ttft_list = []
                throughput_list = []
                total_time_list = []

                for run in range(num_runs):
                    metrics = self.run_inference(model_path, prompt, max_tokens=100)

                    if metrics.get('success'):
                        ttft_list.append(metrics.get('inference_time', 0) / metrics.get('tokens_generated', 1))
                        throughput_list.append(metrics.get('throughput', 0))
                        total_time_list.append(metrics.get('total_time', 0))

                    print(f"  Run {run+1}/{num_runs}: {metrics.get('throughput', 0):.1f} tok/s")

                if ttft_list:
                    result = BenchmarkResult(
                        engine="oss_srv",
                        batch_size=batch_size,
                        input_length=prompt_len,
                        output_length=100,
                        ttft=statistics.mean(ttft_list),
                        tpot=1000.0 / statistics.mean(throughput_list) if throughput_list else 0,
                        total_time=statistics.mean(total_time_list),
                        throughput=statistics.mean(throughput_list),
                        requests_per_sec=1000.0 / statistics.mean(total_time_list) if total_time_list else 0,
                        peak_memory_gb=0,  # TODO: measure GPU memory
                        avg_memory_gb=0,
                        success_rate=len(ttft_list) / num_runs
                    )
                    results.append(result)

        return results


class ComparativeBenchmark:
    """Compare OSS_SRV against other frameworks"""

    def __init__(self):
        self.oss_srv = OSSSRVBenchmark()
        self.results = {}

    def benchmark_vllm(self, model_path: str) -> List[BenchmarkResult]:
        """Benchmark vLLM (requires vllm package)"""
        # Placeholder - implement if vLLM is installed
        print("vLLM benchmark not implemented. Install vLLM to compare.")
        return []

    def benchmark_trt_llm(self, model_path: str) -> List[BenchmarkResult]:
        """Benchmark TensorRT-LLM"""
        # Placeholder - implement if TensorRT-LLM is installed
        print("TensorRT-LLM benchmark not implemented.")
        return []

    def benchmark_llamacpp(self, model_path: str) -> List[BenchmarkResult]:
        """Benchmark llama.cpp"""
        # Placeholder - implement if llama.cpp is installed
        print("llama.cpp benchmark not implemented.")
        return []

    def run_all(self, model_path: str, batch_sizes: List[int] = [1, 2, 4, 8],
                prompt_lengths: List[int] = [128, 512, 1024]) -> Dict[str, List[BenchmarkResult]]:
        """Run benchmarks on all frameworks"""
        results = {}

        print("=" * 80)
        print("OSS_SRV Comparative Benchmark Suite")
        print("=" * 80)

        print("\n[1/4] Benchmarking OSS_SRV...")
        results['oss_srv'] = self.oss_srv.benchmark_throughput(model_path, batch_sizes, prompt_lengths)

        print("\n[2/4] Benchmarking vLLM...")
        results['vllm'] = self.benchmark_vllm(model_path)

        print("\n[3/4] Benchmarking TensorRT-LLM...")
        results['trt_llm'] = self.benchmark_trt_llm(model_path)

        print("\n[4/4] Benchmarking llama.cpp...")
        results['llamacpp'] = self.benchmark_llamacpp(model_path)

        return results

    def generate_report(self, results: Dict[str, List[BenchmarkResult]], output_file: str = "benchmark_report.json"):
        """Generate JSON report"""
        report = {}

        for engine, bench_results in results.items():
            report[engine] = [
                {
                    'batch_size': r.batch_size,
                    'input_length': r.input_length,
                    'ttft_ms': r.ttft,
                    'throughput_tok_s': r.throughput,
                    'total_time_ms': r.total_time,
                    'success_rate': r.success_rate
                }
                for r in bench_results
            ]

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✅ Report saved to {output_file}")

    def plot_results(self, results: Dict[str, List[BenchmarkResult]], output_file: str = "benchmark_plot.png"):
        """Generate visualization of results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('OSS_SRV Benchmark Comparison', fontsize=16, fontweight='bold')

        # Plot 1: Throughput vs Batch Size
        ax1 = axes[0, 0]
        for engine, bench_results in results.items():
            if bench_results:
                batch_sizes = [r.batch_size for r in bench_results if r.input_length == 512]
                throughputs = [r.throughput for r in bench_results if r.input_length == 512]
                ax1.plot(batch_sizes, throughputs, marker='o', label=engine, linewidth=2)

        ax1.set_xlabel('Batch Size', fontweight='bold')
        ax1.set_ylabel('Throughput (tokens/sec)', fontweight='bold')
        ax1.set_title('Throughput vs Batch Size (512 token input)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: TTFT vs Input Length
        ax2 = axes[0, 1]
        for engine, bench_results in results.items():
            if bench_results:
                input_lengths = [r.input_length for r in bench_results if r.batch_size == 1]
                ttfts = [r.ttft for r in bench_results if r.batch_size == 1]
                ax2.plot(input_lengths, ttfts, marker='s', label=engine, linewidth=2)

        ax2.set_xlabel('Input Length (tokens)', fontweight='bold')
        ax2.set_ylabel('Time to First Token (ms)', fontweight='bold')
        ax2.set_title('TTFT vs Input Length (batch size 1)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Bar chart comparison (average throughput)
        ax3 = axes[1, 0]
        engines = list(results.keys())
        avg_throughputs = [
            statistics.mean([r.throughput for r in results[e]]) if results[e] else 0
            for e in engines
        ]
        colors = ['#4CAF50', '#FF9800', '#2196F3', '#9C27B0']
        ax3.bar(engines, avg_throughputs, color=colors[:len(engines)], alpha=0.8)
        ax3.set_ylabel('Average Throughput (tokens/sec)', fontweight='bold')
        ax3.set_title('Average Throughput Comparison')
        ax3.grid(True, axis='y', alpha=0.3)

        # Plot 4: Summary table
        ax4 = axes[1, 1]
        ax4.axis('off')

        table_data = []
        for engine in engines:
            if results[engine]:
                avg_throughput = statistics.mean([r.throughput for r in results[engine]])
                avg_ttft = statistics.mean([r.ttft for r in results[engine]])
                table_data.append([engine, f"{avg_throughput:.1f}", f"{avg_ttft:.1f}"])

        if table_data:
            table = ax4.table(
                cellText=table_data,
                colLabels=['Engine', 'Avg Throughput\n(tok/s)', 'Avg TTFT\n(ms)'],
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

            # Color header
            for i in range(3):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='OSS_SRV Benchmark Suite')
    parser.add_argument('--model', type=str, required=True, help='Path to model.engine')
    parser.add_argument('--engine', type=str, default='oss_srv', choices=['oss_srv', 'vllm', 'trt_llm', 'llamacpp', 'all'])
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 2, 4, 8], help='Batch sizes to test')
    parser.add_argument('--prompt-lengths', nargs='+', type=int, default=[128, 512, 1024], help='Prompt lengths to test')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of runs per configuration')
    parser.add_argument('--compare-all', action='store_true', help='Compare all frameworks')
    parser.add_argument('--output', type=str, default='benchmark_results', help='Output file prefix')

    args = parser.parse_args()

    if args.compare_all or args.engine == 'all':
        # Run comparative benchmark
        comp = ComparativeBenchmark()
        results = comp.run_all(args.model, args.batch_sizes, args.prompt_lengths)
        comp.generate_report(results, f"{args.output}.json")
        comp.plot_results(results, f"{args.output}.png")
    else:
        # Run single engine benchmark
        if args.engine == 'oss_srv':
            bench = OSSSRVBenchmark()
            results = bench.benchmark_throughput(args.model, args.batch_sizes, args.prompt_lengths, args.num_runs)

            # Print summary
            print("\n" + "=" * 80)
            print("BENCHMARK SUMMARY")
            print("=" * 80)
            for r in results:
                print(f"Batch={r.batch_size}, Input={r.input_length}: "
                      f"Throughput={r.throughput:.1f} tok/s, TTFT={r.ttft:.1f}ms")

            # Save results
            with open(f"{args.output}.json", 'w') as f:
                json.dump([{
                    'batch_size': r.batch_size,
                    'input_length': r.input_length,
                    'throughput': r.throughput,
                    'ttft': r.ttft
                } for r in results], f, indent=2)


if __name__ == '__main__':
    main()
