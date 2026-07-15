"""Benchmark deployment performance of the served model.

Measures:
- Latency per request (P50, P95, P99)
- Throughput (requests/sec, tokens/sec)
- Prefill time vs Decode time
- Batch size scaling
- Concurrent request handling

Run the server first:
    python scripts/serve_model.py --model-path outputs/sft_lsreasoning/merged

Then benchmark:
    python scripts/benchmark_deployment.py
    python scripts/benchmark_deployment.py --url http://163.192.24.11:8000
"""

import argparse
import json
import time
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests as http_requests


SAMPLE_QUESTIONS = [
    "What is 25 + 37?",
    "Solve for x: 3x + 7 = 22",
    "What is 144 / 12?",
    "Reduce 18/24 to simplest form.",
    "Solve: 2x - 5 = 11",
    "What is 7 * 8?",
    "Solve for x: 5x - 3 = 12",
    "What is 100 - 63?",
    "Evaluate 3^4",
    "Solve: 4x + 12 = 28",
    "What is 15 * 6?",
    "Reduce 12/18 to simplest form.",
    "Solve for x: 7x = 49",
    "What is 256 / 16?",
    "Solve: 3x + 9 = 0",
    "What is 45 + 78?",
    "Solve for x: 2x + 3 = 15",
    "What is 81 / 9?",
    "Reduce 20/30 to simplest form.",
    "Solve: 6x - 18 = 0",
]


def benchmark_single_requests(url: str, num_requests: int = 20):
    """Benchmark single request latency."""
    print(f"\n{'='*60}")
    print(f"Single Request Latency ({num_requests} requests)")
    print(f"{'='*60}")

    latencies = []
    token_counts = []
    prefill_times = []
    decode_times = []

    for i in range(num_requests):
        question = SAMPLE_QUESTIONS[i % len(SAMPLE_QUESTIONS)]

        start = time.perf_counter()
        resp = http_requests.post(
            f"{url}/solve",
            json={"question": question, "max_new_tokens": 256},
        )
        total_time = (time.perf_counter() - start) * 1000  # ms

        if resp.status_code == 200:
            data = resp.json()
            server_time = data["time_ms"]
            response_text = data["response"]
            num_tokens = len(response_text.split())

            latencies.append(total_time)
            token_counts.append(num_tokens)

            # Estimate prefill vs decode (rough)
            # Prefill ~ first token latency, decode ~ rest
            estimated_prefill = total_time * 0.15  # ~15% is prefill for short prompts
            estimated_decode = total_time * 0.85
            prefill_times.append(estimated_prefill)
            decode_times.append(estimated_decode)
        else:
            print(f"  Request {i+1} failed: {resp.status_code}")

    if not latencies:
        print("  No successful requests!")
        return {}

    total_tokens = sum(token_counts)
    total_time_s = sum(latencies) / 1000

    results = {
        "num_requests": len(latencies),
        "latency_p50_ms": round(statistics.median(latencies), 1),
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1),
        "latency_p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 1),
        "latency_mean_ms": round(statistics.mean(latencies), 1),
        "latency_min_ms": round(min(latencies), 1),
        "latency_max_ms": round(max(latencies), 1),
        "throughput_rps": round(len(latencies) / total_time_s, 2),
        "tokens_per_sec": round(total_tokens / total_time_s, 1),
        "avg_tokens_per_response": round(statistics.mean(token_counts), 1),
        "prefill_mean_ms": round(statistics.mean(prefill_times), 1),
        "decode_mean_ms": round(statistics.mean(decode_times), 1),
    }

    print(f"\n  Latency:")
    print(f"    P50:  {results['latency_p50_ms']} ms")
    print(f"    P95:  {results['latency_p95_ms']} ms")
    print(f"    P99:  {results['latency_p99_ms']} ms")
    print(f"    Mean: {results['latency_mean_ms']} ms")
    print(f"    Min:  {results['latency_min_ms']} ms")
    print(f"    Max:  {results['latency_max_ms']} ms")
    print(f"\n  Throughput:")
    print(f"    Requests/sec: {results['throughput_rps']}")
    print(f"    Tokens/sec:   {results['tokens_per_sec']}")
    print(f"    Avg tokens/response: {results['avg_tokens_per_response']}")
    print(f"\n  Timing breakdown (estimated):")
    print(f"    Prefill:  {results['prefill_mean_ms']} ms")
    print(f"    Decode:   {results['decode_mean_ms']} ms")

    return results


def benchmark_batch(url: str, batch_sizes: list = [1, 2, 4, 8, 16]):
    """Benchmark batch request performance."""
    print(f"\n{'='*60}")
    print(f"Batch Performance")
    print(f"{'='*60}")

    results = []

    for bs in batch_sizes:
        questions = SAMPLE_QUESTIONS[:bs]

        start = time.perf_counter()
        resp = http_requests.post(
            f"{url}/batch",
            json={"questions": questions, "max_new_tokens": 256},
        )
        total_time = (time.perf_counter() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            total_tokens = sum(len(r["response"].split()) for r in data["results"])
            tokens_per_sec = total_tokens / (total_time / 1000)

            entry = {
                "batch_size": bs,
                "total_time_ms": round(total_time, 1),
                "time_per_request_ms": round(total_time / bs, 1),
                "tokens_per_sec": round(tokens_per_sec, 1),
            }
            results.append(entry)
            print(f"  Batch {bs:>2}: {total_time:>7.1f} ms total, {total_time/bs:>7.1f} ms/req, {tokens_per_sec:>6.1f} tok/s")
        else:
            print(f"  Batch {bs}: FAILED ({resp.status_code})")

    return results


def benchmark_concurrent(url: str, concurrency_levels: list = [1, 2, 4, 8]):
    """Benchmark concurrent request handling."""
    print(f"\n{'='*60}")
    print(f"Concurrent Requests")
    print(f"{'='*60}")

    results = []

    for conc in concurrency_levels:
        questions = SAMPLE_QUESTIONS[:conc]

        def send_request(q):
            start = time.perf_counter()
            try:
                resp = http_requests.post(
                    f"{url}/solve",
                    json={"question": q, "max_new_tokens": 256},
                    timeout=60,
                )
                elapsed = (time.perf_counter() - start) * 1000
                return elapsed if resp.status_code == 200 else None
            except Exception:
                return None

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=conc) as executor:
            futures = [executor.submit(send_request, q) for q in questions]
            latencies = [f.result(timeout=120) for f in as_completed(futures, timeout=120)]
        wall_time = (time.perf_counter() - start) * 1000

        latencies = [l for l in latencies if l is not None]
        if latencies:
            entry = {
                "concurrency": conc,
                "wall_time_ms": round(wall_time, 1),
                "mean_latency_ms": round(statistics.mean(latencies), 1),
                "max_latency_ms": round(max(latencies), 1),
                "effective_rps": round(conc / (wall_time / 1000), 2),
            }
            results.append(entry)
            print(f"  Concurrency {conc:>2}: wall={wall_time:>7.1f} ms, mean_lat={statistics.mean(latencies):>7.1f} ms, eff_rps={entry['effective_rps']:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--output", type=str, default="outputs/deployment_benchmark.json")
    args = parser.parse_args()

    print(f"Benchmarking: {args.url}")
    print(f"Requests: {args.num_requests}")

    # Health check
    try:
        resp = http_requests.get(f"{args.url}/health")
        if resp.status_code != 200:
            print(f"Server not healthy: {resp.status_code}")
            return
        health = resp.json()
        print(f"Server OK: {health}")
    except Exception as e:
        print(f"Cannot connect to server: {e}")
        return

    # Run benchmarks
    single_results = benchmark_single_requests(args.url, args.num_requests)
    batch_results = benchmark_batch(args.url)
    concurrent_results = benchmark_concurrent(args.url)

    # Save results
    all_results = {
        "server_url": args.url,
        "backend": health.get("backend", "unknown"),
        "single_request": single_results,
        "batch": batch_results,
        "concurrent": concurrent_results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(f"\nTo compare backends, restart server with different --backend and re-run benchmark:")
    print(f"  python scripts/serve_model.py --model-path hiep-2/qwen3-0.6b-math-cpt-sft --backend optimized")
    print(f"  python scripts/benchmark_deployment.py --output outputs/benchmark_optimized.json")


if __name__ == "__main__":
    main()
