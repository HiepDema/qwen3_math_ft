# Deployment Report: Qwen3-0.6B Math Reasoning API

## 1. Deployment Setting

| Component | Specification |
|-----------|--------------|
| **Model** | Qwen3-0.6B (CPT+SFT fine-tuned) |
| **Model Source** | [hiep-2/qwen3-0.6b-math-cpt-sft](https://huggingface.co/hiep-2/qwen3-0.6b-math-cpt-sft) |
| **GPU** | NVIDIA A10 (24GB VRAM) |
| **Quantization** | 4-bit QLoRA (float16 compute) |
| **Framework** | FastAPI + Unsloth + Transformers |
| **Serving** | Uvicorn, single worker |
| **Max Sequence Length** | 1024 tokens |
| **Max New Tokens** | 256 tokens |
| **Decoding** | Greedy (do_sample=False) |
| **Batch Support** | Up to 64 questions per request |

---

## 2. Backend Comparison

### 2.1 Single Request Performance

| Metric | Transformers (Baseline) | Optimized (Static KV Cache) | Improvement |
|--------|:-----------------------:|:---------------------------:|:-----------:|
| **Latency P50** | 1002.9 ms | 1028.5 ms | -2.5% |
| **Latency P95** | 1192.0 ms | 1425.3 ms | -19.6% |
| **Latency Mean** | 937.5 ms | 993.9 ms | -6.0% |
| **Latency Min** | 557.5 ms | 595.7 ms | -6.8% |
| **Throughput** | 1.07 req/s | 1.01 req/s | -5.6% |
| **Tokens/sec** | 18.7 tok/s | 17.6 tok/s | -5.9% |
| **Prefill** | 140.6 ms | 149.1 ms | -6.0% |
| **Decode** | 796.9 ms | 844.9 ms | -6.0% |

### 2.2 Timing Breakdown

```
Transformers (Baseline):
┌─────────────────────────────────────────────────┐
│           Single Request (~938 ms)              │
├────────────┬────────────────────────────────────┤
│  Prefill   │           Decode                   │
│  ~141 ms   │          ~797 ms                   │
│   (15%)    │           (85%)                    │
└────────────┴────────────────────────────────────┘

Optimized (Static KV Cache):
┌─────────────────────────────────────────────────┐
│           Single Request (~994 ms)              │
├────────────┬────────────────────────────────────┤
│  Prefill   │           Decode                   │
│  ~149 ms   │          ~845 ms                   │
│   (15%)    │           (85%)                    │
└────────────┴────────────────────────────────────┘
```

### 2.3 Batch Performance

| Batch Size | Transformers (tok/s) | Optimized (tok/s) | Improvement |
|:----------:|:--------------------:|:-----------------:|:-----------:|
| 1 | 17.4 | 17.3 | -0.6% |
| 2 | 24.4 | 24.1 | -1.2% |
| 4 | 50.4 | 50.5 | +0.2% |
| 8 | 96.1 | 98.5 | **+2.5%** |
| 16 | 186.3 | 185.5 | -0.4% |

```
Batch Throughput Comparison (tok/s):

 186 ┤ ██ ██                        ██ = Transformers
     │ ██ ██                        ░░ = Optimized
 150 ┤ ██ ██
     │ ██ ██
 120 ┤ ██ ██
     │ ██ ░░        ██ ░░
  96 ┤ ██ ░░        ██ ░░
     │ ██ ░░        ██ ░░
  50 ┤ ██ ░░  ██ ░░ ██ ░░
  24 ┤ ██ ░░  ██ ░░ ██ ░░  ██ ░░
  17 ┤ ██ ░░  ██ ░░ ██ ░░  ██ ░░  ██ ░░
     └──────────────────────────────────────
      BS=16   BS=8   BS=4   BS=2   BS=1
```

**Conclusion**: Batch performance is nearly identical between the two backends (~±2%).

### 2.4 Concurrent Request Performance

| Concurrency | Transformers (RPS) | Optimized (RPS) | Improvement |
|:-----------:|:------------------:|:---------------:|:-----------:|
| 1 | 1.55 | 1.55 | 0% |
| 2 | 1.60 | **3.40** | **+112%** |
| 4 | 0.16 | **3.90** | **+2337%** |
| 8 | 0.19 | 0.42 | +121% |

```
Concurrent Request Handling (Effective RPS):

  3.9 ┤              ░░
      │         ░░   ░░
  3.4 ┤         ░░   ░░
      │         ░░   ░░
  2.0 ┤         ░░   ░░
      │    ░░   ░░   ░░
  1.6 ┤██  ░░   ░░   ░░         ██ = Transformers
  1.5 ┤██  ██   ░░   ░░         ░░ = Optimized
      │██  ██   ░░   ░░
  0.4 ┤██  ██   ░░   ░░   ██ ░░
  0.2 ┤██  ██   ██   ██   ██ ░░
      └──────────────────────────
       C=1 C=2  C=4  C=4  C=8
```

**Critical finding**: Static KV Cache dramatically improves concurrent request handling:
- Concurrency 2: 1.60 → 3.40 RPS (+112%)
- Concurrency 4: 0.16 → 3.90 RPS (+2337%)
- Server no longer collapses at concurrency 4

---

## 3. Analysis

### 3.1 Why Static KV Cache Helps Concurrency

Dynamic KV cache causes **memory fragmentation** under concurrent load — multiple requests compete for GPU memory allocation, causing contention. Static cache pre-allocates fixed memory regions, eliminating this contention.

| Aspect | Dynamic KV Cache | Static KV Cache |
|--------|:----------------:|:---------------:|
| Memory allocation | Per-token, on demand | Pre-allocated |
| Concurrent overhead | High (fragmentation) | Low (fixed layout) |
| Single request | Slightly faster | Slightly slower |
| Multiple requests | **Collapses** at 4+ | **Stable** up to 4 |

### 3.2 Why Single Request is Slightly Slower

Static cache pre-allocates memory for max sequence length (1024 tokens) even for short responses (~17 tokens). This wastes memory bandwidth on unused cache slots, adding ~6% overhead for single short requests.

### 3.3 Optimal Strategy

| Scenario | Best Backend | Reason |
|----------|:------------:|--------|
| 1 user, interactive | Transformers | Lower single-request latency |
| 2-4 concurrent users | **Optimized** | Handles concurrency without collapse |
| Batch processing | Either | Similar throughput |
| High concurrency (8+) | vLLM needed | Both degrade at 8+ |

---

## 4. Performance Summary

### Best Results Per Backend

| Metric | Transformers | Optimized | Best |
|--------|:------------:|:---------:|:----:|
| Single request latency | **938 ms** | 994 ms | Transformers |
| Batch 16 throughput | **186 tok/s** | 185 tok/s | Tie |
| Concurrent 4 RPS | 0.16 | **3.90** | Optimized |
| Concurrent 2 RPS | 1.60 | **3.40** | Optimized |

### Recommended Configuration

| Use Case | Backend | Expected Performance |
|----------|:-------:|---------------------|
| Single user, low latency | `transformers` | ~940 ms/request |
| 2-4 concurrent users | `optimized` | 3.4-3.9 req/s |
| Batch jobs (offline) | Either | ~186 tok/s at BS=16 |
| Production (5+ users) | `vllm` | Requires vLLM installation |

---

## 5. Deployment Commands

```bash
# Baseline (best for single user)
python scripts/serve_model.py --model-path hiep-2/qwen3-0.6b-math-cpt-sft --backend transformers

# Optimized (best for 2-4 concurrent users)
python scripts/serve_model.py --model-path hiep-2/qwen3-0.6b-math-cpt-sft --backend optimized

# vLLM (production, 5+ users)
pip install vllm
python scripts/serve_model.py --model-path hiep-2/qwen3-0.6b-math-cpt-sft --backend vllm

# Benchmark any backend
python scripts/benchmark_deployment.py --output outputs/benchmark_results.json
```

---

## 6. API Reference

### POST /solve
```json
// Request
{"question": "Solve for x: 3x + 7 = 22", "max_new_tokens": 256}

// Response
{"question": "...", "response": "Approach: ...\nAnswer: 5", "time_ms": 938.0}
```

### POST /batch
```json
// Request
{"questions": ["What is 2+2?", "Solve: 5x=25"], "max_new_tokens": 256}

// Response
{"results": [...], "total_time_ms": 1583.0}
```

### GET /health
```json
{"status": "ok", "backend": "optimized", "model_loaded": true}
```
