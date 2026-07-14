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

## 2. Single Request Performance

| Metric | Value |
|--------|:-----:|
| **Latency P50** | 1002.9 ms |
| **Latency P95** | 1192.0 ms |
| **Latency P99** | 1192.0 ms |
| **Latency Mean** | 937.5 ms |
| **Latency Min** | 557.5 ms |
| **Latency Max** | 1192.0 ms |
| **Throughput** | 1.07 req/s |
| **Tokens/sec** | 18.7 tok/s |
| **Avg Response Length** | 17.5 tokens |

### Timing Breakdown

```
┌─────────────────────────────────────────────────┐
│           Single Request (~938 ms)              │
├────────────┬────────────────────────────────────┤
│  Prefill   │           Decode                   │
│  ~141 ms   │          ~797 ms                   │
│   (15%)    │           (85%)                    │
└────────────┴────────────────────────────────────┘
```

| Phase | Time | Description |
|-------|:----:|-------------|
| **Prefill** | ~141 ms (15%) | Encode input prompt, compute initial KV cache |
| **Decode** | ~797 ms (85%) | Generate tokens one by one (autoregressive) |

Decode dominates — model generates ~17.5 tokens per response at ~22 tokens/sec decode speed.

---

## 3. Batch Performance

| Batch Size | Total Time | Time/Request | Tokens/sec | Speedup vs BS=1 |
|:----------:|:----------:|:------------:|:----------:|:----------------:|
| 1 | 632 ms | 632 ms | 17.4 | 1.0x |
| 2 | 1515 ms | 758 ms | 24.4 | 1.4x |
| 4 | 1529 ms | 382 ms | 50.4 | 2.9x |
| 8 | 1572 ms | 197 ms | 96.1 | 5.5x |
| **16** | **1583 ms** | **99 ms** | **186.3** | **10.7x** |

### Batch Scaling Chart

```
Tokens/sec vs Batch Size:

 186 ┤                                          ████
     │                                          ████
 150 ┤                                          ████
     │                                          ████
 120 ┤                                          ████
     │                              ████        ████
  96 ┤                              ████        ████
     │                              ████        ████
  60 ┤                  ████        ████        ████
  50 ┤                  ████        ████        ████
     │      ████        ████        ████        ████
  24 ┤      ████        ████        ████        ████
  17 ┤████  ████        ████        ████        ████
     └──────────────────────────────────────────────
      BS=1  BS=2        BS=4        BS=8       BS=16
```

**Key insight**: Batch size 16 gives **10.7x throughput improvement** over single request at only 2.5x wall time — near-linear scaling due to GPU parallelism on the small 0.6B model.

---

## 4. Concurrent Request Performance

| Concurrency | Wall Time | Mean Latency | Max Latency | Effective RPS |
|:-----------:|:---------:|:------------:|:-----------:|:-------------:|
| 1 | 645 ms | 644 ms | 644 ms | 1.55 |
| 2 | 1252 ms | 1250 ms | 1250 ms | 1.60 |
| 4 | 25,090 ms | 24,922 ms | 25,089 ms | 0.16 |
| 8 | 41,211 ms | 40,943 ms | 41,207 ms | 0.19 |

### Analysis

```
Effective RPS vs Concurrency:

  1.60 ┤  ●────●
       │ /
  1.55 ┤●
       │
  1.00 ┤
       │
  0.50 ┤
       │              ●
  0.19 ┤                        ●
  0.16 ┤
       └──────────────────────────
        1     2       4         8
            Concurrency
```

**Concurrent requests degrade significantly beyond 2**. This is because:

1. **Single-worker server**: Uvicorn processes requests sequentially by default
2. **No continuous batching**: Requests queue up instead of being batched together
3. **GPU memory contention**: Multiple simultaneous forward passes compete for VRAM

**Recommendation**: Use the `/batch` endpoint instead of concurrent `/solve` calls — batch size 16 achieves 186 tok/s vs concurrent 4 at effectively 0.16 RPS.

---

## 5. Performance Summary

| Use Case | Recommended Approach | Expected Performance |
|----------|---------------------|---------------------|
| Interactive (1 user) | `/solve` endpoint | ~1s latency, 1 req/s |
| Batch processing | `/batch` with BS=16 | 99 ms/req, 186 tok/s |
| High throughput | `/batch` with BS=8-16 | 10x single request throughput |
| Multiple concurrent users | Queue + batch | NOT concurrent `/solve` |

---

## 6. Bottleneck Analysis

| Bottleneck | Impact | Mitigation |
|-----------|--------|-----------|
| **Autoregressive decoding** | 85% of latency | Shorter responses, speculative decoding |
| **Single worker** | No parallelism for concurrent users | Add workers or use vLLM |
| **4-bit quantization overhead** | Dequantization per token | Use merged fp16 model |
| **No continuous batching** | Concurrent requests queue | Switch to vLLM/TGI |

---

## 7. Production Recommendations

### Current Setup (suitable for demo/prototype)
- Single A10, single worker, FastAPI + Transformers
- Good for: ≤2 concurrent users, batch processing jobs

### Upgrade Path for Production

| Level | Change | Expected Improvement |
|-------|--------|---------------------|
| 1 | Switch to **vLLM** backend | 3-5x throughput (continuous batching) |
| 2 | Add **multiple workers** (2-4) | Linear RPS scaling for concurrent users |
| 3 | Use **fp16 merged model** (no 4-bit) | ~20% faster decode, uses more VRAM |
| 4 | Add **load balancer** + multiple GPUs | Horizontal scaling |

### Quick vLLM Deployment

```bash
pip install vllm
python scripts/serve_model.py --model-path hiep-2/qwen3-0.6b-math-cpt-sft --backend vllm
```

Expected improvement with vLLM:
- Continuous batching → handle concurrent requests efficiently
- PagedAttention → better memory utilization
- ~3-5x throughput improvement over current setup

---

## 8. API Reference

### POST /solve
```json
// Request
{"question": "Solve for x: 3x + 7 = 22", "max_new_tokens": 256}

// Response
{"question": "...", "response": "Approach: ...\nAnswer: 5", "time_ms": 937.5}
```

### POST /batch
```json
// Request
{"questions": ["What is 2+2?", "Solve: 5x=25"], "max_new_tokens": 256}

// Response
{"results": [...], "total_time_ms": 1583.3}
```

### GET /health
```json
{"status": "ok", "backend": "transformers", "model_loaded": true}
```

---

## 9. Deployment Commands

```bash
# Start server
python scripts/serve_model.py --model-path hiep-2/qwen3-0.6b-math-cpt-sft

# Run benchmark
python scripts/benchmark_deployment.py --url http://localhost:8000

# Access API docs
http://<server-ip>:8000/docs
```
