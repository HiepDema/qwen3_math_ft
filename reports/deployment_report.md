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

## 2. Benchmark Results — Transformers Backend (Baseline)

### 2.1 Single Request Performance

| Metric | Value |
|--------|:-----:|
| **Latency P50** | 1591.2 ms |
| **Latency P95** | 2046.7 ms |
| **Latency P99** | 2046.7 ms |
| **Latency Mean** | 1513.7 ms |
| **Latency Min** | 881.3 ms |
| **Latency Max** | 2046.7 ms |
| **Throughput** | 0.66 req/s |
| **Tokens/sec** | 11.6 tok/s |
| **Avg Response Length** | 17.5 tokens |

### 2.2 Timing Breakdown

```
┌─────────────────────────────────────────────────────┐
│           Single Request (~1514 ms)                 │
├────────────┬────────────────────────────────────────┤
│  Prefill   │              Decode                    │
│  ~227 ms   │            ~1287 ms                    │
│   (15%)    │             (85%)                      │
└────────────┴────────────────────────────────────────┘
```

| Phase | Time | % | Description |
|-------|:----:|:-:|-------------|
| **Prefill** | 227 ms | 15% | Encode input prompt, compute initial KV cache |
| **Decode** | 1287 ms | 85% | Generate tokens autoregressively (~17.5 tokens) |

### 2.3 Batch Performance

| Batch Size | Total Time | Time/Request | Tokens/sec | Speedup vs BS=1 |
|:----------:|:----------:|:------------:|:----------:|:----------------:|
| 1 | 1234 ms | 1234 ms | 8.9 | 1.0x |
| 2 | 2908 ms | 1454 ms | 12.7 | 1.4x |
| 4 | 2792 ms | 698 ms | 27.6 | 3.1x |
| 8 | 2847 ms | 356 ms | 53.0 | 6.0x |
| **16** | **2831 ms** | **177 ms** | **104.2** | **11.7x** |

```
Tokens/sec vs Batch Size (Transformers Backend):

 104 ┤                                          ████
     │                                          ████
  80 ┤                                          ████
     │                              ████        ████
  53 ┤                              ████        ████
     │                              ████        ████
  40 ┤                              ████        ████
     │                  ████        ████        ████
  28 ┤                  ████        ████        ████
     │                  ████        ████        ████
  13 ┤      ████        ████        ████        ████
   9 ┤████  ████        ████        ████        ████
     └──────────────────────────────────────────────
      BS=1  BS=2        BS=4        BS=8       BS=16
```

**Key insight**: Batch size 16 delivers **11.7x throughput** over single request — near-linear GPU utilization scaling.

### 2.4 Concurrent Request Performance

| Concurrency | Wall Time | Mean Latency | Effective RPS |
|:-----------:|:---------:|:------------:|:-------------:|
| 1 | 1.15s | 1.15s | 0.87 |
| 2 | 2.30s | 2.21s | 0.87 |
| 4 | 59.07s | 58.94s | 0.07 |
| 8 | 107.17s | 106.98s | 0.07 |

**Critical finding**: Server collapses at concurrency ≥ 4 (from 0.87 RPS to 0.07 RPS). Single-worker FastAPI cannot handle concurrent GPU requests — they queue sequentially and each waits for the previous to complete.

---

## 3. Optimization Strategies

### 3.1 Available Backends

| Backend | Description | Command |
|---------|-------------|---------|
| `transformers` | Baseline, dynamic KV cache | `--backend transformers` |
| `optimized` | Static KV cache + torch.compile | `--backend optimized` |
| `vllm` | Continuous batching, PagedAttention | `--backend vllm` |

### 3.2 Static KV Cache

Pre-allocates memory for the KV cache instead of dynamically growing it per token:

```python
model.generate(..., cache_implementation="static")
```

**Expected improvement**: 5-15% latency reduction on decode phase by eliminating memory reallocation overhead per generated token.

### 3.3 torch.compile

JIT-compiles the model's forward pass into optimized CUDA kernels:

```python
model.forward = torch.compile(model.forward, mode="reduce-overhead")
```

**Expected improvement**: 10-30% faster forward passes after initial warmup. Most effective on repeated identical computation patterns (autoregressive decode).

### 3.4 vLLM Backend

Production-grade serving with:
- **Continuous batching**: New requests join mid-generation without waiting
- **PagedAttention**: Efficient KV cache memory management
- **Speculative decoding**: Predict multiple tokens at once

**Expected improvement**: 3-5x throughput, handles concurrent requests natively.

---

## 4. Performance Summary & Recommendations

### Current Baseline Performance

| Metric | Single Request | Batch (BS=16) |
|--------|:--------------:|:-------------:|
| Latency | 1514 ms | 177 ms/req |
| Throughput | 0.66 req/s | ~5.6 req/s |
| Tokens/sec | 11.6 | 104.2 |

### Recommendations by Use Case

| Use Case | Approach | Expected Performance |
|----------|---------|---------------------|
| Interactive (1 user) | `/solve`, transformers | ~1.5s per question |
| Batch jobs | `/batch` BS=16 | 177ms/question, 104 tok/s |
| Multiple users (2-4) | vLLM backend | ~500ms/question concurrent |
| High throughput | vLLM + batch | 300+ tok/s |

### Bottleneck Analysis

| Bottleneck | Impact | Solution |
|-----------|--------|----------|
| Autoregressive decode | 85% of latency | Shorter max_new_tokens, speculative decoding |
| Single worker | No concurrent handling | vLLM or multiple workers |
| Dynamic KV cache | Memory reallocation per token | Static cache (`--backend optimized`) |
| 4-bit dequantization | Per-token overhead | fp16 merged model (uses more VRAM) |
| No continuous batching | Requests queue up | vLLM backend |

---

## 5. Deployment Commands

```bash
# Baseline
python scripts/serve_model.py --model-path hiep-2/qwen3-0.6b-math-cpt-sft --backend transformers

# Optimized (static KV cache + torch.compile)
python scripts/serve_model.py --model-path hiep-2/qwen3-0.6b-math-cpt-sft --backend optimized

# vLLM (production)
pip install vllm
python scripts/serve_model.py --model-path hiep-2/qwen3-0.6b-math-cpt-sft --backend vllm

# Benchmark
python scripts/benchmark_deployment.py --output outputs/benchmark_results.json
```

---

## 6. API Reference

### POST /solve
```json
// Request
{"question": "Solve for x: 3x + 7 = 22", "max_new_tokens": 256}

// Response  
{"question": "...", "response": "Approach: ...\nAnswer: 5", "time_ms": 1514.0}
```

### POST /batch
```json
// Request
{"questions": ["What is 2+2?", "Solve: 5x=25"], "max_new_tokens": 256}

// Response
{"results": [...], "total_time_ms": 2831.0}
```

### GET /health
```json
{"status": "ok", "backend": "transformers", "model_loaded": true}
```
