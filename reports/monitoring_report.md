# Monitoring Report: Qwen3-0.6B Math Reasoning API

## 1. Deployment Setting

| Component | Specification |
|-----------|--------------|
| **Model** | Qwen3-0.6B (SFT fine-tuned) |
| **Model Source** | [hiep-2/qwen3-0.6b-math-cpt-sft](https://huggingface.co/hiep-2/qwen3-0.6b-math-cpt-sft) |
| **GPU** | NVIDIA A10 (24GB VRAM) |
| **Backend** | vLLM (PagedAttention + Continuous Batching) |
| **Framework** | FastAPI + Uvicorn + Prometheus |
| **Monitoring** | 26 Prometheus metrics across 4 categories |

---

## 2. Model Performance (Evaluation)

### 2.1 Overall Accuracy

| Metric | Score | Count |
|--------|:-----:|:-----:|
| **Exact Match** | **83.0%** | 415/500 |
| **Close Match (1% tolerance)** | **83.4%** | 417/500 |
| **Format Compliance (has "Answer:")** | **100.0%** | 500/500 |
| **Reasoning (step-by-step)** | **100.0%** | 500/500 |

### 2.2 Accuracy by Problem Type

| Problem Type | Accuracy | Correct/Total |
|-------------|:--------:|:-------------:|
| Subtract the second number from the first | **100%** | 59/59 |
| Add the two numbers | **100%** | 58/58 |
| Reduce the fraction to simplest form | **100%** | 46/46 |
| Divide evenly | **100%** | 45/45 |
| Solve using algebra | **100%** | 34/34 |
| Evaluate the integer exponent | **98%** | 43/44 |
| Multiply the numbers | **96%** | 49/51 |
| Solve the linear equation | **85%** | 46/54 |
| Solve a two-step equation | **56%** | 35/63 |
| Solve the inequality | **0%** | 0/46 |

### 2.3 Error Analysis

**Inequality failure (0%)**: Model extracts correct numeric value nhưng thiếu comparison operator.

```
Q: Solve: -7x + 8 < -27
Expected: x < 5     → Model output: 5.0    (số đúng, thiếu "< ")
Q: Solve: 9x + 6 ≤ -3
Expected: x ≤ -1    → Model output: -1.0   (số đúng, thiếu "≤ ")
```

→ Root cause: `extract_number()` chỉ parse số, không parse biểu thức bất đẳng thức. Model **đã giải đúng** nhưng evaluation metric không capture được.

**Two-step equation (56%)**: Lỗi tính toán ở bước trung gian.

```
Q: Solve for x: 9x + 25 = -56    Expected: -9, Got: -8.0
Q: Solve for x: -2(x + -8) = 42  Expected: -13, Got: -10.0
```

→ Root cause: Phân phối dấu âm (negative distribution) và xử lý dấu ngoặc.

---

## 3. System Monitoring (9 Metrics)

### 3.1 Request Performance

| Metric | Value |
|--------|:-----:|
| **Total Requests** | 25 (20 single + 5 batch) |
| **Uptime** | 201.9 seconds |
| **Error Rate** | **0%** (0 errors / 25 requests) |

### 3.2 Latency (vLLM Backend)

| Endpoint | Requests | Total Time | Mean Latency | P95 |
|----------|:--------:|:----------:|:------------:|:---:|
| `/solve` | 20 | 1.637s | **81.8 ms** | < 100 ms |
| `/batch` (5 questions) | 1 | 0.153s | **152.7 ms** | — |

```
Latency Distribution (/solve):
─────────────────────────────────────────────────────
  < 50ms   │                                │  0 req
  < 100ms  │████████████████████████████████ │ 19 req (95%)
  < 250ms  │██                              │  1 req  (5%)
  < 500ms  │                                │  0 req
─────────────────────────────────────────────────────
  Mean: 81.8ms    95th percentile: < 100ms
```

### 3.3 Resource Utilization

| Resource | Value |
|----------|:-----:|
| **Process RSS Memory** | 1.40 GB |
| **Process Virtual Memory** | 21.56 GB |
| **Open File Descriptors** | 58 |
| **CPU Time** | 17.33 seconds |
| **GPU Memory** | Managed by vLLM (outside torch allocator) |

*Note: GPU metrics `gpu_memory_used_bytes=0` vì vLLM quản lý VRAM riêng qua custom allocator, không đi qua `torch.cuda.memory_allocated()`. Thực tế vLLM sử dụng ~80% VRAM (19.2GB) theo config `gpu_memory_utilization=0.8`.*

---

## 4. Prediction Monitoring (7 Metrics)

### 4.1 Response Quality

| Metric | Value | Status |
|--------|:-----:|:------:|
| **Answer Parse Rate** | 100% | ✅ Tất cả response đều extract được số |
| **Format Compliance** | 100% | ✅ Tất cả có "Answer:" format |
| **Reasoning Rate** | 100% | ✅ Tất cả có step-by-step (≥3 dòng) |
| **Schema Valid Rate** | 40% | ⚠️ Chỉ 40% có đủ cả format + reasoning + math expression |

### 4.2 Response Length

| Metric | Value |
|--------|:-----:|
| **Total Tokens** | 290 tokens (25 responses) |
| **Mean Length** | **11.6 tokens/response** |
| **Distribution** | 100% nằm trong khoảng 10-25 tokens |
| **Outliers** | 0 (không có response < 5 hoặc > 500 tokens) |

```
Response Length Distribution:
─────────────────────────────────────────────────────
  < 10 tokens  │                                │  0
  10-25 tokens │████████████████████████████████ │ 25 (100%)
  25-50 tokens │                                │  0
  > 50 tokens  │                                │  0
─────────────────────────────────────────────────────
```

→ Responses ngắn gọn, phù hợp với math questions đơn giản (i*i).

### 4.3 Predicted Answer Distribution

| Range | Count | Percentage |
|-------|:-----:|:----------:|
| 0-1 | 1 | 4% |
| 1-5 | 3 | 12% |
| 5-10 | 1 | 4% |
| 10-25 | 4 | 16% |
| 25-50 | 2 | 8% |
| 50-100 | 4 | 16% |
| 100-500 | 10 | 40% |

→ Phân bố hợp lý cho input `i*i` (i=1..20): giá trị tăng theo bình phương.

### 4.4 Confidence & Entropy

| Metric | Mean | Distribution |
|--------|:----:|:------------:|
| **Confidence Score** | 0.045 | Tất cả ≤ 0.1 (response ngắn → normalized length thấp) |
| **Prediction Entropy** | 0.14 | Tất cả ≤ 0.5 (response length đồng đều → entropy thấp) |

→ Low entropy = model output rất consistent, không có variance bất thường.

---

## 5. Data Quality (4 Metrics)

| Metric | Value | Threshold | Status |
|--------|:-----:|:---------:|:------:|
| **Empty Responses** | 0 | 0 | ✅ |
| **Outlier Responses** | 0 | < 1% | ✅ |
| **Invalid Inputs** | 0 | 0 | ✅ |
| **Schema Valid Rate** | 40% | > 80% | ⚠️ |

### Schema Validation Analysis

Schema valid = response có **cả 3**: format ("Answer:") + reasoning (≥3 dòng) + math expression (regex `\d+\s*[+\-*/=]\s*\d+`).

- 100% có format ✅
- 100% có reasoning ✅
- ~40% có math expression ⚠️

→ Nhiều response trả lời trực tiếp (vd: `"Answer: 400"`) mà không viết phép tính dạng `20 * 20 = 400`. Đây là behavior bình thường cho câu hỏi đơn giản — không phải lỗi chất lượng.

---

## 6. Monitoring Metrics Summary

### 26 Metrics Across 4 Categories

| # | Category | Metric | Prometheus Name | Value | Status |
|---|----------|--------|-----------------|:-----:|:------:|
| 1 | System | Latency P50 | `request_latency_seconds` | ~82ms | ✅ |
| 2 | System | Latency P95 | `request_latency_seconds` | <100ms | ✅ |
| 3 | System | Throughput | `request_throughput` | 25 req | ✅ |
| 4 | System | Error Rate | `request_count{status=5xx}` | 0% | ✅ |
| 5 | System | Uptime | `uptime_seconds` | 201.9s | ✅ |
| 6 | System | GPU Memory | `gpu_memory_used_bytes` | vLLM-managed | ℹ️ |
| 7 | System | GPU Utilization | `gpu_utilization_percent` | 0%* | ℹ️ |
| 8 | System | CPU Utilization | `cpu_utilization_percent` | idle | ✅ |
| 9 | System | RAM Usage | `ram_used_bytes` | 1.4GB RSS | ✅ |
| 10 | Prediction | Response Length | `response_length_tokens` | 11.6 avg | ✅ |
| 11 | Prediction | Answer Distribution | `predicted_answer_value` | Normal | ✅ |
| 12 | Prediction | Parse Rate | `answer_parse_rate` | 100% | ✅ |
| 13 | Prediction | Format Compliance | `format_compliance_rate` | 100% | ✅ |
| 14 | Prediction | Reasoning Rate | `reasoning_rate` | 100% | ✅ |
| 15 | Prediction | Confidence | `response_confidence` | 0.045 | ✅ |
| 16 | Prediction | Entropy | `prediction_entropy` | 0.14 | ✅ |
| 17 | Data Quality | Empty Responses | `empty_response_total` | 0 | ✅ |
| 18 | Data Quality | Outlier Responses | `outlier_response_total` | 0 | ✅ |
| 19 | Data Quality | Invalid Input | `invalid_input_total` | 0 | ✅ |
| 20 | Data Quality | Schema Valid Rate | `schema_valid_rate` | 40% | ⚠️ |
| 21 | Model Perf | Exact Match | `model_exact_match_accuracy` | 83.0% | ✅ |
| 22 | Model Perf | Close Match | `model_close_match_accuracy` | 83.4% | ✅ |
| 23 | Model Perf | MAE | `model_mae` | — | — |
| 24 | Model Perf | RMSE | `model_rmse` | — | — |
| 25 | Model Perf | MAPE | `model_mape` | — | — |
| 26 | Model Perf | Format Score | `model_format_score` | 100% | ✅ |

*GPU metrics = 0 vì vLLM quản lý VRAM qua custom allocator, `nvidia-smi` shows actual usage ~19.2GB.*

---

## 7. Key Findings

### Strengths
1. **83% accuracy** trên 500 test samples — strong performance cho 0.6B model
2. **100% format compliance** — model luôn output đúng format "Answer: X"
3. **81.8ms mean latency** với vLLM — production-ready response time
4. **0% error rate** — không có request nào fail
5. **Consistent output** — entropy thấp, không có outlier responses

### Areas for Improvement
1. **Inequality solving (0%)** — cần cải thiện evaluation metric để capture comparison operators, hoặc train thêm data inequality
2. **Two-step equations (56%)** — cần thêm training data cho multi-step reasoning với negative numbers
3. **Schema valid rate (40%)** — responses ngắn không có math expression rõ ràng; cần adjust schema validation rule hoặc prompt

### Recommendations
1. **Short-term**: Cập nhật `extract_number()` để parse inequality expressions (`x < 5`, `x ≤ -1`)
2. **Mid-term**: Thêm inequality + multi-step equation samples vào training data
3. **Long-term**: Implement automated periodic evaluation (cron job chạy eval set mỗi 24h) để detect accuracy drift

---

## 8. Monitoring Architecture

```
┌─────────────┐     ┌──────────────────────┐     ┌─────────────┐
│   Client     │────▶│  FastAPI + vLLM       │────▶│  Prometheus  │
│  (requests)  │◀────│  serve_model_         │     │  (scrape     │
│              │     │  monitored.py         │     │   /metrics)  │
└─────────────┘     │                      │     └──────┬──────┘
                    │  GET /metrics ────────┼────────────┘
                    │  26 Prometheus metrics│     ┌──────┴──────┐
                    └──────────────────────┘     │   Grafana    │
                                                 │  (dashboard) │
                                                 └─────────────┘
```

### Endpoints

| Endpoint | Method | Description |
|----------|:------:|-------------|
| `/solve` | POST | Solve single math problem |
| `/batch` | POST | Solve multiple problems (max 64) |
| `/health` | GET | Health check + uptime + request count |
| `/metrics` | GET | Prometheus metrics (26 metrics) |

---

## 9. Deployment Commands

```bash
# Deploy with vLLM + monitoring
VLLM_USE_V1=0 python scripts/serve_model_monitored.py \
    --model-path hiep-2/qwen3-0.6b-math-cpt-sft \
    --backend vllm

# Evaluate model accuracy
python scripts/evaluate_lsreasoning_v2.py \
    --model-path hiep-2/qwen3-0.6b-math-cpt-sft \
    --num-eval 500

# Run benchmark
python scripts/benchmark_deployment.py --output outputs/benchmark_results.json

# Check metrics
curl http://localhost:8000/metrics
curl http://localhost:8000/health
```
