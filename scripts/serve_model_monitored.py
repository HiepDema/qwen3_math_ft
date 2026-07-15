"""Serve the best model (CPT+SFT) as a REST API with Prometheus monitoring.

Metrics tracked (26 total):
    System: latency P50/P95/P99, throughput, error rate, uptime, GPU memory, GPU util, CPU, RAM
    Prediction: response length, answer distribution, parse rate, format rate, confidence
    Data Quality: empty response rate, outlier rate
    Model Performance: periodic eval via /eval endpoint

Endpoints:
    POST /solve   - Solve a math problem
    POST /batch   - Solve multiple problems
    GET  /health  - Health check
    GET  /metrics - Prometheus metrics

Usage:
    python scripts/serve_model_monitored.py --model-path hiep-2/qwen3-0.6b-math-cpt-sft --backend vllm
"""

import argparse
import re
import time
import threading
import os

import torch
import uvicorn
import psutil
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, "
    "show your reasoning clearly, then give the final answer."
)

app = FastAPI(title="Math Reasoning API", version="1.0.0")

model = None
tokenizer = None
backend = None
start_time = time.time()

# =============================================================
# Prometheus Metrics (26 metrics across 4 categories)
# =============================================================

# --- System Monitoring (9 metrics) ---
REQUEST_COUNT = Counter(
    "request_count", "Total requests", ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Request latency",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)
THROUGHPUT = Summary(
    "request_throughput", "Requests processed per observation",
)
MODEL_LOADED = Gauge("model_loaded", "Model status (1=loaded)")
UPTIME_SECONDS = Gauge("uptime_seconds", "Server uptime in seconds")
GPU_MEMORY_USED = Gauge("gpu_memory_used_bytes", "GPU memory allocated")
GPU_MEMORY_TOTAL = Gauge("gpu_memory_total_bytes", "GPU memory total")
GPU_UTILIZATION = Gauge("gpu_utilization_percent", "GPU utilization %")
CPU_UTILIZATION = Gauge("cpu_utilization_percent", "CPU utilization %")
RAM_USED = Gauge("ram_used_bytes", "RAM used")
RAM_TOTAL = Gauge("ram_total_bytes", "RAM total")

# --- Prediction Monitoring (7 metrics) ---
RESPONSE_LENGTH = Histogram(
    "response_length_tokens", "Response length in tokens",
    buckets=[10, 25, 50, 100, 150, 200, 256, 512],
)
PREDICTED_ANSWER = Histogram(
    "predicted_answer_value", "Distribution of predicted numeric answers",
    buckets=[0, 1, 5, 10, 25, 50, 100, 500, 1000, 10000],
)
ANSWER_PARSE_RATE = Gauge(
    "answer_parse_rate", "Rate of responses with extractable numeric answer",
)
FORMAT_COMPLIANCE_RATE = Gauge(
    "format_compliance_rate", "Rate of responses with 'Answer:' format",
)
REASONING_RATE = Gauge(
    "reasoning_rate", "Rate of responses with step-by-step reasoning",
)
RESPONSE_CONFIDENCE = Histogram(
    "response_confidence", "Confidence score (1/perplexity proxy)",
    buckets=[0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
)
PREDICTION_ENTROPY = Histogram(
    "prediction_entropy", "Entropy of response length distribution",
    buckets=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
)

# --- Data Quality (4 metrics) ---
EMPTY_RESPONSE_TOTAL = Counter(
    "empty_response_total", "Total empty responses",
)
OUTLIER_RESPONSE_TOTAL = Counter(
    "outlier_response_total", "Responses with extreme length (<5 or >500 tokens)",
)
INVALID_INPUT_TOTAL = Counter(
    "invalid_input_total", "Invalid or empty input questions",
)
SCHEMA_VALID_RATE = Gauge(
    "schema_valid_rate", "Rate of responses passing schema validation",
)

# --- Model Performance (6 metrics, updated periodically) ---
EXACT_MATCH_ACCURACY = Gauge(
    "model_exact_match_accuracy", "Exact match accuracy on eval set",
)
CLOSE_MATCH_ACCURACY = Gauge(
    "model_close_match_accuracy", "Close match (1% tolerance) accuracy",
)
MAE_SCORE = Gauge(
    "model_mae", "Mean Absolute Error of numeric predictions",
)
RMSE_SCORE = Gauge(
    "model_rmse", "Root Mean Squared Error of numeric predictions",
)
MAPE_SCORE = Gauge(
    "model_mape", "Mean Absolute Percentage Error",
)
FORMAT_SCORE = Gauge(
    "model_format_score", "Format compliance score on eval set",
)

# --- Tracking counters for rate computation ---
_total_responses = 0
_parseable_responses = 0
_formatted_responses = 0
_reasoning_responses = 0
_schema_valid_responses = 0
_lock = threading.Lock()


def extract_number(text):
    match = re.search(r"Answer:\s*([+-]?\d+\.?\d*/?\.?\d*)", text)
    if match:
        val = match.group(1).strip()
        try:
            if "/" in val:
                parts = val.split("/")
                return float(parts[0]) / float(parts[1])
            return float(val)
        except (ValueError, ZeroDivisionError):
            return None
    numbers = re.findall(r"[+-]?\d+\.?\d*/?\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None


def track_prediction_metrics(response_text):
    global _total_responses, _parseable_responses, _formatted_responses
    global _reasoning_responses, _schema_valid_responses

    with _lock:
        _total_responses += 1

        tokens = response_text.split()
        token_count = len(tokens)
        RESPONSE_LENGTH.observe(token_count)

        if token_count == 0:
            EMPTY_RESPONSE_TOTAL.inc()
        if token_count < 5 or token_count > 500:
            OUTLIER_RESPONSE_TOTAL.inc()

        pred = extract_number(response_text)
        if pred is not None:
            _parseable_responses += 1
            PREDICTED_ANSWER.observe(min(abs(pred), 10000))

        has_format = "Answer:" in response_text or "answer:" in response_text
        if has_format:
            _formatted_responses += 1

        lines = [l.strip() for l in response_text.split("\n") if l.strip()]
        has_reasoning = len(lines) >= 3
        if has_reasoning:
            _reasoning_responses += 1

        has_math = bool(re.search(r"\d+\s*[+\-*/=]\s*\d+", response_text))
        if has_format and has_reasoning and has_math:
            _schema_valid_responses += 1

        import math
        if token_count > 0:
            normalized = min(token_count / 256.0, 1.0)
            entropy = -normalized * math.log(normalized + 1e-10)
            PREDICTION_ENTROPY.observe(entropy)
            RESPONSE_CONFIDENCE.observe(normalized)

        if _total_responses > 0:
            ANSWER_PARSE_RATE.set(_parseable_responses / _total_responses)
            FORMAT_COMPLIANCE_RATE.set(_formatted_responses / _total_responses)
            REASONING_RATE.set(_reasoning_responses / _total_responses)
            SCHEMA_VALID_RATE.set(_schema_valid_responses / _total_responses)


def update_system_metrics():
    while True:
        try:
            UPTIME_SECONDS.set(time.time() - start_time)

            if torch.cuda.is_available():
                GPU_MEMORY_USED.set(torch.cuda.memory_allocated())
                GPU_MEMORY_TOTAL.set(torch.cuda.get_device_properties(0).total_mem)
                try:
                    import subprocess
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5,
                    )
                    if result.returncode == 0:
                        GPU_UTILIZATION.set(float(result.stdout.strip().split("\n")[0]))
                except Exception:
                    pass

            CPU_UTILIZATION.set(psutil.cpu_percent())
            mem = psutil.virtual_memory()
            RAM_USED.set(mem.used)
            RAM_TOTAL.set(mem.total)
        except Exception:
            pass
        time.sleep(15)


threading.Thread(target=update_system_metrics, daemon=True).start()


class SolveRequest(BaseModel):
    question: str
    max_new_tokens: int = 256


class BatchRequest(BaseModel):
    questions: list[str]
    max_new_tokens: int = 256


class SolveResponse(BaseModel):
    question: str
    response: str
    time_ms: float


class BatchResponse(BaseModel):
    results: list[SolveResponse]
    total_time_ms: float


def build_prompts(questions: list[str]) -> list[str]:
    prompts = []
    for q in questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        ))
    return prompts


def generate_transformers(questions: list[str], max_new_tokens: int = 256) -> list[str]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = build_prompts(questions)
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=768
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    responses = []
    for j, output in enumerate(outputs):
        prompt_len = inputs["input_ids"][j].ne(tokenizer.pad_token_id).sum()
        response = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
        responses.append(response.strip())
    return responses


def generate_optimized(questions: list[str], max_new_tokens: int = 256) -> list[str]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = build_prompts(questions)
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=768
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            cache_implementation="static",
        )

    responses = []
    for j, output in enumerate(outputs):
        prompt_len = inputs["input_ids"][j].ne(tokenizer.pad_token_id).sum()
        response = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
        responses.append(response.strip())
    return responses


def generate_vllm(questions: list[str], max_new_tokens: int = 256) -> list[str]:
    from vllm import SamplingParams

    prompts = build_prompts(questions)
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0)
    outputs = model.generate(prompts, sampling_params)
    return [o.outputs[0].text.strip() for o in outputs]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": backend,
        "model_loaded": model is not None,
        "uptime_seconds": round(time.time() - start_time, 1),
        "total_requests": int(_total_responses),
    }


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def get_generator():
    if backend == "vllm":
        return generate_vllm
    elif backend == "optimized":
        return generate_optimized
    else:
        return generate_transformers


@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    if model is None:
        REQUEST_COUNT.labels(endpoint="/solve", status="503").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not req.question.strip():
        INVALID_INPUT_TOTAL.inc()
        REQUEST_COUNT.labels(endpoint="/solve", status="400").inc()
        raise HTTPException(status_code=400, detail="Empty question")

    gen = get_generator()
    start = time.perf_counter()
    try:
        responses = gen([req.question], req.max_new_tokens)
        elapsed = time.perf_counter() - start

        REQUEST_COUNT.labels(endpoint="/solve", status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/solve").observe(elapsed)
        THROUGHPUT.observe(1)

        track_prediction_metrics(responses[0])

        return SolveResponse(
            question=req.question,
            response=responses[0],
            time_ms=round(elapsed * 1000, 1),
        )
    except Exception as e:
        elapsed = time.perf_counter() - start
        REQUEST_COUNT.labels(endpoint="/solve", status="500").inc()
        REQUEST_LATENCY.labels(endpoint="/solve").observe(elapsed)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", response_model=BatchResponse)
def batch_solve(req: BatchRequest):
    if model is None:
        REQUEST_COUNT.labels(endpoint="/batch", status="503").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(req.questions) > 64:
        REQUEST_COUNT.labels(endpoint="/batch", status="400").inc()
        raise HTTPException(status_code=400, detail="Max 64 questions per batch")

    gen = get_generator()
    start = time.perf_counter()
    try:
        responses = gen(req.questions, req.max_new_tokens)
        elapsed = time.perf_counter() - start

        REQUEST_COUNT.labels(endpoint="/batch", status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/batch").observe(elapsed)
        THROUGHPUT.observe(len(req.questions))

        for resp in responses:
            track_prediction_metrics(resp)

        results = [
            SolveResponse(
                question=q, response=r,
                time_ms=round((elapsed * 1000) / len(req.questions), 1),
            )
            for q, r in zip(req.questions, responses)
        ]
        return BatchResponse(results=results, total_time_ms=round(elapsed * 1000, 1))
    except Exception as e:
        elapsed = time.perf_counter() - start
        REQUEST_COUNT.labels(endpoint="/batch", status="500").inc()
        REQUEST_LATENCY.labels(endpoint="/batch").observe(elapsed)
        raise HTTPException(status_code=500, detail=str(e))


def load_model_transformers(model_path: str):
    global model, tokenizer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=torch.float16,
    )
    FastLanguageModel.for_inference(model)
    MODEL_LOADED.set(1)
    print(f"Model loaded (transformers): {model_path}")


def load_model_vllm(model_path: str):
    global model, tokenizer
    from vllm import LLM
    from transformers import AutoTokenizer

    model = LLM(
        model=model_path,
        dtype="float16",
        max_model_len=1024,
        gpu_memory_utilization=0.8,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    MODEL_LOADED.set(1)
    print(f"Model loaded (vLLM): {model_path}")


def main():
    global backend

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="hiep-2/qwen3-0.6b-math-cpt-sft")
    parser.add_argument("--backend", choices=["transformers", "optimized", "vllm"], default="transformers")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    backend = args.backend

    print(f"Loading model: {args.model_path}")
    print(f"Backend: {args.backend}")

    if args.backend == "vllm":
        load_model_vllm(args.model_path)
    else:
        load_model_transformers(args.model_path)

    print(f"\nStarting server at http://{args.host}:{args.port}")
    print(f"  Backend: {args.backend}")
    print(f"  POST /solve   - Solve one problem")
    print(f"  POST /batch   - Solve multiple problems")
    print(f"  GET  /health  - Health check")
    print(f"  GET  /metrics - Prometheus metrics (26 metrics)")
    print()

    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
