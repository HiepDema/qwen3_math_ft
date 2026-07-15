"""Serve the best model (CPT+SFT) as a REST API with Prometheus monitoring.

Endpoints:
    POST /solve   - Solve a math problem
    POST /batch   - Solve multiple problems
    GET  /health  - Health check
    GET  /metrics - Prometheus metrics

Backends:
    transformers  - Default, simple
    optimized     - Static KV Cache + torch.compile (faster)
    vllm          - Production-grade (fastest, requires vllm package)

Metrics:
    request_count          - Counter (labels: endpoint, status)
    request_latency_seconds - Histogram (labels: endpoint)
    model_loaded           - Gauge (1 if model is loaded, 0 otherwise)
    gpu_memory_used_bytes  - Gauge (GPU memory usage in bytes)

Usage:
    python scripts/serve_model_monitored.py --model-path hiep-2/qwen3-0.6b-math-cpt-sft --backend vllm

Requirements:
    pip install fastapi uvicorn prometheus_client
    pip install vllm  # optional, for vllm backend
"""

import argparse
import time
import threading

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
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

# --- Prometheus Metrics ---
REQUEST_COUNT = Counter(
    "request_count",
    "Total number of requests",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether the model is currently loaded (1=yes, 0=no)",
)

GPU_MEMORY_USED = Gauge(
    "gpu_memory_used_bytes",
    "GPU memory usage in bytes",
)


def update_gpu_metrics():
    """Periodically update GPU memory metrics."""
    while True:
        try:
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated()
                GPU_MEMORY_USED.set(memory_used)
        except Exception:
            pass
        time.sleep(15)


# Start background thread for GPU metrics
gpu_metrics_thread = threading.Thread(target=update_gpu_metrics, daemon=True)
gpu_metrics_thread.start()


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
    """Build chat prompts from questions."""
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
    """Generate using transformers + unsloth (baseline)."""
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
    """Generate with static KV cache (optimized)."""
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
    """Generate using vLLM (production)."""
    from vllm import SamplingParams

    prompts = build_prompts(questions)
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
    )
    outputs = model.generate(prompts, sampling_params)
    return [o.outputs[0].text.strip() for o in outputs]


@app.get("/health")
def health():
    return {"status": "ok", "backend": backend, "model_loaded": model is not None}


@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


def get_generator():
    """Get the appropriate generator function based on backend."""
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

    gen = get_generator()
    start = time.perf_counter()
    try:
        responses = gen([req.question], req.max_new_tokens)
        elapsed = time.perf_counter() - start

        REQUEST_COUNT.labels(endpoint="/solve", status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/solve").observe(elapsed)

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

        results = [
            SolveResponse(
                question=q,
                response=r,
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
    """Load model with unsloth/transformers."""
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
    """Load model with vLLM."""
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
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of uvicorn workers (for concurrent requests)")
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
    print(f"  Workers: {args.workers}")
    print(f"  POST /solve   - Solve one problem")
    print(f"  POST /batch   - Solve multiple problems")
    print(f"  GET  /health  - Health check")
    print(f"  GET  /metrics - Prometheus metrics")
    print()

    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
