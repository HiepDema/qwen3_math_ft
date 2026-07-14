"""Serve the best model (CPT+SFT) as a REST API.

Endpoints:
    POST /solve  - Solve a math problem
    POST /batch  - Solve multiple problems
    GET  /health - Health check

Usage:
    # Using transformers (simple, works everywhere)
    python scripts/serve_model.py --model-path outputs/cpt_sft_lsreasoning/merged

    # Using vLLM (faster, recommended for production)
    python scripts/serve_model.py --model-path outputs/cpt_sft_lsreasoning/merged --backend vllm

    # Custom port
    python scripts/serve_model.py --model-path outputs/cpt_sft_lsreasoning/merged --port 8080

Requirements:
    pip install fastapi uvicorn
    pip install vllm  # optional, for vllm backend
"""

import argparse
import json
import time

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, "
    "show your reasoning clearly, then give the final answer."
)

app = FastAPI(title="Math Reasoning API", version="1.0.0")

model = None
tokenizer = None
backend = None


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


def generate_transformers(questions: list[str], max_new_tokens: int = 256) -> list[str]:
    """Generate using transformers + unsloth."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = []
    for q in questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

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


def generate_vllm(questions: list[str], max_new_tokens: int = 256) -> list[str]:
    """Generate using vLLM."""
    from vllm import SamplingParams

    prompts = []
    for q in questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
    )
    outputs = model.generate(prompts, sampling_params)
    return [o.outputs[0].text.strip() for o in outputs]


@app.get("/health")
def health():
    return {"status": "ok", "backend": backend, "model_loaded": model is not None}


@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    if backend == "vllm":
        responses = generate_vllm([req.question], req.max_new_tokens)
    else:
        responses = generate_transformers([req.question], req.max_new_tokens)
    elapsed = (time.perf_counter() - start) * 1000

    return SolveResponse(
        question=req.question,
        response=responses[0],
        time_ms=round(elapsed, 1),
    )


@app.post("/batch", response_model=BatchResponse)
def batch_solve(req: BatchRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(req.questions) > 64:
        raise HTTPException(status_code=400, detail="Max 64 questions per batch")

    start = time.perf_counter()
    if backend == "vllm":
        responses = generate_vllm(req.questions, req.max_new_tokens)
    else:
        responses = generate_transformers(req.questions, req.max_new_tokens)
    elapsed = (time.perf_counter() - start) * 1000

    results = [
        SolveResponse(question=q, response=r, time_ms=round(elapsed / len(req.questions), 1))
        for q, r in zip(req.questions, responses)
    ]
    return BatchResponse(results=results, total_time_ms=round(elapsed, 1))


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
    print(f"Model loaded (vLLM): {model_path}")


def main():
    global backend

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="outputs/cpt_sft_lsreasoning/merged")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    backend = args.backend

    print(f"Loading model: {args.model_path}")
    print(f"Backend: {args.backend}")

    if args.backend == "vllm":
        load_model_vllm(args.model_path)
    else:
        load_model_transformers(args.model_path)

    print(f"\nStarting server at http://{args.host}:{args.port}")
    print(f"  POST /solve  - Solve one problem")
    print(f"  POST /batch  - Solve multiple problems")
    print(f"  GET  /health - Health check")
    print()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
