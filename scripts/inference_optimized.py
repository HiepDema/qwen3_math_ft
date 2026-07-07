"""Optimized inference with KV Cache + Static Cache for production.

Optimizations applied:
1. KV Cache (default in transformers generate) - avoids recomputing past keys/values
2. Static KV Cache - pre-allocates cache memory to avoid dynamic allocation overhead
3. torch.compile - JIT compile the model for faster forward passes
4. Batch inference - process multiple samples in parallel

Usage:
    # Standard inference (with KV Cache)
    python scripts/inference_optimized.py --model-path outputs/grpo_lsreasoning_dense/final

    # With torch.compile optimization
    python scripts/inference_optimized.py --model-path outputs/grpo_lsreasoning_dense/final --compile

    # Benchmark mode (compare optimized vs baseline)
    python scripts/inference_optimized.py --model-path outputs/grpo_lsreasoning_dense/final --benchmark
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import StaticCache
from unsloth import FastLanguageModel


SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, "
    "show your reasoning clearly, then give the final answer."
)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_model(model_path, max_seq_length=1024):
    """Load model optimized for inference."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def build_prompts(questions, tokenizer):
    """Build tokenized prompts from questions."""
    prompts = []
    for q in questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))
    return prompts


def inference_baseline(model, tokenizer, questions, batch_size=8, max_new_tokens=256):
    """Baseline inference with default KV Cache."""
    responses = []
    prompts = build_prompts(questions, tokenizer)

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=768,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        for j, output in enumerate(outputs):
            prompt_len = inputs["input_ids"][j].ne(tokenizer.pad_token_id).sum()
            response = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            responses.append(response.strip())

    return responses


def inference_static_cache(model, tokenizer, questions, batch_size=8, max_new_tokens=256):
    """Inference with Static KV Cache (pre-allocated, no reallocation overhead)."""
    responses = []
    prompts = build_prompts(questions, tokenizer)

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=768,
        ).to(model.device)

        current_batch_size = inputs["input_ids"].shape[0]
        seq_length = inputs["input_ids"].shape[1] + max_new_tokens

        past_key_values = StaticCache(
            config=model.config,
            batch_size=current_batch_size,
            max_cache_len=seq_length,
            device=model.device,
            dtype=model.dtype,
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                past_key_values=past_key_values,
            )

        for j, output in enumerate(outputs):
            prompt_len = inputs["input_ids"][j].ne(tokenizer.pad_token_id).sum()
            response = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            responses.append(response.strip())

    return responses


def inference_compiled(model, tokenizer, questions, batch_size=8, max_new_tokens=256):
    """Inference with torch.compile optimization."""
    model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    # Warmup
    warmup_prompt = build_prompts(["What is 2 + 2?"], tokenizer)
    warmup_inputs = tokenizer(warmup_prompt, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        model.generate(**warmup_inputs, max_new_tokens=10, do_sample=False, use_cache=True)

    return inference_baseline(model, tokenizer, questions, batch_size, max_new_tokens)


def benchmark(model, tokenizer, questions, batch_size=8, max_new_tokens=256):
    """Benchmark different inference strategies."""
    print(f"\nBenchmarking with {len(questions)} questions, batch_size={batch_size}")
    print("=" * 60)

    results = {}

    # Baseline (dynamic KV Cache)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    responses_baseline = inference_baseline(model, tokenizer, questions, batch_size, max_new_tokens)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_baseline = time.perf_counter() - start
    results["baseline_kv_cache"] = {
        "time": t_baseline,
        "tokens_per_sec": len(questions) * max_new_tokens / t_baseline,
    }
    print(f"  Baseline (dynamic KV Cache):  {t_baseline:.2f}s  ({results['baseline_kv_cache']['tokens_per_sec']:.0f} tok/s)")

    # Static KV Cache
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    responses_static = inference_static_cache(model, tokenizer, questions, batch_size, max_new_tokens)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_static = time.perf_counter() - start
    results["static_kv_cache"] = {
        "time": t_static,
        "tokens_per_sec": len(questions) * max_new_tokens / t_static,
    }
    speedup = t_baseline / t_static if t_static > 0 else 0
    print(f"  Static KV Cache:              {t_static:.2f}s  ({results['static_kv_cache']['tokens_per_sec']:.0f} tok/s)  {speedup:.2f}x")

    print()
    print(f"  Speedup (Static vs Baseline): {speedup:.2f}x")
    print("=" * 60)

    return results, responses_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--test-file", type=str,
                        default="data/lsreasoning_split/test.jsonl")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--compile", action="store_true",
                        help="Apply torch.compile (requires warmup)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark comparing strategies")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Save responses to JSONL file")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    model, tokenizer = load_model(args.model_path)
    print("Model loaded!")

    # Load test questions
    test_data = load_jsonl(args.test_file)[:args.num_samples]
    questions = [item["question"] for item in test_data]
    print(f"Test questions: {len(questions)}")

    if args.benchmark:
        results, responses = benchmark(
            model, tokenizer, questions, args.batch_size, args.max_new_tokens
        )
        # Save benchmark results
        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Benchmark results saved to {args.output_file}")
    else:
        # Run optimized inference
        if args.compile:
            print("Applying torch.compile (warmup may take a moment)...")
            responses = inference_compiled(
                model, tokenizer, questions, args.batch_size, args.max_new_tokens
            )
        else:
            print("Running inference with static KV cache...")
            responses = inference_static_cache(
                model, tokenizer, questions, args.batch_size, args.max_new_tokens
            )

        print(f"\nGenerated {len(responses)} responses")
        for i in range(min(3, len(responses))):
            print(f"\n  Q: {questions[i][:80]}")
            print(f"  A: {responses[i][:150]}")

    # Save responses
    if args.output_file and not args.benchmark:
        output = []
        for q, r, item in zip(questions, responses, test_data):
            output.append({
                "question": q,
                "response": r,
                "true_answer": item["answer"],
            })
        with open(args.output_file, "w", encoding="utf-8") as f:
            for item in output:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Responses saved to {args.output_file}")


if __name__ == "__main__":
    main()
