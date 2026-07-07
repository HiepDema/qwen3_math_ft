"""Evaluation module for LSReasoning experiment.

Evaluates on the pre-split test file (JSONL) for fair comparison.
Logs results to wandb with comparison charts.

Metrics:
- Exact Match: answer is exactly correct
- Close Match: answer within 1% relative error
- Format Score: output has structured reasoning
- By problem type breakdown

Usage:
    python scripts/evaluate_lsreasoning_v2.py --model-path outputs/sft_lsreasoning/final --test-file data/lsreasoning_split/test.jsonl
"""

import argparse
import json
import re

import wandb
from unsloth import FastLanguageModel


SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, "
    "show your reasoning clearly, then give the final answer."
)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


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
        val = numbers[-1]
        try:
            if "/" in val:
                parts = val.split("/")
                return float(parts[0]) / float(parts[1])
            return float(val)
        except (ValueError, ZeroDivisionError):
            return None
    return None


def parse_answer(answer_str):
    try:
        answer_str = answer_str.strip()
        if "/" in answer_str:
            parts = answer_str.split("/")
            return float(parts[0]) / float(parts[1])
        return float(answer_str)
    except (ValueError, ZeroDivisionError):
        return None


def generate_response(model, tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
    )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def evaluate_model(
    model_path: str,
    test_file: str = "data/lsreasoning_split/test.jsonl",
    num_eval: int = None,
    verbose: bool = False,
    method_name: str = None,
):
    """Evaluate model on test set. Returns dict of metrics."""
    if method_name is None:
        if "grpo" in model_path and "dense" in model_path:
            method_name = "GRPO_dense"
        elif "grpo" in model_path and "sparse" in model_path:
            method_name = "GRPO_sparse"
        else:
            method_name = "SFT"

    print(f"Model: {model_path}")
    print(f"Test file: {test_file}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)

    test_data = load_jsonl(test_file)
    if num_eval and num_eval < len(test_data):
        test_data = test_data[:num_eval]
    total = len(test_data)
    print(f"Test samples: {total}")

    results = {
        "exact_match": 0,
        "close_match": 0,
        "has_answer_format": 0,
        "has_reasoning": 0,
    }
    by_problem_type = {}
    failures = []

    for i, item in enumerate(test_data):
        question = item["question"]
        true_answer = item["answer"]
        problem_type = item.get("problem", "unknown")

        response = generate_response(model, tokenizer, question)

        pred = extract_number(response)
        true_val = parse_answer(true_answer)

        exact = False
        close = False
        if pred is not None and true_val is not None:
            if abs(pred - true_val) < 1e-6:
                exact = True
                close = True
            elif true_val != 0 and abs(pred - true_val) / abs(true_val) < 0.01:
                close = True

        if exact:
            results["exact_match"] += 1
        if close:
            results["close_match"] += 1
        if "Answer:" in response or "answer:" in response:
            results["has_answer_format"] += 1
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        if len(lines) >= 3:
            results["has_reasoning"] += 1

        if problem_type not in by_problem_type:
            by_problem_type[problem_type] = [0, 0]
        by_problem_type[problem_type][1] += 1
        if exact:
            by_problem_type[problem_type][0] += 1

        if not exact and len(failures) < 10:
            failures.append({
                "question": question[:80],
                "expected": true_answer,
                "got": str(pred),
            })

        if verbose:
            status = "OK" if exact else "X"
            print(f"  [{i+1:3d}] {status} Q: {question[:60]}  A: {pred} (true: {true_answer})")
        elif (i + 1) % 50 == 0:
            print(f"  [{i+1}/{total}] processed...")

    # Report
    print(f"\n{'=' * 50}")
    print(f"  Model: {model_path}")
    print(f"  Test: {total} samples")
    print()
    print(f"  {'Metric':<25} {'Score':<10} {'Count'}")
    print(f"  {'_' * 45}")
    print(f"  {'Exact Match':<25} {results['exact_match']/total:>6.1%}     {results['exact_match']}/{total}")
    print(f"  {'Close Match (1%)':<25} {results['close_match']/total:>6.1%}     {results['close_match']}/{total}")
    print(f"  {'Has Answer: format':<25} {results['has_answer_format']/total:>6.1%}     {results['has_answer_format']}/{total}")
    print(f"  {'Has Reasoning':<25} {results['has_reasoning']/total:>6.1%}     {results['has_reasoning']}/{total}")
    print()

    print("  By problem type:")
    for ptype, (correct, total_t) in sorted(by_problem_type.items(), key=lambda x: -x[1][1]):
        if total_t > 0:
            print(f"    {ptype:<35} {correct}/{total_t} ({correct/total_t:.0%})")
    print()

    if failures[:5]:
        print("  Sample failures:")
        for f in failures[:5]:
            print(f"    Q: {f['question']}")
            print(f"    Expected: {f['expected']}, Got: {f['got']}")
            print()

    eval_results = {
        "exact_match": results["exact_match"] / total,
        "close_match": results["close_match"] / total,
        "format_score": results["has_answer_format"] / total,
        "reasoning_score": results["has_reasoning"] / total,
        "by_type": {k: v[0] / v[1] for k, v in by_problem_type.items() if v[1] > 0},
        "total": total,
    }

    # Log to wandb
    wandb.init(
        project="lsreasoning-sft-vs-grpo",
        name=f"eval_{method_name}",
        config={"method": method_name, "model_path": model_path, "num_eval": total},
        reinit=True,
    )
    wandb.log({
        f"eval/exact_match": eval_results["exact_match"],
        f"eval/close_match": eval_results["close_match"],
        f"eval/format_score": eval_results["format_score"],
        f"eval/reasoning_score": eval_results["reasoning_score"],
    })
    # Log per-problem-type as a table
    if by_problem_type:
        table = wandb.Table(columns=["problem_type", "accuracy", "correct", "total"])
        for ptype, (correct_count, total_t) in sorted(
            by_problem_type.items(), key=lambda x: -x[1][1]
        ):
            if total_t > 0:
                table.add_data(ptype, correct_count / total_t, correct_count, total_t)
        wandb.log({"eval/by_problem_type": table})
    wandb.finish()

    return eval_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--test-file", type=str,
                        default="data/lsreasoning_split/test.jsonl")
    parser.add_argument("--num-eval", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        test_file=args.test_file,
        num_eval=args.num_eval,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
