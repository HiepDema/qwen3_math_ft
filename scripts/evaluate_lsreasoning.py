"""Evaluate model on LSReasoning-15000 test split.

Metrics:
- Exact Match: answer is exactly correct
- Close Match: answer within 1% relative error
- Format Score: output has structured reasoning
- Average reward (dense): overall quality score

Usage:
    python scripts/evaluate_lsreasoning.py --model-path outputs/sft_lsreasoning/final
    python scripts/evaluate_lsreasoning.py --model-path outputs/grpo_lsreasoning_dense/final
"""

import argparse
import re

from datasets import load_dataset
from unsloth import FastLanguageModel


SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, "
    "show your reasoning clearly, then give the final answer."
)


def extract_number(text):
    """Extract numeric answer from response."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset-name", type=str,
                        default="DataMuncher-Labs/LSReasoning-15000")
    parser.add_argument("--num-tests", type=int, default=200)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded!\n")

    # Use last N samples as test (not seen during training)
    print(f"Loading test data from {args.dataset_name}...")
    full_dataset = load_dataset(args.dataset_name, split="train")
    test_dataset = full_dataset.shuffle(seed=args.seed).select(
        range(len(full_dataset) - args.num_tests, len(full_dataset))
    )
    print(f"Test size: {len(test_dataset)}\n")

    # Evaluate
    results = {
        "exact_match": 0,
        "close_match": 0,
        "has_answer_format": 0,
        "has_reasoning": 0,
        "no_output": 0,
    }
    by_problem_type = {}
    failures = []

    for i, item in enumerate(test_dataset):
        question = item["question"]
        true_answer = item["answer"]
        problem_type = item["problem"]

        response = generate_response(model, tokenizer, question)

        # Check exact match
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

        # Format
        if "Answer:" in response or "answer:" in response:
            results["has_answer_format"] += 1
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        if len(lines) >= 3:
            results["has_reasoning"] += 1
        if not response:
            results["no_output"] += 1

        # By type
        if problem_type not in by_problem_type:
            by_problem_type[problem_type] = [0, 0]
        by_problem_type[problem_type][1] += 1
        if exact:
            by_problem_type[problem_type][0] += 1

        # Log
        status = "✓" if exact else "✗"
        if args.verbose or not exact:
            if len(failures) < 20:
                failures.append({
                    "question": question,
                    "expected": true_answer,
                    "got": str(pred),
                })
        if args.verbose:
            print(f"  [{i+1:3d}] {status} Q: {question[:60]}  A: {pred} (true: {true_answer})")
        elif (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(test_dataset)}] processed...")

    # Report
    total = len(test_dataset)
    print("\n" + "═" * 60)
    print("EVALUATION REPORT - LSReasoning-15000")
    print("═" * 60)
    print(f"\n  Model: {args.model_path}")
    print(f"  Test size: {total}")
    print()
    print(f"  {'Metric':<25} {'Score':<10} {'Count'}")
    print(f"  {'─' * 50}")
    print(f"  {'Exact Match':<25} {results['exact_match']/total:>6.1%}     {results['exact_match']}/{total}")
    print(f"  {'Close Match (1%)':<25} {results['close_match']/total:>6.1%}     {results['close_match']}/{total}")
    print(f"  {'Has Answer: format':<25} {results['has_answer_format']/total:>6.1%}     {results['has_answer_format']}/{total}")
    print(f"  {'Has Reasoning':<25} {results['has_reasoning']/total:>6.1%}     {results['has_reasoning']}/{total}")
    print()

    print("  By problem type:")
    print(f"  {'─' * 50}")
    for ptype, (correct, total_t) in sorted(by_problem_type.items(), key=lambda x: -x[1][1]):
        if total_t > 0:
            print(f"    {ptype:<40} {correct}/{total_t} ({correct/total_t:.0%})")
    print()

    if failures[:5]:
        print("  Sample failures:")
        print(f"  {'─' * 50}")
        for f in failures[:5]:
            print(f"    Q: {f['question'][:60]}")
            print(f"    Expected: {f['expected']}, Got: {f['got']}")
            print()

    print("═" * 60)
    rate = results["exact_match"] / total
    print(f"  Exact Match: {rate:.1%}")
    print("═" * 60)

    return results


if __name__ == "__main__":
    main()
