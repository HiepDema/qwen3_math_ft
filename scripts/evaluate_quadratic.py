"""Evaluate fine-tuned model on quadratic equation solving.

Metrics:
- Exact Match: đáp án cuối cùng đúng/sai
- Format Compliance: output đúng format yêu cầu
- Delta Correct: tính delta đúng
- Roots Correct: tìm nghiệm đúng
- All Steps Correct: toàn bộ lời giải đúng

Usage:
    python scripts/evaluate_quadratic.py
    python scripts/evaluate_quadratic.py --model-path outputs/grpo_quadratic_eq/final --num-tests 50
"""

import argparse
import math
import random
import re
from fractions import Fraction

from unsloth import FastLanguageModel


SYSTEM_PROMPT = "Bạn là trợ lý toán học. Hãy giải phương trình bậc hai theo từng bước, sử dụng công thức delta."


def is_perfect_square(n):
    if n < 0:
        return False
    root = math.isqrt(n)
    return root * root == n


def generate_test_equations(num_tests: int, seed: int = 999):
    """Generate test equations (different seed from training data)."""
    random.seed(seed)
    tests = []
    seen = set()

    while len(tests) < num_tests:
        a = random.choice([i for i in range(-8, 9) if i != 0])
        b = random.randint(-20, 20)
        c = random.randint(-30, 30)
        key = (a, b, c)
        if key in seen:
            continue
        seen.add(key)

        parts = []
        if a == 1:
            parts.append("x²")
        elif a == -1:
            parts.append("-x²")
        else:
            parts.append(f"{a}x²")
        if b > 0:
            parts.append(f" + {b}x")
        elif b < 0:
            parts.append(f" - {abs(b)}x")
        if c > 0:
            parts.append(f" + {c}")
        elif c < 0:
            parts.append(f" - {abs(c)}")

        eq_str = "".join(parts) + " = 0"

        delta = b * b - 4 * a * c
        if delta < 0:
            answer_type = "no_solution"
            correct_answer = "Phương trình vô nghiệm"
        elif delta == 0:
            x = Fraction(-b, 2 * a)
            answer_type = "one_solution"
            correct_answer = f"x = {x}"
        else:
            sqrt_d = math.isqrt(delta) if is_perfect_square(delta) else None
            if sqrt_d is not None:
                x1 = Fraction(-b + sqrt_d, 2 * a)
                x2 = Fraction(-b - sqrt_d, 2 * a)
                answer_type = "two_solutions"
                correct_answer = f"x₁ = {x1}, x₂ = {x2}"
            else:
                answer_type = "two_solutions_irrational"
                correct_answer = f"x₁ = ({-b} + √{delta})/{2*a}, x₂ = ({-b} - √{delta})/{2*a}"

        tests.append({
            "a": a, "b": b, "c": c,
            "equation": eq_str,
            "instruction": f"Giải phương trình bậc hai: {eq_str}",
            "delta": delta,
            "answer_type": answer_type,
            "correct_answer": correct_answer,
        })

    return tests


def extract_answer(text: str) -> str | None:
    """Extract final answer from model output."""
    match = re.search(r"Đáp án[:\s]*(.+?)(?:\n|$)", text)
    if match:
        return match.group(1).strip()
    if "vô nghiệm" in text.lower():
        return "Phương trình vô nghiệm"
    return None


def check_delta(output: str, expected_delta: int) -> bool:
    """Check if delta calculation is correct."""
    patterns = [
        rf"Δ\s*=\s*.*?=\s*{expected_delta}",
        rf"delta\s*=\s*.*?=\s*{expected_delta}",
        rf"=\s*{expected_delta}\s*$",
    ]
    for p in patterns:
        if re.search(p, output, re.IGNORECASE | re.MULTILINE):
            return True
    return str(expected_delta) in output


def check_answer_correct(output: str, test: dict) -> bool:
    """Check if the final answer is correct."""
    if test["answer_type"] == "no_solution":
        return "vô nghiệm" in output.lower()

    elif test["answer_type"] == "one_solution":
        delta = test["delta"]
        a, b = test["a"], test["b"]
        x = Fraction(-b, 2 * a)
        return f"= {x}" in output

    elif test["answer_type"] == "two_solutions":
        a, b, c = test["a"], test["b"], test["c"]
        delta = test["delta"]
        sqrt_d = math.isqrt(delta)
        x1 = Fraction(-b + sqrt_d, 2 * a)
        x2 = Fraction(-b - sqrt_d, 2 * a)
        found_x1 = f"= {x1}" in output or f"={x1}" in output
        found_x2 = f"= {x2}" in output or f"={x2}" in output
        return found_x1 and found_x2

    else:
        return str(test["delta"]) in output


def check_format(output: str) -> dict:
    """Check if output follows the required format."""
    has_equation = "phương trình" in output.lower() or "Ta có" in output
    has_delta = "Δ" in output or "delta" in output.lower()
    has_answer = "Đáp án" in output or "Vậy" in output
    lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
    has_enough_steps = len(lines) >= 5

    return {
        "has_equation": has_equation,
        "has_delta": has_delta,
        "has_answer": has_answer,
        "has_enough_steps": has_enough_steps,
        "all_ok": has_equation and has_delta and has_answer and has_enough_steps,
    }


def generate_response(model, tokenizer, question: str) -> str:
    """Generate model response."""
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
        max_new_tokens=512,
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
    parser.add_argument("--model-path", type=str, default="outputs/grpo_quadratic_eq/final")
    parser.add_argument("--num-tests", type=int, default=50)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded!\n")

    test_set = generate_test_equations(args.num_tests, seed=args.seed)
    print(f"Running evaluation on {len(test_set)} test equations...\n")

    results = {
        "exact_match": 0,
        "format_ok": 0,
        "delta_correct": 0,
        "answer_correct": 0,
        "all_steps_ok": 0,
    }
    by_type = {"no_solution": [0, 0], "one_solution": [0, 0], "two_solutions": [0, 0], "two_solutions_irrational": [0, 0]}
    failures = []

    for i, test in enumerate(test_set):
        prediction = generate_response(model, tokenizer, test["instruction"])

        delta_ok = check_delta(prediction, test["delta"])
        answer_ok = check_answer_correct(prediction, test)
        fmt = check_format(prediction)

        if delta_ok:
            results["delta_correct"] += 1
        if answer_ok:
            results["answer_correct"] += 1
            by_type[test["answer_type"]][0] += 1
        by_type[test["answer_type"]][1] += 1
        if fmt["all_ok"]:
            results["format_ok"] += 1
        if delta_ok and answer_ok:
            results["exact_match"] += 1
        if delta_ok and answer_ok and fmt["all_ok"]:
            results["all_steps_ok"] += 1

        status = "✓" if answer_ok else "✗"
        if args.verbose or not answer_ok:
            print(f"  [{i+1:2d}] {status} {test['instruction']}")
            if not answer_ok:
                failures.append({
                    "equation": test["instruction"],
                    "expected": test["correct_answer"],
                    "full_output": prediction[:200],
                })
        elif (i + 1) % 10 == 0:
            print(f"  [{i+1:2d}/{len(test_set)}] processed...")

    total = len(test_set)
    print("\n" + "═" * 60)
    print("EVALUATION REPORT - Quadratic Equations")
    print("═" * 60)
    print(f"\n  Model: {args.model_path}")
    print(f"  Test size: {total}")
    print()
    print(f"  {'Metric':<25} {'Score':<10} {'Count'}")
    print(f"  {'─' * 50}")
    print(f"  {'Exact Match':<25} {results['exact_match']/total:>6.1%}     {results['exact_match']}/{total}")
    print(f"  {'Answer Correct':<25} {results['answer_correct']/total:>6.1%}     {results['answer_correct']}/{total}")
    print(f"  {'Delta Correct':<25} {results['delta_correct']/total:>6.1%}     {results['delta_correct']}/{total}")
    print(f"  {'Format Compliance':<25} {results['format_ok']/total:>6.1%}     {results['format_ok']}/{total}")
    print(f"  {'All Steps Correct':<25} {results['all_steps_ok']/total:>6.1%}     {results['all_steps_ok']}/{total}")
    print()

    print("  By equation type:")
    print(f"  {'─' * 50}")
    for etype, (correct, total_t) in by_type.items():
        if total_t > 0:
            print(f"    {etype:<30} {correct}/{total_t} ({correct/total_t:.0%})")
    print()

    if failures:
        print(f"  Failed cases ({len(failures)}):")
        print(f"  {'─' * 50}")
        for f in failures[:5]:
            print(f"    {f['equation']}")
            print(f"      Expected: {f['expected']}")
            print()

    print("═" * 60)
    pass_rate = results["answer_correct"] / total
    if pass_rate >= 0.7:
        print(f"  Model PASSED (Answer Correct = {pass_rate:.0%} >= 70%)")
    else:
        print(f"  Model NEEDS IMPROVEMENT (Answer Correct = {pass_rate:.0%} < 70%)")
    print("═" * 60)


if __name__ == "__main__":
    main()
