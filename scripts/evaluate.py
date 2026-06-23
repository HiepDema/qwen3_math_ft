"""Evaluate fine-tuned model on linear equation solving.

Metrics:
- Exact Match: đáp án cuối cùng đúng/sai
- Format Compliance: output đúng format yêu cầu
- Step Correctness: các bước trung gian đúng

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model-path outputs/sft_linear_eq/final --num-tests 50
"""

import argparse
import random
import re
from fractions import Fraction

from unsloth import FastLanguageModel


SYSTEM_PROMPT = "Bạn là trợ lý toán học. Hãy giải phương trình bậc nhất theo từng bước."


def generate_test_equations(num_tests: int, seed: int = 123):
    """Generate test equations (different seed from training data)."""
    random.seed(seed)
    tests = []
    seen = set()

    while len(tests) < num_tests:
        a = random.choice([i for i in range(-10, 11) if i != 0])
        b = random.randint(-20, 20)
        c = random.randint(-30, 30)
        key = (a, b, c)
        if key in seen:
            continue
        seen.add(key)

        # Format equation
        if a == 1:
            left = "x"
        elif a == -1:
            left = "-x"
        else:
            left = f"{a}x"

        if b > 0:
            left += f" + {b}"
        elif b < 0:
            left += f" - {abs(b)}"

        eq_str = f"{left} = {c}"

        # Compute correct answer
        rhs = c - b
        if a == 1:
            x = rhs
        elif a == -1:
            x = -rhs
        elif rhs % a == 0:
            x = rhs // a
        else:
            frac = Fraction(rhs, a)
            x = f"{frac.numerator}/{frac.denominator}"

        tests.append({
            "a": a,
            "b": b,
            "c": c,
            "equation": eq_str,
            "instruction": f"Giải phương trình: {eq_str}",
            "correct_answer": str(x),
            "correct_rhs": rhs,
        })

    return tests


def extract_answer(text: str) -> str | None:
    """Extract final answer from model output."""
    # Try "Đáp án: x = ..."
    match = re.search(r"Đáp án[:\s]*x\s*=\s*([^\n]+)", text)
    if match:
        return match.group(1).strip()

    # Try last "x = ..." line
    matches = re.findall(r"x\s*=\s*([^\n]+)", text)
    if matches:
        return matches[-1].strip()

    return None


def check_format(output: str) -> dict:
    """Check if output follows the required format."""
    has_ta_co = "Ta có:" in output or "Ta có:\n" in output
    has_dap_an = "Đáp án:" in output
    lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
    has_enough_steps = len(lines) >= 4

    return {
        "has_ta_co": has_ta_co,
        "has_dap_an": has_dap_an,
        "has_enough_steps": has_enough_steps,
        "all_ok": has_ta_co and has_dap_an and has_enough_steps,
    }


def check_steps(output: str, a: int, b: int, c: int) -> dict:
    """Check intermediate steps are correct."""
    rhs = c - b

    # Check step: ax = rhs (after moving b)
    step1_patterns = [
        f"{a}x = {rhs}",
        f"{a}x={rhs}",
    ]
    if a == 1:
        step1_patterns.extend([f"x = {rhs}", f"x={rhs}"])
    elif a == -1:
        step1_patterns.extend([f"-x = {rhs}", f"-x={rhs}"])

    step1_correct = any(p in output for p in step1_patterns)

    # Check final x = answer
    if a == 1:
        x = rhs
    elif a == -1:
        x = -rhs
    elif rhs % a == 0:
        x = rhs // a
    else:
        frac = Fraction(rhs, a)
        x = f"{frac.numerator}/{frac.denominator}"

    step2_patterns = [f"x = {x}", f"x={x}"]
    step2_correct = any(p in output for p in step2_patterns)

    return {
        "step1_correct": step1_correct,
        "step2_correct": step2_correct,
        "all_ok": step1_correct and step2_correct,
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
    parser.add_argument("--model-path", type=str, default="outputs/sft_linear_eq/final")
    parser.add_argument("--num-tests", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--verbose", action="store_true", help="Print each prediction")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from: {args.model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded!\n")

    # Generate test set
    test_set = generate_test_equations(args.num_tests, seed=args.seed)
    print(f"Running evaluation on {len(test_set)} test equations...\n")

    # Evaluate
    results = {
        "exact_match": 0,
        "format_ok": 0,
        "step1_ok": 0,
        "step2_ok": 0,
        "steps_all_ok": 0,
    }
    failures = []

    for i, test in enumerate(test_set):
        prediction = generate_response(model, tokenizer, test["instruction"])

        # Exact Match
        pred_answer = extract_answer(prediction)
        is_correct = pred_answer == test["correct_answer"]
        if is_correct:
            results["exact_match"] += 1

        # Format Check
        fmt = check_format(prediction)
        if fmt["all_ok"]:
            results["format_ok"] += 1

        # Step Check
        steps = check_steps(prediction, test["a"], test["b"], test["c"])
        if steps["step1_correct"]:
            results["step1_ok"] += 1
        if steps["step2_correct"]:
            results["step2_ok"] += 1
        if steps["all_ok"]:
            results["steps_all_ok"] += 1

        # Log
        status = "✓" if is_correct else "✗"
        if args.verbose or not is_correct:
            print(f"  [{i+1:2d}] {status} {test['instruction']}")
            if args.verbose:
                print(f"       Pred: {pred_answer} | True: {test['correct_answer']}")
            if not is_correct:
                failures.append({
                    "equation": test["instruction"],
                    "expected": test["correct_answer"],
                    "got": pred_answer,
                    "full_output": prediction,
                })
        elif (i + 1) % 10 == 0:
            print(f"  [{i+1:2d}/{len(test_set)}] processed...")

    # Print report
    total = len(test_set)
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"\n  Model: {args.model_path}")
    print(f"  Test size: {total}")
    print()
    print(f"  {'Metric':<25} {'Score':<10} {'Count'}")
    print(f"  {'─' * 50}")
    print(f"  {'Exact Match':<25} {results['exact_match']/total:>6.1%}     {results['exact_match']}/{total}")
    print(f"  {'Format Compliance':<25} {results['format_ok']/total:>6.1%}     {results['format_ok']}/{total}")
    print(f"  {'Step 1 (chuyển vế)':<25} {results['step1_ok']/total:>6.1%}     {results['step1_ok']}/{total}")
    print(f"  {'Step 2 (tìm x)':<25} {results['step2_ok']/total:>6.1%}     {results['step2_ok']}/{total}")
    print(f"  {'All Steps Correct':<25} {results['steps_all_ok']/total:>6.1%}     {results['steps_all_ok']}/{total}")
    print()

    # Show failures
    if failures:
        print(f"  Failed cases ({len(failures)}):")
        print(f"  {'─' * 50}")
        for f in failures[:10]:
            print(f"    {f['equation']}")
            print(f"      Expected: x = {f['expected']}, Got: {f['got']}")
            if args.verbose:
                print(f"      Output: {f['full_output'][:100]}...")
            print()

    # Summary
    print("=" * 60)
    if results["exact_match"] / total >= 0.8:
        print("  Model PASSED (Exact Match >= 80%)")
    else:
        print("  Model NEEDS IMPROVEMENT (Exact Match < 80%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
