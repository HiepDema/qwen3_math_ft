"""Generate quadratic equation training data (local deterministic solver).

No external model needed — uses math solver to generate correct step-by-step solutions.
For the full data pipeline (VietJack + quadratic), use prepare_data.py instead.

Usage:
    python scripts/generate_quadratic_data.py
    python scripts/generate_quadratic_data.py --num-samples 300
"""

import argparse
import json
import math
import random
from fractions import Fraction
from pathlib import Path


def is_perfect_square(n):
    if n < 0:
        return False
    root = math.isqrt(n)
    return root * root == n


def format_quadratic(a, b, c):
    """Format ax² + bx + c = 0."""
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
    return "".join(parts) + " = 0"


def solve_quadratic_steps(a, b, c):
    """Generate step-by-step solution for ax² + bx + c = 0."""
    eq_str = format_quadratic(a, b, c)
    delta = b * b - 4 * a * c

    steps = [f"Ta có phương trình: {eq_str}"]
    steps.append(f"Với a = {a}, b = {b}, c = {c}")
    steps.append(f"Tính delta: Δ = b² - 4ac = ({b})² - 4·({a})·({c}) = {b*b} - ({4*a*c}) = {delta}")

    if delta < 0:
        steps.append("Vì Δ < 0 nên phương trình vô nghiệm.")
        answer = "Phương trình vô nghiệm"
    elif delta == 0:
        x = Fraction(-b, 2 * a)
        steps.append("Vì Δ = 0 nên phương trình có nghiệm kép:")
        steps.append(f"x = -b/(2a) = -({b})/(2·{a}) = {-b}/{2*a} = {x}")
        answer = f"x = {x}"
    else:
        sqrt_delta = math.isqrt(delta) if is_perfect_square(delta) else None
        if sqrt_delta is not None:
            steps.append("Vì Δ > 0 nên phương trình có hai nghiệm phân biệt:")
            steps.append(f"√Δ = √{delta} = {sqrt_delta}")
            x1 = Fraction(-b + sqrt_delta, 2 * a)
            x2 = Fraction(-b - sqrt_delta, 2 * a)
            steps.append(f"x₁ = (-b + √Δ)/(2a) = ({-b} + {sqrt_delta})/{2*a} = {x1}")
            steps.append(f"x₂ = (-b - √Δ)/(2a) = ({-b} - {sqrt_delta})/{2*a} = {x2}")
            answer = f"x₁ = {x1}, x₂ = {x2}"
        else:
            steps.append("Vì Δ > 0 nên phương trình có hai nghiệm phân biệt:")
            steps.append(f"x₁ = ({-b} + √{delta})/{2*a}")
            steps.append(f"x₂ = ({-b} - √{delta})/{2*a}")
            answer = f"x₁ = ({-b} + √{delta})/{2*a}, x₂ = ({-b} - √{delta})/{2*a}"

    return "\n".join(steps) + f"\nĐáp án: {answer}"


def generate_data(num_samples=300, seed=42):
    """Generate quadratic equation SFT data."""
    random.seed(seed)
    samples = []
    seen = set()

    while len(samples) < num_samples:
        a = random.choice([i for i in range(-8, 9) if i != 0])
        b = random.randint(-20, 20)
        c = random.randint(-30, 30)
        key = (a, b, c)
        if key in seen:
            continue
        seen.add(key)

        eq_str = format_quadratic(a, b, c)
        solution = solve_quadratic_steps(a, b, c)
        samples.append({
            "instruction": f"Giải phương trình bậc hai: {eq_str}",
            "output": solution,
        })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate quadratic equation data")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default="data/raw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_samples} quadratic equation samples...")
    samples = generate_data(args.num_samples, seed=args.seed)

    sft_path = output_dir / "sft_quadratic_equations.jsonl"
    with open(sft_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Saved {len(samples)} samples to {sft_path}")

    # Stats
    stats = {"no_solution": 0, "one_solution": 0, "two_solutions": 0}
    for s in samples:
        if "vô nghiệm" in s["output"]:
            stats["no_solution"] += 1
        elif "nghiệm kép" in s["output"]:
            stats["one_solution"] += 1
        else:
            stats["two_solutions"] += 1

    print(f"\nStats: {stats}")
    print(f"\nSample:")
    print(f"  Q: {samples[0]['instruction']}")
    print(f"  A: {samples[0]['output']}")


if __name__ == "__main__":
    main()
