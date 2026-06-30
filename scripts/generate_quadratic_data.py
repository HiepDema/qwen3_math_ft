"""Generate training data for quadratic equation solving using Qwen3-VL-30B-A3B-Instruct.

Uses vLLM to serve the teacher model on A10 GPU and generate ~300 high-quality
instruction-response pairs for solving quadratic equations (ax² + bx + c = 0).

Usage:
    # Generate data using Qwen3-VL-30B-A3B-Instruct (requires A10 GPU)
    python scripts/generate_quadratic_data.py --mode teacher

    # Generate data locally (no GPU needed, deterministic)
    python scripts/generate_quadratic_data.py --mode local

    # Generate mixed (teacher + local supplement)
    python scripts/generate_quadratic_data.py --mode mixed --num-samples 300
"""

import argparse
import json
import math
import random
from fractions import Fraction
from pathlib import Path


def generate_quadratic_coefficients(difficulty="mixed"):
    """Generate random quadratic equation coefficients ax² + bx + c = 0."""
    if difficulty == "easy":
        a = random.choice([1, -1, 2, -2])
        b = random.randint(-10, 10)
        c = random.randint(-10, 10)
    elif difficulty == "medium":
        a = random.choice([i for i in range(-5, 6) if i != 0])
        b = random.randint(-15, 15)
        c = random.randint(-20, 20)
    else:
        a = random.choice([i for i in range(-8, 9) if i != 0])
        b = random.randint(-20, 20)
        c = random.randint(-30, 30)
    return a, b, c


def format_quadratic(a, b, c):
    """Format quadratic equation as string: ax² + bx + c = 0."""
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
    elif b == 0:
        pass

    if c > 0:
        parts.append(f" + {c}")
    elif c < 0:
        parts.append(f" - {abs(c)}")
    elif c == 0:
        pass

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
        steps.append(f"Vì Δ = 0 nên phương trình có nghiệm kép:")
        steps.append(f"x = -b/(2a) = -({b})/(2·{a}) = {-b}/{2*a} = {x}")
        answer = f"x = {x}"
    else:
        sqrt_delta = math.isqrt(delta) if is_perfect_square(delta) else None

        if sqrt_delta is not None:
            steps.append(f"Vì Δ > 0 nên phương trình có hai nghiệm phân biệt:")
            steps.append(f"√Δ = √{delta} = {sqrt_delta}")
            x1 = Fraction(-b + sqrt_delta, 2 * a)
            x2 = Fraction(-b - sqrt_delta, 2 * a)
            steps.append(f"x₁ = (-b + √Δ)/(2a) = (-({b}) + {sqrt_delta})/(2·{a}) = {-b + sqrt_delta}/{2*a} = {x1}")
            steps.append(f"x₂ = (-b - √Δ)/(2a) = (-({b}) - {sqrt_delta})/(2·{a}) = {-b - sqrt_delta}/{2*a} = {x2}")
            answer = f"x₁ = {x1}, x₂ = {x2}"
        else:
            steps.append(f"Vì Δ > 0 nên phương trình có hai nghiệm phân biệt:")
            steps.append(f"x₁ = (-b + √Δ)/(2a) = (-({b}) + √{delta})/(2·{a}) = ({-b} + √{delta})/{2*a}")
            steps.append(f"x₂ = (-b - √Δ)/(2a) = (-({b}) - √{delta})/(2·{a}) = ({-b} - √{delta})/{2*a}")
            answer = f"x₁ = ({-b} + √{delta})/{2*a}, x₂ = ({-b} - √{delta})/{2*a}"

    solution = "\n".join(steps)
    return solution + f"\nĐáp án: {answer}"


def is_perfect_square(n):
    """Check if n is a perfect square."""
    if n < 0:
        return False
    root = math.isqrt(n)
    return root * root == n


def generate_sft_sample(a, b, c):
    """Generate an SFT training sample."""
    instruction = f"Giải phương trình bậc hai: {format_quadratic(a, b, c)}"
    output = solve_quadratic_steps(a, b, c)
    return {"instruction": instruction, "output": output}


def generate_local_data(num_samples=300, seed=42):
    """Generate data locally with deterministic solutions."""
    random.seed(seed)
    samples = []
    seen = set()

    difficulties = ["easy"] * (num_samples // 3) + ["medium"] * (num_samples // 3) + ["hard"] * (num_samples - 2 * (num_samples // 3))
    random.shuffle(difficulties)

    for diff in difficulties:
        attempts = 0
        while attempts < 100:
            a, b, c = generate_quadratic_coefficients(diff)
            key = (a, b, c)
            if key not in seen:
                seen.add(key)
                samples.append(generate_sft_sample(a, b, c))
                break
            attempts += 1

    return samples


def generate_with_teacher(num_samples=300, model_name="Qwen/Qwen3-VL-30B-A3B-Instruct"):
    """Generate data using Qwen3-VL-30B-A3B-Instruct as teacher model via vLLM."""
    from vllm import LLM, SamplingParams

    print(f"Loading teacher model: {model_name}")
    print("This requires A10 GPU (24GB VRAM)...")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
        repetition_penalty=1.1,
    )

    samples = []
    batch_size = 20
    num_batches = (num_samples + batch_size - 1) // batch_size

    system_prompt = (
        "Bạn là giáo viên toán giỏi. Hãy giải phương trình bậc hai theo ĐÚNG format sau (không thêm bớt):\n\n"
        "Ta có phương trình: [phương trình]\n"
        "Với a = [giá trị], b = [giá trị], c = [giá trị]\n"
        "Tính delta: Δ = b² - 4ac = ([b])² - 4·([a])·([c]) = [b²] - ([4ac]) = [delta]\n"
        "Vì Δ [> 0 / = 0 / < 0] nên phương trình [có hai nghiệm phân biệt / có nghiệm kép / vô nghiệm]:\n"
        "[Nếu Δ > 0: √Δ = √[delta] = [giá trị]]\n"
        "[Nếu Δ > 0: x₁ = (-b + √Δ)/(2a) = ... = [kết quả]]\n"
        "[Nếu Δ > 0: x₂ = (-b - √Δ)/(2a) = ... = [kết quả]]\n"
        "[Nếu Δ = 0: x = -b/(2a) = ... = [kết quả]]\n"
        "Đáp án: [x₁ = ..., x₂ = ... / x = ... / Phương trình vô nghiệm]\n\n"
        "QUAN TRỌNG: Luôn bắt đầu bằng 'Ta có phương trình:' và kết thúc bằng 'Đáp án:'. "
        "Tính toán phải chính xác."
    )

    for batch_idx in range(num_batches):
        prompts = []
        batch_metadata = []
        current_batch_size = min(batch_size, num_samples - len(samples))

        for _ in range(current_batch_size):
            a, b, c = generate_quadratic_coefficients("mixed")
            eq_str = format_quadratic(a, b, c)
            user_msg = f"Giải phương trình bậc hai: {eq_str}"

            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            prompts.append(prompt)
            batch_metadata.append({
                "a": a, "b": b, "c": c,
                "instruction": user_msg,
            })

        outputs = llm.generate(prompts, sampling_params)

        for meta, output in zip(batch_metadata, outputs):
            response = output.outputs[0].text.strip()
            a, b, c = meta["a"], meta["b"], meta["c"]
            sample = {"instruction": meta["instruction"], "output": response}

            if is_valid_format(response, a, b, c):
                samples.append(sample)
            else:
                samples.append(generate_sft_sample(a, b, c))

        print(f"  Batch {batch_idx + 1}/{num_batches}: {len(samples)} valid samples so far")

    print(f"Teacher generated {len(samples)} samples (with fallback corrections)")
    return samples


def is_valid_format(response, a, b, c):
    """Check if teacher response follows the required SFT format and is correct."""
    if not response:
        return False

    delta = b * b - 4 * a * c

    has_header = "Ta có phương trình" in response or "Ta có" in response
    has_coefficients = "a =" in response and "b =" in response
    has_delta = "Δ" in response or "delta" in response.lower()
    has_answer = "Đáp án" in response
    has_enough_lines = len(response.strip().split("\n")) >= 4

    if not (has_header and has_coefficients and has_delta and has_answer and has_enough_lines):
        return False

    if delta < 0 and "vô nghiệm" not in response.lower():
        return False
    if delta == 0 and "nghiệm kép" not in response.lower():
        return False
    if delta > 0 and "hai nghiệm" not in response.lower():
        return False

    if str(delta) not in response:
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Generate quadratic equation training data")
    parser.add_argument("--mode", choices=["teacher", "local", "mixed"], default="local",
                        help="Data generation mode")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct",
                        help="Teacher model name")
    parser.add_argument("--num-samples", type=int, default=300, help="Number of SFT samples")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    if args.mode == "teacher":
        samples = generate_with_teacher(args.num_samples, args.model_name)
        if len(samples) < args.num_samples:
            print(f"Supplementing with local data ({args.num_samples - len(samples)} more)...")
            local = generate_local_data(args.num_samples - len(samples), seed=args.seed + 1)
            samples.extend(local)

    elif args.mode == "local":
        print("Generating data locally (deterministic solver)...")
        samples = generate_local_data(args.num_samples, seed=args.seed)

    elif args.mode == "mixed":
        num_teacher = args.num_samples * 2 // 3
        num_local = args.num_samples - num_teacher
        print(f"Mixed mode: {num_teacher} from teacher, {num_local} local")
        try:
            teacher_samples = generate_with_teacher(num_teacher, args.model_name)
        except Exception as e:
            print(f"Teacher generation failed: {e}")
            print("Falling back to all local generation...")
            teacher_samples = []
        local_samples = generate_local_data(
            args.num_samples - len(teacher_samples), seed=args.seed
        )
        samples = teacher_samples + local_samples

    sft_path = output_dir / "sft_quadratic_equations.jsonl"
    with open(sft_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(samples)} SFT samples to {sft_path}")

    print("\n" + "=" * 60)
    print("SAMPLE DATA:")
    print("=" * 60)
    for sample in samples[:5]:
        print(f"\nInstruction: {sample['instruction']}")
        print(f"Output:\n{sample['output']}\n---")

    stats = {"total": len(samples), "no_solution": 0, "one_solution": 0, "two_solutions": 0}
    for s in samples:
        if "vô nghiệm" in s["output"]:
            stats["no_solution"] += 1
        elif "nghiệm kép" in s["output"]:
            stats["one_solution"] += 1
        else:
            stats["two_solutions"] += 1

    print(f"\nStats: {stats}")


if __name__ == "__main__":
    main()
