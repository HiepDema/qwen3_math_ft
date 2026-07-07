"""Generate 1k SFT data using Qwen3-VL-8B-Thinking-FP8 via vLLM.

Uses the thinking model to generate high-quality chain-of-thought solutions
for math problems (arithmetic, equations, fractions, word problems).

Usage:
    python scripts/generate_data_thinking.py
    python scripts/generate_data_thinking.py --num-samples 1000 --batch-size 64
"""

import argparse
import json
import random
from pathlib import Path

from vllm import LLM, SamplingParams


def generate_prompts(num_samples=1000, seed=42):
    """Generate diverse math problem prompts."""
    random.seed(seed)
    prompts = []

    for _ in range(num_samples):
        problem_type = random.choice([
            "linear_eq", "two_step_eq", "arithmetic", "fraction",
            "word_problem", "negative_ops", "multi_step",
        ])

        if problem_type == "linear_eq":
            a = random.choice([i for i in range(2, 10)])
            x = random.randint(-15, 15)
            b = random.randint(-20, 20)
            rhs = a * x + b
            prompts.append({
                "question": f"Solve for x: {a}x + {b} = {rhs}",
                "answer": str(x),
                "type": "linear_eq",
            })

        elif problem_type == "two_step_eq":
            a = random.choice([i for i in range(2, 8)])
            x = random.randint(-10, 10)
            b = random.randint(-15, 15)
            c = random.randint(-20, 20)
            rhs = a * x + b + c
            sign_b = f"+ {b}" if b >= 0 else f"- {abs(b)}"
            sign_c = f"+ {c}" if c >= 0 else f"- {abs(c)}"
            prompts.append({
                "question": f"Solve for x: {a}x {sign_b} {sign_c} = {rhs}",
                "answer": str(x),
                "type": "two_step_eq",
            })

        elif problem_type == "arithmetic":
            op = random.choice(["+", "-", "*"])
            a = random.randint(-50, 50)
            b = random.randint(-50, 50)
            if op == "+":
                ans = a + b
            elif op == "-":
                ans = a - b
            else:
                a = random.randint(-12, 12)
                b = random.randint(-12, 12)
                ans = a * b
            prompts.append({
                "question": f"What is {a} {op} {b}?",
                "answer": str(ans),
                "type": "arithmetic",
            })

        elif problem_type == "fraction":
            ops = random.choice(["simplify", "compute"])
            if ops == "simplify":
                factor = random.randint(2, 6)
                num = random.randint(1, 8) * factor
                den = random.randint(2, 8) * factor
                from math import gcd
                g = gcd(num, den)
                ans = f"{num//g}/{den//g}" if den // g != 1 else str(num // g)
                sign = random.choice(["", "-"])
                prompts.append({
                    "question": f"Simplify {sign}{num}/{den}",
                    "answer": f"{sign}{ans}",
                    "type": "fraction",
                })
            else:
                a = random.randint(1, 9)
                b = random.randint(2, 9)
                c = random.randint(1, 9)
                d = random.randint(2, 9)
                op = random.choice(["+", "-", "*"])
                if op == "+":
                    num = a * d + c * b
                    den = b * d
                elif op == "-":
                    num = a * d - c * b
                    den = b * d
                else:
                    num = a * c
                    den = b * d
                from math import gcd
                g = gcd(abs(num), den)
                ans = f"{num//g}/{den//g}" if den // g != 1 else str(num // g)
                prompts.append({
                    "question": f"Compute {a}/{b} {op} {c}/{d}",
                    "answer": ans,
                    "type": "fraction",
                })

        elif problem_type == "word_problem":
            template = random.choice([
                "cost", "distance", "items", "age",
            ])
            if template == "cost":
                price = random.randint(2, 15)
                qty = random.randint(3, 20)
                total = price * qty
                prompts.append({
                    "question": f"If each item costs {price} dollars and the total cost is {total} dollars, how many items were bought?",
                    "answer": str(qty),
                    "type": "word_problem",
                })
            elif template == "distance":
                speed = random.choice([30, 40, 50, 60, 70, 80])
                time = random.randint(2, 8)
                dist = speed * time
                prompts.append({
                    "question": f"A car travels at {speed} mph for {time} hours. What is the total distance traveled?",
                    "answer": str(dist),
                    "type": "word_problem",
                })
            elif template == "items":
                a = random.randint(10, 50)
                b = random.randint(5, 30)
                prompts.append({
                    "question": f"A store had {a + b} items. After selling {b} items, how many remain?",
                    "answer": str(a),
                    "type": "word_problem",
                })
            else:  # age
                age1 = random.randint(5, 30)
                diff = random.randint(2, 10)
                prompts.append({
                    "question": f"John is {diff} years older than Mary. If Mary is {age1} years old, how old is John?",
                    "answer": str(age1 + diff),
                    "type": "word_problem",
                })

        elif problem_type == "negative_ops":
            a = random.randint(-30, 30)
            b = random.randint(-30, 30)
            op = random.choice(["+", "-"])
            ans = a + b if op == "+" else a - b
            prompts.append({
                "question": f"What is {a} {op} {b}?",
                "answer": str(ans),
                "type": "negative_ops",
            })

        else:  # multi_step
            a = random.randint(2, 6)
            b = random.randint(1, 10)
            x = random.randint(1, 10)
            rhs = a * (x + b)
            prompts.append({
                "question": f"Solve for x: {a}(x + {b}) = {rhs}",
                "answer": str(x),
                "type": "multi_step",
            })

    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str,
                        default="Qwen/Qwen3-VL-8B-Thinking-FP8")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="data/raw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate prompts
    print(f"Generating {args.num_samples} math problem prompts...")
    prompt_data = generate_prompts(args.num_samples, seed=args.seed)
    print(f"  Generated {len(prompt_data)} prompts")

    # Load model
    print(f"\nLoading {args.model_name} with vLLM...")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        repetition_penalty=1.05,
    )

    # Build prompts for vLLM
    system_prompt = (
        "You are a math tutor. Solve the problem step by step with clear reasoning. "
        "Show your work, explain each step, then give the final answer on a line starting with 'Answer: '"
    )

    vllm_prompts = []
    for item in prompt_data:
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{item['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        vllm_prompts.append(prompt)

    # Generate in batches
    print(f"\nGenerating responses ({len(vllm_prompts)} prompts)...")
    outputs = llm.generate(vllm_prompts, sampling_params)
    print(f"  Done! Got {len(outputs)} responses")

    # Process results
    sft_samples = []
    valid_count = 0
    for item, output in zip(prompt_data, outputs):
        response = output.outputs[0].text.strip()

        # Validate: must have some content and ideally "Answer:"
        if len(response) > 20:
            sft_samples.append({
                "instruction": item["question"],
                "output": response,
                "type": item["type"],
                "ground_truth": item["answer"],
            })
            valid_count += 1

    print(f"\n  Valid samples: {valid_count}/{len(prompt_data)}")

    # Save
    sft_path = output_dir / "sft_thinking_1k.jsonl"
    with open(sft_path, "w", encoding="utf-8") as f:
        for s in sft_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved to: {sft_path}")

    # Stats
    from collections import Counter
    type_counts = Counter(s["type"] for s in sft_samples)
    print(f"\n  By type:")
    for t, c in type_counts.most_common():
        print(f"    {t}: {c}")

    # Show samples
    print("\n" + "=" * 60)
    print("SAMPLES:")
    print("=" * 60)
    for s in sft_samples[:3]:
        print(f"\n  Q: {s['instruction']}")
        print(f"  A: {s['output'][:200]}...")
        print(f"  True: {s['ground_truth']}")
        print("  ---")


if __name__ == "__main__":
    main()
