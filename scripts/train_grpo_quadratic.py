"""GRPO (Group Relative Policy Optimization) training for quadratic equation solving.

After SFT, this script applies GRPO to further improve the model's reasoning
ability on quadratic equations. GRPO uses a reward function instead of a
separate reward model - it groups multiple completions per prompt and optimizes
relative to the group's performance.

Reward signals:
1. Correctness: Does the final answer match the true solution?
2. Format: Does the output follow the step-by-step format?
3. Reasoning: Are intermediate steps (delta calculation, roots) correct?

Usage:
    python scripts/train_grpo_quadratic.py
    python scripts/train_grpo_quadratic.py --sft-path outputs/sft_quadratic_eq/final
"""

import argparse
import math
import random
import re
from fractions import Fraction
from pathlib import Path

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer


SYSTEM_PROMPT = "Bạn là trợ lý toán học. Hãy giải phương trình bậc hai theo từng bước, sử dụng công thức delta."


def is_perfect_square(n):
    if n < 0:
        return False
    root = math.isqrt(n)
    return root * root == n


def generate_training_prompts(num_prompts=300, seed=42):
    """Generate quadratic equation prompts for GRPO training."""
    random.seed(seed)
    prompts = []
    seen = set()

    while len(prompts) < num_prompts:
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
        instruction = f"Giải phương trình bậc hai: {eq_str}"

        prompts.append({
            "prompt": instruction,
            "a": a,
            "b": b,
            "c": c,
        })

    return prompts


def compute_correct_answer(a, b, c):
    """Compute the correct answer for verification."""
    delta = b * b - 4 * a * c

    if delta < 0:
        return {"type": "no_solution", "delta": delta}
    elif delta == 0:
        x = Fraction(-b, 2 * a)
        return {"type": "one_solution", "delta": delta, "x": str(x)}
    else:
        sqrt_delta = math.isqrt(delta) if is_perfect_square(delta) else None
        if sqrt_delta is not None:
            x1 = Fraction(-b + sqrt_delta, 2 * a)
            x2 = Fraction(-b - sqrt_delta, 2 * a)
            return {"type": "two_solutions", "delta": delta, "x1": str(x1), "x2": str(x2)}
        else:
            return {"type": "two_solutions_irrational", "delta": delta}


def reward_correctness(response, a, b, c):
    """Reward for correct final answer (0.0 to 1.0)."""
    answer = compute_correct_answer(a, b, c)

    if answer["type"] == "no_solution":
        if "vô nghiệm" in response.lower():
            return 1.0
        return 0.0

    elif answer["type"] == "one_solution":
        x_val = answer["x"]
        if f"x = {x_val}" in response or f"x={x_val}" in response:
            return 1.0
        if "nghiệm kép" in response.lower():
            return 0.5
        return 0.0

    elif answer["type"] == "two_solutions":
        x1, x2 = answer["x1"], answer["x2"]
        found_x1 = f"= {x1}" in response or f"={x1}" in response
        found_x2 = f"= {x2}" in response or f"={x2}" in response
        if found_x1 and found_x2:
            return 1.0
        elif found_x1 or found_x2:
            return 0.5
        return 0.0

    else:
        delta = answer["delta"]
        if f"Δ = {delta}" in response or f"delta = {delta}" in response.lower():
            return 0.5
        if "hai nghiệm" in response.lower():
            return 0.3
        return 0.0


def reward_format(response):
    """Reward for following the expected format (0.0 to 1.0)."""
    score = 0.0
    checks = [
        ("Ta có" in response or "phương trình" in response.lower(), 0.15),
        ("a =" in response or "a=" in response, 0.1),
        ("b =" in response or "b=" in response, 0.1),
        ("c =" in response or "c=" in response, 0.1),
        ("Δ" in response or "delta" in response.lower(), 0.15),
        ("Đáp án" in response or "Vậy" in response, 0.2),
        (len(response.strip().split("\n")) >= 4, 0.2),
    ]

    for condition, weight in checks:
        if condition:
            score += weight

    return min(score, 1.0)


def reward_reasoning(response, a, b, c):
    """Reward for correct intermediate reasoning steps (0.0 to 1.0)."""
    delta = b * b - 4 * a * c
    score = 0.0

    if str(delta) in response:
        score += 0.4

    if delta > 0 and is_perfect_square(delta):
        sqrt_d = math.isqrt(delta)
        if str(sqrt_d) in response:
            score += 0.3

    if delta < 0 and ("< 0" in response or "âm" in response):
        score += 0.3
    elif delta == 0 and ("= 0" in response):
        score += 0.3
    elif delta > 0 and ("> 0" in response or "dương" in response):
        score += 0.3

    if "2a" in response or str(2 * a) in response:
        score += 0.3

    return min(score, 1.0)


def combined_reward(response, a, b, c):
    """Combined reward function for GRPO."""
    r_correct = reward_correctness(response, a, b, c)
    r_format = reward_format(response)
    r_reasoning = reward_reasoning(response, a, b, c)

    reward = 0.5 * r_correct + 0.2 * r_format + 0.3 * r_reasoning
    return reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-path", type=str, default="outputs/sft_quadratic_eq/final",
                        help="Path to SFT model checkpoint")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B",
                        help="Base model (used if no SFT path)")
    parser.add_argument("--output-dir", type=str, default="outputs/grpo_quadratic_eq")
    parser.add_argument("--num-prompts", type=int, default=300)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Number of completions per prompt for GRPO")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty coefficient")
    args = parser.parse_args()

    print("=" * 60)
    print("GRPO - Group Relative Policy Optimization")
    print("Task: Quadratic Equation Solving")
    print("=" * 60)

    model_path = args.sft_path if Path(args.sft_path).exists() else args.model_name
    print(f"\nLoading model: {model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    print(f"Applying LoRA for GRPO (r={args.lora_r}, alpha={args.lora_alpha})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print(f"\nGenerating {args.num_prompts} training prompts...")
    prompt_data = generate_training_prompts(args.num_prompts, seed=42)

    dataset_records = []
    for item in prompt_data:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["prompt"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        dataset_records.append({
            "prompt": prompt_text,
            "a": item["a"],
            "b": item["b"],
            "c": item["c"],
        })

    dataset = Dataset.from_list(dataset_records)
    print(f"Dataset size: {len(dataset)} prompts")

    def reward_fn(completions, prompts=None, **kwargs):
        """Reward function for GRPO trainer."""
        rewards = []
        for i, completion in enumerate(completions):
            if isinstance(completion, list):
                text = completion[-1]["content"] if completion else ""
            else:
                text = completion

            idx = i // args.num_generations
            if idx < len(dataset_records):
                a = dataset_records[idx]["a"]
                b = dataset_records[idx]["b"]
                c = dataset_records[idx]["c"]
                reward = combined_reward(text, a, b, c)
            else:
                reward = 0.0
            rewards.append(reward)
        return rewards

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        max_completion_length=512,
        num_generations=args.num_generations,
        beta=args.beta,
        seed=42,
        optim="adamw_8bit",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        args=training_args,
    )

    print("\nStarting GRPO training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num generations per prompt: {args.num_generations}")
    print(f"  Learning rate: {args.lr}")
    print(f"  KL beta: {args.beta}")
    print(f"  Base model: {model_path}")
    print()

    train_result = trainer.train()

    final_dir = Path(args.output_dir) / "final"
    print(f"\nSaving GRPO model to {final_dir}")
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metrics = train_result.metrics
    print("\n" + "=" * 60)
    print("GRPO Training Complete!")
    print(f"  Final train loss: {metrics.get('train_loss', 'N/A')}")
    print(f"  Mean reward: {metrics.get('reward/mean', 'N/A')}")
    print(f"  Output: {final_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
