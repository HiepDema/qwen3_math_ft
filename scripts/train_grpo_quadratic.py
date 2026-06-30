"""GRPO (Group Relative Policy Optimization) with Dense Reward.

Trains Qwen3-0.6B using GRPO on quadratic equation prompts.
Dense reward = multiple sub-rewards summed with weights, giving rich gradient signal.

Reward breakdown:
- Correctness (0.4): final answer matches ground truth
- Delta (0.2): discriminant calculation is correct
- Format (0.2): follows step-by-step format
- Reasoning (0.2): intermediate steps are valid

Usage:
    # GRPO only (from base model)
    python scripts/train_grpo_quadratic.py --model-name Qwen/Qwen3-0.6B

    # GRPO after SFT
    python scripts/train_grpo_quadratic.py --sft-path outputs/sft_quadratic_eq/final

    # Compare: run with different reward weights
    python scripts/train_grpo_quadratic.py --reward-mode sparse   # only correctness
    python scripts/train_grpo_quadratic.py --reward-mode dense    # all sub-rewards
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


def format_quadratic(a, b, c):
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


def generate_prompts(num_prompts=300, seed=42):
    """Generate quadratic equation prompts with ground truth."""
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

        eq_str = format_quadratic(a, b, c)
        delta = b * b - 4 * a * c

        prompts.append({
            "instruction": f"Giải phương trình bậc hai: {eq_str}",
            "a": a, "b": b, "c": c, "delta": delta,
        })

    return prompts


# ============================================================
# Dense Reward Functions
# ============================================================

def reward_correctness(response, a, b, c):
    """Is the final answer correct? (0.0 or 1.0)"""
    delta = b * b - 4 * a * c

    if delta < 0:
        return 1.0 if "vô nghiệm" in response.lower() else 0.0

    elif delta == 0:
        x = Fraction(-b, 2 * a)
        if f"= {x}" in response:
            return 1.0
        if "nghiệm kép" in response.lower():
            return 0.5
        return 0.0

    else:
        sqrt_d = math.isqrt(delta) if is_perfect_square(delta) else None
        if sqrt_d is not None:
            x1 = Fraction(-b + sqrt_d, 2 * a)
            x2 = Fraction(-b - sqrt_d, 2 * a)
            found_x1 = f"= {x1}" in response
            found_x2 = f"= {x2}" in response
            if found_x1 and found_x2:
                return 1.0
            if found_x1 or found_x2:
                return 0.5
        else:
            if str(delta) in response and "hai nghiệm" in response.lower():
                return 0.7
        return 0.0


def reward_delta(response, a, b, c):
    """Is delta calculated correctly? (0.0 or 1.0)"""
    delta = b * b - 4 * a * c
    return 1.0 if str(delta) in response else 0.0


def reward_format(response):
    """Does it follow step-by-step format? (0.0 to 1.0)"""
    score = 0.0
    if "Ta có" in response or "phương trình" in response:
        score += 0.2
    if "a =" in response or "a=" in response:
        score += 0.15
    if "Δ" in response or "delta" in response.lower():
        score += 0.25
    if "Đáp án" in response or "Vậy" in response:
        score += 0.2
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    if len(lines) >= 5:
        score += 0.2
    return min(score, 1.0)


def reward_reasoning(response, a, b, c):
    """Are intermediate reasoning steps present and correct? (0.0 to 1.0)"""
    delta = b * b - 4 * a * c
    score = 0.0

    if "b²" in response or "b² -" in response:
        score += 0.2
    if "4ac" in response or "4·" in response:
        score += 0.2
    if delta < 0 and ("< 0" in response):
        score += 0.3
    elif delta == 0 and ("= 0" in response):
        score += 0.3
    elif delta > 0 and ("> 0" in response):
        score += 0.3
    if "2a" in response or f"/{2*a}" in response:
        score += 0.3

    return min(score, 1.0)


def compute_dense_reward(response, a, b, c):
    """Combined dense reward (weighted sum of sub-rewards)."""
    return (
        0.4 * reward_correctness(response, a, b, c) +
        0.2 * reward_delta(response, a, b, c) +
        0.2 * reward_format(response) +
        0.2 * reward_reasoning(response, a, b, c)
    )


def compute_sparse_reward(response, a, b, c):
    """Sparse reward: only correctness matters."""
    return reward_correctness(response, a, b, c)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-path", type=str, default=None,
                        help="Path to SFT checkpoint (optional, for SFT→GRPO)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", type=str, default="outputs/grpo_quadratic_eq")
    parser.add_argument("--num-prompts", type=int, default=300)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--reward-mode", choices=["dense", "sparse"], default="dense",
                        help="dense = multi-signal reward, sparse = only correctness")
    args = parser.parse_args()

    print("=" * 60)
    print("GRPO Training - Quadratic Equations")
    print(f"Reward mode: {args.reward_mode}")
    print("=" * 60)

    # Load model
    if args.sft_path and Path(args.sft_path).exists():
        model_path = args.sft_path
        print(f"\nLoading SFT model: {model_path}")
    else:
        model_path = args.model_name
        print(f"\nLoading base model: {model_path}")
        if args.sft_path:
            print(f"  (SFT path '{args.sft_path}' not found, using base model)")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    print(f"Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
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

    # Generate prompts
    print(f"\nGenerating {args.num_prompts} training prompts...")
    prompt_data = generate_prompts(args.num_prompts, seed=42)

    # Build dataset
    dataset_records = []
    for item in prompt_data:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["instruction"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        dataset_records.append({
            "prompt": prompt_text,
            "a": item["a"],
            "b": item["b"],
            "c": item["c"],
            "delta": item["delta"],
        })

    dataset = Dataset.from_list(dataset_records)
    print(f"Dataset: {len(dataset)} prompts")

    # Choose reward function
    reward_func = compute_dense_reward if args.reward_mode == "dense" else compute_sparse_reward

    def reward_fn(completions, prompts=None, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            text = completion[-1]["content"] if isinstance(completion, list) else completion
            idx = i // args.num_generations
            if idx < len(dataset_records):
                a = dataset_records[idx]["a"]
                b = dataset_records[idx]["b"]
                c = dataset_records[idx]["c"]
                rewards.append(reward_func(text, a, b, c))
            else:
                rewards.append(0.0)
        return rewards

    # Training
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

    print(f"\nStarting GRPO ({args.reward_mode} reward)...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num generations: {args.num_generations}")
    print(f"  LR: {args.lr}")
    print(f"  Beta (KL): {args.beta}")
    print()

    train_result = trainer.train()

    final_dir = Path(args.output_dir) / "final"
    print(f"\nSaving to {final_dir}")
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metrics = train_result.metrics
    print("\n" + "=" * 60)
    print("GRPO Training Complete!")
    print(f"  Reward mode: {args.reward_mode}")
    print(f"  Train loss: {metrics.get('train_loss', 'N/A')}")
    print(f"  Output: {final_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
