"""GRPO on LSReasoning-15000 with Dense vs Sparse reward.

Dense reward signals:
- Correctness (0.4): final answer matches ground truth
- Format (0.2): shows "Answer:" and has reasoning steps
- Reasoning (0.2): mentions approach/method
- Numeric proximity (0.2): how close the answer is if not exact

Sparse reward:
- Only correctness (0 or 1)

Usage:
    # Dense reward (after SFT)
    python scripts/train_grpo_lsreasoning.py --reward-mode dense --sft-path outputs/sft_lsreasoning/final

    # Sparse reward (after SFT)
    python scripts/train_grpo_lsreasoning.py --reward-mode sparse --sft-path outputs/sft_lsreasoning/final

    # Dense reward from base model (no SFT)
    python scripts/train_grpo_lsreasoning.py --reward-mode dense
"""

import argparse
import re
from pathlib import Path

from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer


SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, "
    "show your reasoning clearly, then give the final answer."
)


def extract_number(text):
    """Extract numeric answer from text."""
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
    """Parse ground truth answer string to float."""
    try:
        answer_str = answer_str.strip()
        if "/" in answer_str:
            parts = answer_str.split("/")
            return float(parts[0]) / float(parts[1])
        return float(answer_str)
    except (ValueError, ZeroDivisionError):
        return None


# ============================================================
# Reward Functions
# ============================================================

def reward_correctness(response, true_answer):
    """Exact correctness (0 or 1)."""
    pred = extract_number(response)
    if pred is None:
        return 0.0
    true_val = parse_answer(true_answer)
    if true_val is None:
        return 0.0
    if abs(pred - true_val) < 1e-6:
        return 1.0
    return 0.0


def reward_numeric_proximity(response, true_answer):
    """Partial credit for close answers (0 to 1)."""
    pred = extract_number(response)
    if pred is None:
        return 0.0
    true_val = parse_answer(true_answer)
    if true_val is None:
        return 0.0
    if abs(pred - true_val) < 1e-6:
        return 1.0
    if true_val == 0:
        return max(0, 1.0 - abs(pred))
    relative_error = abs(pred - true_val) / max(abs(true_val), 1.0)
    return max(0.0, 1.0 - relative_error)


def reward_format(response):
    """Format reward: has reasoning structure (0 to 1)."""
    score = 0.0
    if "Answer:" in response or "answer:" in response:
        score += 0.3
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    if len(lines) >= 3:
        score += 0.3
    if any(word in response.lower() for word in ["step", "solve", "approach", "therefore", "thus", "so "]):
        score += 0.2
    if re.search(r"[=+\-*/]", response):
        score += 0.2
    return min(score, 1.0)


def reward_reasoning(response, how_to_solve):
    """Does the response show appropriate reasoning? (0 to 1)."""
    score = 0.0
    keywords = how_to_solve.lower().split()
    matches = sum(1 for kw in keywords if kw in response.lower() and len(kw) > 3)
    score += min(0.5, matches * 0.15)

    if any(op in response for op in ["=", "+", "-", "*", "/"]):
        score += 0.3
    if re.search(r"\d+\s*[+\-*/=]\s*\d+", response):
        score += 0.2

    return min(score, 1.0)


def compute_dense_reward(response, true_answer, how_to_solve):
    """Dense reward: weighted combination of sub-rewards."""
    return (
        0.4 * reward_correctness(response, true_answer) +
        0.2 * reward_numeric_proximity(response, true_answer) +
        0.2 * reward_format(response) +
        0.2 * reward_reasoning(response, how_to_solve)
    )


def compute_sparse_reward(response, true_answer, how_to_solve):
    """Sparse reward: only exact correctness."""
    return reward_correctness(response, true_answer)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-path", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset-name", type=str,
                        default="DataMuncher-Labs/LSReasoning-15000")
    parser.add_argument("--max-prompts", type=int, default=3000,
                        help="Number of prompts for GRPO")
    parser.add_argument("--output-dir", type=str, default="outputs/grpo_lsreasoning")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--reward-mode", choices=["dense", "sparse"], default="dense")
    args = parser.parse_args()

    # Output dir includes reward mode
    if args.output_dir == "outputs/grpo_lsreasoning":
        args.output_dir = f"outputs/grpo_lsreasoning_{args.reward_mode}"

    print("=" * 60)
    print(f"GRPO - LSReasoning ({args.reward_mode} reward)")
    print("=" * 60)

    # Load model
    # Prefer merged model (no LoRA conflict), fallback to base
    sft_merged = str(Path(args.sft_path).parent / "merged") if args.sft_path else None
    if sft_merged and Path(sft_merged).exists():
        print(f"\nLoading merged SFT model from: {sft_merged}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=sft_merged,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
    elif args.sft_path and Path(args.sft_path).exists():
        print(f"\nLoading SFT adapter from: {args.sft_path}")
        print("  (No merged model found, loading base + adapter)")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.sft_path)
        model = model.merge_and_unload()
    else:
        print(f"\nLoading base model: {args.model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
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

    # Load dataset
    print(f"\nLoading dataset: {args.dataset_name}")
    raw_dataset = load_dataset(args.dataset_name, split="train")
    if args.max_prompts < len(raw_dataset):
        raw_dataset = raw_dataset.shuffle(seed=42).select(range(args.max_prompts))
    print(f"  Using {len(raw_dataset)} prompts for GRPO")

    # Build prompt dataset
    dataset_records = []
    metadata = []
    for item in raw_dataset:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        dataset_records.append({"prompt": prompt_text})
        metadata.append({
            "answer": item["answer"],
            "how_to_solve": item["how_to_solve"],
        })

    dataset = Dataset.from_list(dataset_records)

    # Reward function
    reward_func = compute_dense_reward if args.reward_mode == "dense" else compute_sparse_reward

    def reward_fn(completions, prompts=None, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            text = completion[-1]["content"] if isinstance(completion, list) else completion
            idx = i // args.num_generations
            if idx < len(metadata):
                meta = metadata[idx]
                r = reward_func(text, meta["answer"], meta["how_to_solve"])
            else:
                r = 0.0
            rewards.append(r)
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
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        max_completion_length=256,
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

    print(f"\nStarting GRPO ({args.reward_mode})...")
    print(f"  Prompts: {len(dataset)}")
    print(f"  Generations/prompt: {args.num_generations}")
    print(f"  LR: {args.lr}, Beta: {args.beta}")
    print()

    train_result = trainer.train()

    final_dir = Path(args.output_dir) / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metrics = train_result.metrics
    print("\n" + "=" * 60)
    print(f"GRPO ({args.reward_mode}) Complete!")
    print(f"  Train loss: {metrics.get('train_loss', 'N/A')}")
    print(f"  Output: {final_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
