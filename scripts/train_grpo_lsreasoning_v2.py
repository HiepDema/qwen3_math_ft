"""GRPO training module for LSReasoning experiment.

Supports dense and sparse reward modes.
Trains on a pre-split train file (JSONL) for fair comparison.

Usage:
    python scripts/train_grpo_lsreasoning_v2.py --train-file data/lsreasoning_split/train.jsonl --reward-mode dense
    python scripts/train_grpo_lsreasoning_v2.py --train-file data/lsreasoning_split/train.jsonl --reward-mode sparse
"""

import argparse
import json
import re
from pathlib import Path

from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer


SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, "
    "show your reasoning clearly, then give the final answer."
)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


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
    score = 0.0
    if "Answer:" in response or "answer:" in response:
        score += 0.3
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    if len(lines) >= 3:
        score += 0.3
    if any(word in response.lower() for word in
           ["step", "solve", "approach", "therefore", "thus", "so "]):
        score += 0.2
    if re.search(r"[=+\-*/]", response):
        score += 0.2
    return min(score, 1.0)


def reward_reasoning(response, how_to_solve):
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
    return (
        0.4 * reward_correctness(response, true_answer) +
        0.2 * reward_numeric_proximity(response, true_answer) +
        0.2 * reward_format(response) +
        0.2 * reward_reasoning(response, how_to_solve)
    )


def compute_sparse_reward(response, true_answer, how_to_solve):
    return reward_correctness(response, true_answer)


# ============================================================
# Main
# ============================================================

def train_grpo(
    sft_path: str = None,
    model_name: str = "Qwen/Qwen3-0.6B",
    train_file: str = "data/lsreasoning_split/train.jsonl",
    output_dir: str = "outputs/grpo_lsreasoning_dense",
    reward_mode: str = "dense",
    max_seq_length: int = 1024,
    epochs: int = 1,
    lr: float = 5e-6,
    batch_size: int = 4,
    num_generations: int = 4,
    beta: float = 0.1,
    lora_r: int = 16,
    lora_alpha: int = 32,
    seed: int = 42,
):
    """Train GRPO on the provided training data file."""
    print(f"Reward mode: {reward_mode}")
    print(f"SFT path: {sft_path}")
    print(f"Train file: {train_file}")
    print(f"Output: {output_dir}")

    # Load model - prefer merged SFT model
    sft_merged = str(Path(sft_path).parent / "merged") if sft_path else None
    if sft_merged and Path(sft_merged).exists():
        print(f"Loading merged SFT model: {sft_merged}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=sft_merged,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
    elif sft_path and Path(sft_path).exists():
        print(f"Loading SFT adapter: {sft_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, sft_path)
        model = model.merge_and_unload()
    else:
        print(f"Loading base model: {model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    # Load training data
    raw_data = load_jsonl(train_file)
    print(f"Training prompts: {len(raw_data)}")

    # Build prompt dataset + metadata
    dataset_records = []
    metadata = []
    for item in raw_data:
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
    reward_func = compute_dense_reward if reward_mode == "dense" else compute_sparse_reward

    def reward_fn(completions, prompts=None, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            text = completion[-1]["content"] if isinstance(completion, list) else completion
            idx = i // num_generations
            if idx < len(metadata):
                meta = metadata[idx]
                r = reward_func(text, meta["answer"], meta["how_to_solve"])
            else:
                r = 0.0
            rewards.append(r)
        return rewards

    # Training config
    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        max_completion_length=256,
        num_generations=num_generations,
        beta=beta,
        seed=seed,
        optim="adamw_8bit",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        args=training_args,
    )

    print(f"\nStarting GRPO ({reward_mode})...")
    print(f"  Prompts: {len(dataset)}")
    print(f"  Generations/prompt: {num_generations}")
    print(f"  LR: {lr}, Beta: {beta}")
    print()

    train_result = trainer.train()

    final_dir = Path(output_dir) / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metrics = train_result.metrics
    print(f"\nGRPO ({reward_mode}) Done!")
    print(f"  Train loss: {metrics.get('train_loss', 'N/A')}")
    print(f"  Saved: {final_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-path", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train-file", type=str,
                        default="data/lsreasoning_split/train.jsonl")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--reward-mode", choices=["dense", "sparse"], default="dense")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"outputs/grpo_lsreasoning_{args.reward_mode}"

    train_grpo(
        sft_path=args.sft_path,
        model_name=args.model_name,
        train_file=args.train_file,
        output_dir=args.output_dir,
        reward_mode=args.reward_mode,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        beta=args.beta,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
