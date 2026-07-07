"""Continued Pre-Training (CPT) on math domain text from LSReasoning.

CPT adapts the base model to math domain BEFORE instruction tuning (SFT).
Uses plain text passages derived from the training split's how_to_solve fields
plus generated math theory passages.

Pipeline: Base Model -> CPT -> SFT -> (optional GRPO)

Usage:
    python scripts/train_cpt_lsreasoning.py --train-file data/lsreasoning_split/train.jsonl
"""

import argparse
import json
import random
from math import gcd
from pathlib import Path

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_cpt_corpus(train_file: str, seed: int = 42):
    """Build CPT corpus from training data + generated math passages."""
    random.seed(seed)
    corpus = []

    # 1. Extract domain text from train data (how_to_solve + question patterns)
    raw_data = load_jsonl(train_file)
    for item in raw_data:
        how_to_solve = item.get("how_to_solve", "")
        question = item.get("question", "")
        answer = item.get("answer", "")
        if how_to_solve and len(how_to_solve) > 20:
            corpus.append(
                f"Problem: {question}\n"
                f"Solution approach: {how_to_solve}\n"
                f"The answer is {answer}."
            )

    # 2. Generate additional math theory passages
    theory_passages = [
        "Addition is the process of combining two or more numbers to find their total. When adding multi-digit numbers, align them by place value and add from right to left, carrying over when a column sum exceeds 9.",
        "Subtraction finds the difference between two numbers. To subtract, align numbers by place value and subtract from right to left, borrowing from the next column when needed.",
        "Multiplication is repeated addition. The distributive property states that a(b + c) = ab + ac. This is useful for mental math: 7 × 13 = 7 × 10 + 7 × 3 = 70 + 21 = 91.",
        "Division splits a number into equal groups. Long division involves dividing, multiplying, subtracting, and bringing down digits repeatedly until complete.",
        "The order of operations (PEMDAS/BODMAS): Parentheses/Brackets first, then Exponents/Orders, then Multiplication and Division left to right, then Addition and Subtraction left to right.",
        "A linear equation has the form ax + b = c. To solve: isolate x by performing inverse operations. Subtract b from both sides, then divide by a.",
        "Fractions represent parts of a whole. To add fractions, find a common denominator: a/b + c/d = (ad + bc)/(bd). Always simplify the result by dividing by the GCD.",
        "Multiplying fractions: multiply numerators and denominators separately. a/b × c/d = ac/bd. Dividing fractions: multiply by the reciprocal. a/b ÷ c/d = a/b × d/c.",
        "Word problems require translation from English to math. Identify the unknown (let x = ...), write the equation from the given relationships, solve, and verify.",
        "Percentages are fractions with denominator 100. To find x% of n: multiply n × x/100. To find what percent a is of b: compute (a/b) × 100.",
        "Negative numbers follow specific sign rules. Positive × Positive = Positive, Negative × Negative = Positive, Positive × Negative = Negative.",
        "When solving equations with variables on both sides, collect variable terms on one side and constants on the other. For 3x + 5 = x + 13: subtract x gives 2x + 5 = 13, subtract 5 gives 2x = 8, divide by 2 gives x = 4.",
        "Simplifying expressions: combine like terms (same variable and exponent). 3x + 2y + 5x - y = 8x + y. Constants combine separately.",
        "Ratios and proportions: if a/b = c/d, then ad = bc (cross multiplication). Use this to solve for unknowns in proportional relationships.",
        "The greatest common divisor (GCD) of two numbers is the largest number that divides both evenly. Use it to simplify fractions. GCD(12, 18) = 6, so 12/18 = 2/3.",
    ]
    for passage in theory_passages:
        corpus.append(passage)

    # 3. Generate computed examples as natural text
    for _ in range(500):
        choice = random.choice(["arithmetic", "equation", "fraction", "word"])

        if choice == "arithmetic":
            op = random.choice(["+", "-", "*"])
            a = random.randint(-50, 50)
            b = random.randint(-50, 50)
            if op == "+":
                result = a + b
                text = f"Computing {a} + {b}: the sum equals {result}."
            elif op == "-":
                result = a - b
                text = f"Computing {a} - {b}: the difference equals {result}."
            else:
                a = random.randint(-12, 12)
                b = random.randint(-12, 12)
                result = a * b
                text = f"Computing {a} × {b}: the product equals {result}."
            corpus.append(text)

        elif choice == "equation":
            coeff = random.choice([i for i in range(2, 10)])
            x_val = random.randint(-15, 15)
            const = random.randint(-30, 30)
            rhs = coeff * x_val + const
            corpus.append(
                f"Solving the equation {coeff}x + {const} = {rhs}: "
                f"subtract {const} from both sides to get {coeff}x = {rhs - const}, "
                f"then divide by {coeff} to get x = {x_val}."
            )

        elif choice == "fraction":
            num = random.randint(1, 30)
            den = random.randint(2, 15)
            g = gcd(num, den)
            corpus.append(
                f"Simplifying {num}/{den}: the GCD of {num} and {den} is {g}, "
                f"so {num}/{den} simplifies to {num // g}/{den // g}."
            )

        else:
            items = random.randint(3, 20)
            price = random.randint(2, 50)
            total = items * price
            corpus.append(
                f"If you buy {items} items at ${price} each, the total cost is "
                f"{items} × ${price} = ${total}."
            )

    random.shuffle(corpus)
    return corpus


def train_cpt(
    model_name: str = "Qwen/Qwen3-0.6B",
    train_file: str = "data/lsreasoning_split/train.jsonl",
    output_dir: str = "outputs/cpt_lsreasoning",
    max_seq_length: int = 1024,
    epochs: int = 2,
    lr: float = 2e-4,
    batch_size: int = 4,
    grad_accum: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    seed: int = 42,
):
    """Continued pre-training on math domain text."""
    import wandb
    wandb.init(
        project="lsreasoning-sft-vs-grpo",
        name="cpt_lsreasoning",
        config={
            "method": "CPT",
            "model": model_name,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
        },
        reinit=True,
    )

    print(f"Model: {model_name}")
    print(f"Train file: {train_file}")
    print(f"Output: {output_dir}")

    # Load model (force float16 to match 4-bit dequantization)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=torch.float16,
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

    # Build CPT corpus
    print("Building CPT corpus...")
    corpus = build_cpt_corpus(train_file, seed=seed)
    print(f"  CPT corpus size: {len(corpus)} passages")

    dataset = Dataset.from_dict({"text": corpus})
    split = dataset.train_test_split(test_size=0.05, seed=seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Val: {len(eval_dataset)}")

    # Training config - CLM (no chat template, just raw text)
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=True,
        bf16=False,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        eval_steps=100,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=True,
        seed=seed,
        optim="adamw_8bit",
        report_to="wandb",
        run_name="cpt_lsreasoning",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    print(f"\nStarting CPT...")
    print(f"  Epochs: {epochs}, LR: {lr}")
    print(f"  Effective batch: {batch_size * grad_accum}")
    print()

    train_result = trainer.train()

    # Save
    final_dir = Path(output_dir) / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    merged_dir = Path(output_dir) / "merged"
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    eval_metrics = trainer.evaluate()
    metrics = train_result.metrics
    print(f"\nCPT Done!")
    print(f"  Train loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Eval loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
    print(f"  Saved: {final_dir}")

    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train-file", type=str,
                        default="data/lsreasoning_split/train.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/cpt_lsreasoning")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_cpt(
        model_name=args.model_name,
        train_file=args.train_file,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
