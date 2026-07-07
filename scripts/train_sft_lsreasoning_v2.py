"""SFT training module for LSReasoning experiment.

Trains on a pre-split train file (JSONL) for fair comparison.
Can be used standalone or imported by run_experiment_lsreasoning.py.

Usage:
    python scripts/train_sft_lsreasoning_v2.py --train-file data/lsreasoning_split/train.jsonl
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig


SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, "
    "show your reasoning clearly, then give the final answer."
)


def load_jsonl(path):
    """Load JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def format_conversations(data, tokenizer):
    """Format raw data into chat conversations for SFT."""
    conversations = []
    for item in data:
        question = item["question"]
        how_to_solve = item["how_to_solve"]
        answer = item["answer"]

        assistant_response = (
            f"Approach: {how_to_solve}\n"
            f"Solving step by step:\n"
            f"{question}\n"
            f"Answer: {answer}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_response},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        conversations.append(text)

    return Dataset.from_dict({"text": conversations})


def train_sft(
    model_name: str = "Qwen/Qwen3-0.6B",
    train_file: str = "data/lsreasoning_split/train.jsonl",
    output_dir: str = "outputs/sft_lsreasoning",
    max_seq_length: int = 1024,
    epochs: int = 3,
    lr: float = 1e-4,
    batch_size: int = 4,
    grad_accum: int = 4,
    lora_r: int = 32,
    lora_alpha: int = 64,
    seed: int = 42,
):
    """Train SFT on the provided training data file."""
    import wandb
    run_name = "cpt_sft" if "cpt" in model_name else "sft_lsreasoning"
    wandb.init(
        project="lsreasoning-sft-vs-grpo",
        name=run_name,
        config={
            "method": "CPT+SFT" if "cpt" in model_name else "SFT",
            "model": model_name,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
        },
        reinit=True,
    )

    print(f"Model: {model_name}")
    print(f"Train file: {train_file}")
    print(f"Output: {output_dir}")

    # Load model (force float16 to match 4-bit dequantization and avoid dtype mismatch)
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
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    # Load and format data
    raw_data = load_jsonl(train_file)
    print(f"Training samples: {len(raw_data)}")

    dataset = format_conversations(raw_data, tokenizer)

    # Use 5% of train as validation for early stopping
    split = dataset.train_test_split(test_size=0.05, seed=seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Val: {len(eval_dataset)}")

    # Training config
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
        save_steps=100,
        save_total_limit=2,
        eval_steps=50,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=True,
        seed=seed,
        optim="adamw_8bit",
        report_to="wandb",
        run_name=run_name,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    print(f"\nStarting SFT...")
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
    print(f"\nSFT Done!")
    print(f"  Train loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Eval loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
    print(f"  Saved: {final_dir}")

    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train-file", type=str,
                        default="data/lsreasoning_split/train.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/sft_lsreasoning")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_sft(
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
