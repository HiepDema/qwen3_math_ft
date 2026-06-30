"""SFT on DataMuncher-Labs/LSReasoning-15000 dataset.

Dataset: math reasoning (arithmetic, linear equations, fractions, word problems)
Fields: question, problem, how_to_solve, answer

Usage:
    python scripts/train_sft_lsreasoning.py
    python scripts/train_sft_lsreasoning.py --max-samples 5000 --epochs 3
"""

import argparse
from pathlib import Path

from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig


SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, "
    "show your reasoning clearly, then give the final answer."
)


def format_dataset(dataset, tokenizer, max_samples=None):
    """Format LSReasoning dataset into chat conversations for SFT."""
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    conversations = []
    for item in dataset:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset-name", type=str,
                        default="DataMuncher-Labs/LSReasoning-15000")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max training samples (dataset has 15000)")
    parser.add_argument("--output-dir", type=str, default="outputs/sft_lsreasoning")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    args = parser.parse_args()

    print("=" * 60)
    print("SFT - LSReasoning-15000 (Math Reasoning)")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    print(f"Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
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
    print(f"  Total samples: {len(raw_dataset)}")
    print(f"  Using: {min(args.max_samples, len(raw_dataset))}")

    dataset = format_dataset(raw_dataset, tokenizer, args.max_samples)
    print(f"  Formatted: {len(dataset)} conversations")
    print(f"  Sample:\n{dataset[0]['text'][:400]}...")

    # Split
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"\n  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Training
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        eval_steps=50,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=True,
        seed=42,
        optim="adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    print(f"\nStarting SFT training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"  Learning rate: {args.lr}")
    print()

    train_result = trainer.train()

    # Save LoRA adapter
    final_dir = Path(args.output_dir) / "final"
    print(f"\nSaving LoRA adapter to {final_dir}")
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save merged model (for GRPO to load cleanly)
    merged_dir = Path(args.output_dir) / "merged"
    print(f"Saving merged model to {merged_dir}")
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    eval_metrics = trainer.evaluate()
    metrics = train_result.metrics
    print("\n" + "=" * 60)
    print("SFT Training Complete!")
    print(f"  Train loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Eval loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
    print(f"  Output: {final_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
