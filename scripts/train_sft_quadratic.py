"""SFT (Supervised Fine-Tuning) for quadratic equation solving.

Trains Qwen3-0.6B with LoRA on quadratic equation data generated from
Qwen3-VL-30B-A3B-Instruct teacher model.

Usage:
    python scripts/train_sft_quadratic.py
    python scripts/train_sft_quadratic.py --data-path data/raw/sft_quadratic_equations.jsonl
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig


SYSTEM_PROMPT = "Bạn là trợ lý toán học. Hãy giải phương trình bậc hai theo từng bước, sử dụng công thức delta."


def load_sft_data(data_path: str, tokenizer) -> Dataset:
    """Load SFT data and format as chat conversations."""
    conversations = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["output"]},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            conversations.append(text)

    return Dataset.from_dict({"text": conversations})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data-path", type=str, default="data/raw/sft_quadratic_equations.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/sft_quadratic_eq")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    args = parser.parse_args()

    print("=" * 60)
    print("SFT - Quadratic Equation Solving (Qwen3-0.6B + LoRA)")
    print("=" * 60)

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

    print(f"\nLoading SFT data from: {args.data_path}")
    dataset = load_sft_data(args.data_path, tokenizer)
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Sample:\n{dataset[0]['text'][:500]}...")

    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

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
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        eval_steps=25,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
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

    print("\nStarting SFT training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"  Learning rate: {args.lr}")
    print()

    train_result = trainer.train()

    final_dir = Path(args.output_dir) / "final"
    print(f"\nSaving LoRA adapter to {final_dir}")
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    merged_dir = Path(args.output_dir) / "merged"
    print(f"Saving merged model to {merged_dir}")
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    eval_metrics = trainer.evaluate()
    metrics = train_result.metrics
    print("\n" + "=" * 60)
    print("SFT Training Complete!")
    print(f"  Final train loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Final eval loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
    print(f"  Output: {final_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
