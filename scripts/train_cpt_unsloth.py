"""CPT (Continual Pre-Training) using Unsloth + LoRA on Qwen3-0.6B.

Trains the model on math-related text corpus to improve its understanding
of linear equation concepts in Vietnamese.

Usage:
    python scripts/train_cpt_unsloth.py
    python scripts/train_cpt_unsloth.py --data-path data/raw/cpt_linear_equations.jsonl
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig


def load_cpt_data(data_path: str) -> Dataset:
    """Load CPT data from JSONL file."""
    texts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            texts.append(item["text"])

    return Dataset.from_dict({"text": texts})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data-path", type=str, default="data/raw/cpt_linear_equations.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/cpt_linear_eq")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    args = parser.parse_args()

    print("=" * 60)
    print("CPT - Continual Pre-Training on Linear Equation Corpus")
    print("=" * 60)

    # Load model with Unsloth (4-bit quantization for efficiency)
    print(f"\nLoading model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,  # auto-detect
    )

    # Apply LoRA
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
    print(f"\nLoading CPT data from: {args.data_path}")
    dataset = load_cpt_data(args.data_path)
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Sample: {dataset[0]['text'][:200]}...")

    # Training config - CPT uses plain text, no chat template
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=True,
        seed=42,
        optim="adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    print("\nStarting CPT training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_accum}")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"  Learning rate: {args.lr}")
    print()

    train_result = trainer.train()

    # Save
    final_dir = Path(args.output_dir) / "final"
    print(f"\nSaving CPT model to {final_dir}")
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Print results
    metrics = train_result.metrics
    print("\n" + "=" * 60)
    print("CPT Training Complete!")
    print(f"  Final loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Total steps: {metrics.get('train_steps', 'N/A')}")
    print(f"  Output: {final_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
