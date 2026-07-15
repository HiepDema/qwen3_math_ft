"""Standalone SFT training + evaluation + save merged model.

Usage:
    python scripts/train_sft_only.py
    python scripts/train_sft_only.py --epochs 3 --lr 1e-4
"""

import argparse
import json
import re
import os
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig


SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, "
    "show your reasoning clearly, then give the final answer."
)

DATA_DIR = Path("data/lsreasoning_split")
TRAIN_FILE = DATA_DIR / "train.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def prepare_data_split(seed=42):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TRAIN_FILE.exists() and TEST_FILE.exists():
        train_count = sum(1 for _ in open(TRAIN_FILE))
        test_count = sum(1 for _ in open(TEST_FILE))
        print(f"Data split exists: train={train_count}, test={test_count}")
        return

    print("Loading dataset: DataMuncher-Labs/LSReasoning-15000")
    dataset = load_dataset("DataMuncher-Labs/LSReasoning-15000", split="train")
    split = dataset.train_test_split(test_size=0.2, seed=seed)

    for path, data in [(TRAIN_FILE, split["train"]), (TEST_FILE, split["test"])]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved: train={len(split['train'])}, test={len(split['test'])}")


def format_conversations(data, tokenizer):
    conversations = []
    for item in data:
        assistant_response = (
            f"Approach: {item['how_to_solve']}\n"
            f"Solving step by step:\n"
            f"{item['question']}\n"
            f"Answer: {item['answer']}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": assistant_response},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        conversations.append(text)
    return Dataset.from_dict({"text": conversations})


def extract_number(text):
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
        try:
            return float(numbers[-1])
        except ValueError:
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


def evaluate(model, tokenizer, test_file, num_eval=500):
    print(f"\nEvaluating on {test_file} ({num_eval} samples)...")
    FastLanguageModel.for_inference(model)

    test_data = load_jsonl(test_file)[:num_eval]
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    correct = 0
    total = len(test_data)
    batch_size = 16

    for i in range(0, total, batch_size):
        batch = test_data[i:i + batch_size]
        prompts = []
        for item in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["question"]},
            ]
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            ))

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=768,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, output in enumerate(outputs):
            prompt_len = inputs["input_ids"][j].ne(tokenizer.pad_token_id).sum()
            response = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            pred = extract_number(response)
            true_val = parse_answer(batch[j]["answer"])
            if pred is not None and true_val is not None and abs(pred - true_val) < 1e-6:
                correct += 1

        if (i // batch_size + 1) % 5 == 0:
            print(f"  [{min(i + batch_size, total)}/{total}] accuracy so far: {correct}/{min(i + batch_size, total)}")

    accuracy = correct / total
    print(f"\nResult: {correct}/{total} = {accuracy:.1%}")
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", type=str, default="outputs/sft_lsreasoning")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--num-eval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ.setdefault("WANDB_PROJECT", "lsreasoning-sft-vs-grpo")

    print("=" * 60)
    print("  SFT Training Pipeline")
    print("=" * 60)

    # Step 1: Prepare data
    prepare_data_split(seed=args.seed)

    # Step 2: Load model
    print(f"\nLoading model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=torch.float16,
    )

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
        random_state=args.seed,
    )

    # Step 3: Format data
    raw_data = load_jsonl(str(TRAIN_FILE))
    print(f"Training samples: {len(raw_data)}")
    dataset = format_conversations(raw_data, tokenizer)

    split = dataset.train_test_split(test_size=0.05, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Val: {len(eval_dataset)}")

    # Step 4: Train
    import wandb
    wandb.init(
        project="lsreasoning-sft-vs-grpo",
        name="sft_lsreasoning",
        config=vars(args),
        reinit=True,
    )

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
        fp16=True,
        bf16=False,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        eval_steps=50,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_seq_length=1024,
        dataset_text_field="text",
        packing=True,
        seed=args.seed,
        optim="adamw_8bit",
        report_to="wandb",
        run_name="sft_lsreasoning",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    print(f"\nStarting SFT training...")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum}")
    train_result = trainer.train()

    # Step 5: Save
    final_dir = Path(args.output_dir) / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    merged_dir = Path(args.output_dir) / "merged"
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    metrics = train_result.metrics
    eval_metrics = trainer.evaluate()
    print(f"\nSFT Done!")
    print(f"  Train loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Eval loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
    print(f"  Saved: {final_dir}")
    print(f"  Merged: {merged_dir}")

    # Step 6: Evaluate on test set
    accuracy = evaluate(model, tokenizer, str(TEST_FILE), num_eval=args.num_eval)

    wandb.log({"eval/accuracy": accuracy})
    wandb.finish()

    print(f"\n{'=' * 60}")
    print(f"  DONE! Accuracy: {accuracy:.1%}")
    print(f"  Model saved: {merged_dir}")
    print(f"  Next: python scripts/serve_model_monitored.py --model-path {merged_dir} --backend vllm")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
