"""Supervised Fine-Tuning (SFT) for Qwen3-VL on math QA with Chain-of-Thought.

Trains the model to solve math problems step-by-step using instruction-response pairs
from NuminaMath-CoT dataset. Builds on CPT checkpoint if available.
"""

import argparse
import logging
from pathlib import Path

import mlflow
import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_model(config: dict):
    """Load model, optionally from CPT checkpoint, with QLoRA."""
    model_cfg = config["model"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_cfg["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, model_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=model_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    base_model_name = model_cfg["name"]
    cpt_checkpoint = model_cfg.get("cpt_checkpoint")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if model_cfg.get("use_flash_attention") else "sdpa",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Merge CPT adapter if exists
    if cpt_checkpoint and Path(cpt_checkpoint).exists():
        logger.info(f"Loading CPT adapter from {cpt_checkpoint}")
        model = PeftModel.from_pretrained(model, cpt_checkpoint)
        model = model.merge_and_unload()
        logger.info("CPT adapter merged into base model")

    model = prepare_model_for_kbit_training(model)

    # Apply new LoRA for SFT
    lora_cfg = config["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=lora_cfg["task_type"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def prepare_dataset(config: dict, tokenizer):
    """Load and format SFT dataset with chat template."""
    dataset = load_from_disk("data/processed/sft")

    # Split into train/val
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    def format_conversation(example):
        """Apply chat template to conversations."""
        conversations = example["conversations"]
        text = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    train_dataset = train_dataset.map(format_conversation, num_proc=4)
    eval_dataset = eval_dataset.map(format_conversation, num_proc=4)

    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    return train_dataset, eval_dataset


def train(config_path: str):
    """Run SFT training."""
    config = load_config(config_path)
    train_cfg = config["training"]
    mlflow_cfg = config["mlflow"]

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(run_name=mlflow_cfg["run_name"]):
        mlflow.log_params({
            "model": config["model"]["name"],
            "cpt_checkpoint": config["model"].get("cpt_checkpoint", "none"),
            "lora_r": config["lora"]["r"],
            "lora_alpha": config["lora"]["lora_alpha"],
            "learning_rate": train_cfg["learning_rate"],
            "batch_size": train_cfg["per_device_train_batch_size"],
            "epochs": train_cfg["num_train_epochs"],
            "packing": train_cfg.get("packing", False),
            "neftune_alpha": train_cfg.get("neftune_noise_alpha", 0),
            "training_type": "sft",
        })

        logger.info("Loading model...")
        model, tokenizer = setup_model(config)

        logger.info("Preparing dataset...")
        train_dataset, eval_dataset = prepare_dataset(config, tokenizer)

        sft_config = SFTConfig(
            output_dir=train_cfg["output_dir"],
            num_train_epochs=train_cfg["num_train_epochs"],
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            warmup_ratio=train_cfg["warmup_ratio"],
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            bf16=train_cfg["bf16"],
            gradient_checkpointing=train_cfg["gradient_checkpointing"],
            logging_steps=train_cfg["logging_steps"],
            save_steps=train_cfg["save_steps"],
            save_total_limit=train_cfg["save_total_limit"],
            eval_steps=train_cfg.get("eval_steps", 200),
            eval_strategy=train_cfg.get("eval_strategy", "steps"),
            load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
            metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
            dataloader_num_workers=train_cfg["dataloader_num_workers"],
            optim=train_cfg["optim"],
            max_grad_norm=train_cfg["max_grad_norm"],
            seed=train_cfg["seed"],
            max_seq_length=config["model"]["max_seq_length"],
            packing=train_cfg.get("packing", False),
            neftune_noise_alpha=train_cfg.get("neftune_noise_alpha"),
            dataset_text_field="text",
            report_to="mlflow",
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )

        logger.info("Starting SFT training...")
        train_result = trainer.train()

        # Save final model
        final_dir = Path(train_cfg["output_dir"]) / "final"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))

        # Log metrics
        metrics = train_result.metrics
        eval_metrics = trainer.evaluate()

        mlflow.log_metrics({
            "final_train_loss": metrics.get("train_loss", 0),
            "final_eval_loss": eval_metrics.get("eval_loss", 0),
            "total_steps": metrics.get("train_steps", 0),
            "train_samples_per_second": metrics.get("train_samples_per_second", 0),
        })

        logger.info(f"SFT training complete. Model saved to {final_dir}")
        logger.info(f"Train loss: {metrics.get('train_loss', 'N/A')}")
        logger.info(f"Eval loss: {eval_metrics.get('eval_loss', 'N/A')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sft_config.yaml")
    args = parser.parse_args()
    train(args.config)
