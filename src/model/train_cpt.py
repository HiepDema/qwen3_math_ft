"""Continual Pre-Training (CPT) for Qwen3-VL on mathematical corpus.

Extends the model's knowledge of mathematical text through next-token prediction
on a filtered, high-quality math corpus (OpenWebMath).
"""

import argparse
import logging
from pathlib import Path

import mlflow
import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_model(config: dict):
    """Load model with QLoRA quantization."""
    model_cfg = config["model"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_cfg["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, model_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=model_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if model_cfg.get("use_flash_attention") else "sdpa",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

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
    """Load and tokenize dataset for CPT."""
    data_cfg = config["dataset"]
    dataset = load_from_disk("data/processed/cpt")

    max_length = data_cfg["max_length"]

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        num_proc=4,
        remove_columns=dataset.column_names,
    )

    return tokenized


class CPTTrainerCallback(mlflow.pytorch.MLflowCallback if hasattr(mlflow, 'pytorch') else object):
    """Custom callback for CPT-specific logging."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step % 50 == 0:
            logger.info(
                f"Step {state.global_step}: loss={logs.get('loss', 'N/A'):.4f}, "
                f"lr={logs.get('learning_rate', 'N/A'):.2e}"
            )


def train(config_path: str):
    """Run CPT training."""
    config = load_config(config_path)
    train_cfg = config["training"]
    mlflow_cfg = config["mlflow"]

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(run_name=mlflow_cfg["run_name"]):
        mlflow.log_params({
            "model": config["model"]["name"],
            "lora_r": config["lora"]["r"],
            "lora_alpha": config["lora"]["lora_alpha"],
            "learning_rate": train_cfg["learning_rate"],
            "batch_size": train_cfg["per_device_train_batch_size"],
            "grad_accum": train_cfg["gradient_accumulation_steps"],
            "training_type": "cpt",
        })

        logger.info("Loading model...")
        model, tokenizer = setup_model(config)

        logger.info("Preparing dataset...")
        dataset = prepare_dataset(config, tokenizer)
        logger.info(f"Dataset size: {len(dataset)} samples")

        training_args = TrainingArguments(
            output_dir=train_cfg["output_dir"],
            num_train_epochs=train_cfg["num_train_epochs"],
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
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
            dataloader_num_workers=train_cfg["dataloader_num_workers"],
            optim=train_cfg["optim"],
            max_grad_norm=train_cfg["max_grad_norm"],
            seed=train_cfg["seed"],
            report_to="mlflow",
            remove_unused_columns=False,
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        logger.info("Starting CPT training...")
        train_result = trainer.train()

        # Save final model
        final_dir = Path(train_cfg["output_dir"]) / "final"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))

        # Log final metrics
        metrics = train_result.metrics
        mlflow.log_metrics({
            "final_loss": metrics.get("train_loss", 0),
            "total_steps": metrics.get("train_steps", 0),
            "samples_per_second": metrics.get("train_samples_per_second", 0),
        })

        logger.info(f"CPT training complete. Model saved to {final_dir}")
        logger.info(f"Final loss: {metrics.get('train_loss', 'N/A')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cpt_config.yaml")
    args = parser.parse_args()
    train(args.config)
