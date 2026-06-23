"""Evaluation pipeline: benchmarks, metrics, and LLM-as-Judge."""

import argparse
import json
import logging
import re
from pathlib import Path

import mlflow
import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(config: dict):
    """Load model for evaluation."""
    model_cfg = config["model"]
    model_path = model_cfg["name"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    if Path(model_path).exists():
        # Load from local checkpoint (adapter)
        base_model_name = "Qwen/Qwen3-VL-7B"
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model.eval()
    return model, tokenizer


def extract_answer(text: str) -> str:
    """Extract final answer from model output."""
    # Look for \boxed{} pattern
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1].strip()

    # Look for "The answer is" pattern
    answer_match = re.search(r"(?:the answer is|answer:|final answer:?)\s*(.+?)(?:\.|$)", text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()

    # Last number in text
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]

    return text.strip().split("\n")[-1].strip()


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.strip().lower()
    answer = answer.replace(",", "").replace("$", "").replace("%", "")
    answer = re.sub(r"\s+", " ", answer)
    try:
        return str(float(answer))
    except ValueError:
        return answer


def evaluate_benchmark(model, tokenizer, benchmark_config: dict, gen_config: dict) -> dict:
    """Evaluate on a single benchmark."""
    name = benchmark_config["name"]
    logger.info(f"Evaluating on {name}...")

    # Load dataset
    load_kwargs = {"path": benchmark_config["dataset"], "split": benchmark_config["split"]}
    ds = load_dataset(**load_kwargs)

    num_samples = benchmark_config.get("num_samples", len(ds))
    ds = ds.select(range(min(num_samples, len(ds))))

    correct = 0
    total = 0
    predictions = []

    for sample in ds:
        # Format prompt
        if name == "gsm8k":
            question = sample["question"]
            gold_answer = sample["answer"].split("####")[-1].strip()
        else:
            question = sample.get("problem", sample.get("question", ""))
            gold_answer = sample.get("solution", sample.get("answer", ""))

        messages = [
            {"role": "system", "content": "Solve the math problem step by step. Put your final answer in \\boxed{}."},
            {"role": "user", "content": question},
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_config.get("max_new_tokens", 1024),
                temperature=gen_config.get("temperature", 0.1),
                top_p=gen_config.get("top_p", 0.95),
                do_sample=gen_config.get("do_sample", False),
                repetition_penalty=gen_config.get("repetition_penalty", 1.1),
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_answer = extract_answer(response)

        is_correct = normalize_answer(pred_answer) == normalize_answer(gold_answer)
        if is_correct:
            correct += 1
        total += 1

        predictions.append({
            "question": question[:200],
            "gold": gold_answer,
            "predicted": pred_answer,
            "correct": is_correct,
            "full_response": response[:500],
        })

    accuracy = correct / max(total, 1)
    result = {
        "benchmark": name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "predictions": predictions,
    }

    logger.info(f"{name}: accuracy={accuracy:.4f} ({correct}/{total})")
    return result


def run_evaluation(config_path: str):
    """Run full evaluation suite."""
    config = load_config(config_path)

    logger.info("Loading model...")
    model, tokenizer = load_model(config)

    results = {}
    for benchmark_cfg in config["benchmarks"]:
        result = evaluate_benchmark(model, tokenizer, benchmark_cfg, config["generation"])
        results[benchmark_cfg["name"]] = result

    # Save results
    output_dir = Path(config["reporting"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Log to MLflow
    if config["reporting"].get("mlflow_tracking"):
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("qwen3-vl-math-eval")

        with mlflow.start_run(run_name="evaluation"):
            for name, result in results.items():
                mlflow.log_metric(f"{name}_accuracy", result["accuracy"])
                mlflow.log_metric(f"{name}_total", result["total"])

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    for name, result in results.items():
        logger.info(f"  {name:20s}: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval_config.yaml")
    args = parser.parse_args()
    run_evaluation(args.config)
