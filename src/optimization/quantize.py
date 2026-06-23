"""Model quantization and optimization for inference."""

import argparse
import logging
import subprocess
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def merge_adapter(base_model: str, adapter_path: str, output_path: str):
    """Merge LoRA adapter into base model for optimized inference."""
    logger.info("Merging LoRA adapter into base model...")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_path / "merged"))
    tokenizer.save_pretrained(str(output_path / "merged"))

    logger.info(f"Merged model saved to {output_path / 'merged'}")
    return str(output_path / "merged")


def quantize_awq(model_path: str, output_path: str, bits: int = 4, group_size: int = 128):
    """Quantize model using AWQ."""
    from awq import AutoAWQForCausalLM

    logger.info(f"Quantizing with AWQ (w{bits}, group_size={group_size})...")

    model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": group_size,
        "w_bit": bits,
        "version": "GEMM",
    }

    model.quantize(tokenizer, quant_config=quant_config)

    awq_output = Path(output_path) / f"awq-w{bits}-g{group_size}"
    awq_output.mkdir(parents=True, exist_ok=True)

    model.save_quantized(str(awq_output))
    tokenizer.save_pretrained(str(awq_output))

    logger.info(f"AWQ model saved to {awq_output}")
    return str(awq_output)


def export_gguf(model_path: str, output_path: str, quant_types: list[str]):
    """Export model to GGUF format for llama.cpp."""
    logger.info("Exporting to GGUF format...")

    gguf_output = Path(output_path) / "gguf"
    gguf_output.mkdir(parents=True, exist_ok=True)

    # Convert to GGUF using llama.cpp's convert script
    for quant_type in quant_types:
        output_file = gguf_output / f"model-{quant_type}.gguf"

        cmd = [
            "python", "-m", "llama_cpp.convert",
            "--outfile", str(output_file),
            "--outtype", quant_type.lower(),
            model_path,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"GGUF export ({quant_type}): {output_file}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"GGUF export failed for {quant_type}: {e.stderr}")
        except FileNotFoundError:
            logger.warning("llama-cpp-python not installed, skipping GGUF export")
            break


def profile_model(model_path: str, output_dir: str):
    """Profile model inference performance."""
    from vllm import LLM, SamplingParams

    logger.info("Profiling model inference...")

    llm = LLM(model=model_path, trust_remote_code=True, dtype="auto")

    test_prompts = [
        "Solve: What is the derivative of x^3 + 2x^2 - 5x + 3?",
        "Find all solutions to the equation: 2x^2 - 7x + 3 = 0",
        "Prove that the sum of the first n natural numbers is n(n+1)/2.",
    ]

    sampling_params = SamplingParams(temperature=0.1, max_tokens=512)

    import time
    latencies = []
    token_counts = []

    for prompt in test_prompts * 10:
        start = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params)
        elapsed = time.perf_counter() - start

        latencies.append(elapsed * 1000)
        token_counts.append(len(outputs[0].outputs[0].token_ids))

    import numpy as np
    total_tokens = sum(token_counts)
    total_time = sum(latencies) / 1000

    profile_results = {
        "throughput_tokens_per_sec": total_tokens / total_time,
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "avg_tokens_per_request": sum(token_counts) / len(token_counts),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_dir / "profile_results.json", "w") as f:
        json.dump(profile_results, f, indent=2)

    logger.info(f"Profiling results: {profile_results}")
    return profile_results


def run_optimization(config_path: str):
    """Run full optimization pipeline."""
    config = load_config(config_path)
    quant_cfg = config["quantization"]

    input_model = quant_cfg["input_model"]
    output_dir = quant_cfg["output_dir"]

    # Step 1: Merge adapter
    merged_path = merge_adapter("Qwen/Qwen3-VL-7B", input_model, output_dir)

    # Step 2: Quantize with AWQ
    quantize_awq(merged_path, output_dir, bits=quant_cfg["bits"], group_size=quant_cfg["group_size"])

    # Step 3: Export GGUF (optional)
    if quant_cfg.get("gguf", {}).get("enabled"):
        export_gguf(merged_path, output_dir, quant_cfg["gguf"]["quantization_types"])

    # Step 4: Profile
    if config.get("profiling", {}).get("enabled"):
        profile_model(merged_path, config["profiling"]["output_dir"])

    logger.info("Optimization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/optimization_config.yaml")
    args = parser.parse_args()
    run_optimization(args.config)
