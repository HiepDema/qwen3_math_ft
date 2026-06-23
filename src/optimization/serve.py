"""Model serving with vLLM and FastAPI monitoring endpoint."""

import argparse
import logging
import subprocess
import time
from pathlib import Path

import yaml
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen3-VL Math Inference")

# Prometheus metrics
REQUEST_COUNT = Counter("inference_requests_total", "Total inference requests", ["status"])
REQUEST_LATENCY = Histogram("inference_latency_seconds", "Inference latency", buckets=[0.1, 0.5, 1, 2, 5, 10, 30])
TOKENS_GENERATED = Counter("tokens_generated_total", "Total tokens generated")
GPU_MEMORY_USED = Gauge("gpu_memory_used_bytes", "GPU memory used")
MODEL_LOADED = Gauge("model_loaded", "Whether model is loaded")


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": MODEL_LOADED._value.get()}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def start_vllm_server(config: dict):
    """Start vLLM OpenAI-compatible server."""
    serving_cfg = config["serving"]
    model_path = serving_cfg["model_path"]

    if not Path(model_path).exists():
        # Try AWQ quantized version
        awq_path = Path(model_path).parent / "awq-w4-g128"
        if awq_path.exists():
            model_path = str(awq_path)
        else:
            logger.error(f"Model not found at {model_path}")
            return

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", serving_cfg["host"],
        "--port", str(serving_cfg["port"]),
        "--max-model-len", str(serving_cfg["max_model_len"]),
        "--gpu-memory-utilization", str(serving_cfg["gpu_memory_utilization"]),
        "--tensor-parallel-size", str(serving_cfg["tensor_parallel_size"]),
        "--dtype", serving_cfg["dtype"],
        "--trust-remote-code",
    ]

    if serving_cfg.get("enable_prefix_caching"):
        cmd.append("--enable-prefix-caching")

    if not serving_cfg.get("enforce_eager", True):
        cmd.append("--enforce-eager")

    if serving_cfg.get("max_num_seqs"):
        cmd.extend(["--max-num-seqs", str(serving_cfg["max_num_seqs"])])

    logger.info(f"Starting vLLM server: {' '.join(cmd)}")
    logger.info(f"Server will be available at http://{serving_cfg['host']}:{serving_cfg['port']}")
    logger.info("OpenAI-compatible API endpoints:")
    logger.info(f"  POST http://localhost:{serving_cfg['port']}/v1/chat/completions")
    logger.info(f"  POST http://localhost:{serving_cfg['port']}/v1/completions")
    logger.info(f"  GET  http://localhost:{serving_cfg['port']}/v1/models")

    process = subprocess.Popen(cmd)

    # Wait for server to be ready
    import urllib.request
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://localhost:{serving_cfg['port']}/v1/models")
            logger.info("vLLM server is ready!")
            MODEL_LOADED.set(1)
            break
        except Exception:
            time.sleep(2)

    return process


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/optimization_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    process = start_vllm_server(config)

    if process:
        try:
            process.wait()
        except KeyboardInterrupt:
            process.terminate()
            logger.info("Server stopped")
