#!/bin/bash
set -e

echo "============================================"
echo "  Setup Lambda Cloud Server (A10 GPU)"
echo "============================================"

# Create and activate venv
python -m venv venv
source venv/bin/activate

# Install torch (CUDA 12.1)
pip install --upgrade pip
pip install torch>=2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install unsloth (pulls compatible transformers, peft, trl)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Training + evaluation deps
pip install wandb matplotlib datasets huggingface_hub

# Deployment + monitoring deps
pip install fastapi uvicorn prometheus_client vllm

# Fix torchao conflict (unsloth may pull it, causes torch.int1 error)
pip uninstall torchao -y 2>/dev/null || true

echo ""
echo "============================================"
echo "  Setup complete! Activate with:"
echo "  source venv/bin/activate"
echo "============================================"
