#!/bin/bash
# Full project setup script
set -e

echo "=== Qwen3-VL Math Pipeline Setup ==="

# 1. Create virtual environment
echo "[1/6] Creating virtual environment..."
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
echo "[2/6] Installing dependencies..."
pip install -e ".[dev]"

# 3. Setup pre-commit hooks
echo "[3/6] Setting up pre-commit..."
pre-commit install

# 4. Initialize DVC
echo "[4/6] Initializing DVC..."
dvc init
dvc remote add -d minio s3://datasets
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key minioadmin123

# 5. Start infrastructure
echo "[5/6] Starting infrastructure..."
docker compose up -d
sleep 15

# 6. Setup storage buckets
echo "[6/6] Setting up storage..."
python -c "from src.data_infrastructure.storage import setup_storage; setup_storage()"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Copy .env.example to .env and fill in your tokens"
echo "  2. Run 'make data-download' to fetch datasets"
echo "  3. Run 'make data-process' to run ETL pipeline"
echo "  4. Run 'make train-cpt' to start CPT training"
echo ""
echo "Services:"
echo "  MLflow:     http://localhost:5000"
echo "  MinIO:      http://localhost:9001 (admin: minioadmin/minioadmin123)"
echo "  Grafana:    http://localhost:3000 (admin: admin/admin123)"
echo "  Prometheus: http://localhost:9090"
