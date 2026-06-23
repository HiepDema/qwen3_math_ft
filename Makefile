.PHONY: help infra-up infra-down data-download data-process data-version train-cpt train-sft eval optimize serve test lint clean

help:
	@echo "Qwen3-VL Math Fine-tuning Pipeline"
	@echo "===================================="
	@echo ""
	@echo "Infrastructure:"
	@echo "  infra-up        Start Docker services (MinIO, PostgreSQL, MLflow, Airflow)"
	@echo "  infra-down      Stop all services"
	@echo ""
	@echo "Data:"
	@echo "  data-download   Download datasets from HuggingFace"
	@echo "  data-process    Run ETL pipeline (clean, transform, validate)"
	@echo "  data-version    Version datasets with DVC"
	@echo ""
	@echo "Training:"
	@echo "  train-cpt       Continual Pre-Training on math corpus"
	@echo "  train-sft       Supervised Fine-Tuning on math QA"
	@echo ""
	@echo "Evaluation:"
	@echo "  eval            Run full evaluation suite"
	@echo "  eval-benchmark  Run benchmark evaluation (MATH, GSM8K)"
	@echo ""
	@echo "Optimization & Serving:"
	@echo "  optimize        Quantize and optimize model"
	@echo "  serve           Start vLLM inference server"
	@echo ""
	@echo "Development:"
	@echo "  test            Run tests"
	@echo "  lint            Run linter"
	@echo "  clean           Remove artifacts"

# ============================================================
# Infrastructure
# ============================================================
infra-up:
	docker compose up -d
	@echo "Waiting for services..."
	@sleep 10
	@echo "Services ready:"
	@echo "  MinIO:      http://localhost:9000"
	@echo "  MLflow:     http://localhost:5000"
	@echo "  Airflow:    http://localhost:8080"
	@echo "  PostgreSQL: localhost:5432"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana:    http://localhost:3000"

infra-down:
	docker compose down -v

# ============================================================
# Data Pipeline
# ============================================================
data-download:
	python -m src.data_engineering.download

data-process:
	python -m src.data_engineering.etl_pipeline

data-version:
	dvc add data/processed/
	git add data/processed.dvc data/.gitignore
	@echo "Run 'dvc push' to upload to remote storage"

# ============================================================
# Training
# ============================================================
train-cpt:
	python -m src.model.train_cpt --config configs/cpt_config.yaml

train-sft:
	python -m src.model.train_sft --config configs/sft_config.yaml

# ============================================================
# Evaluation
# ============================================================
eval:
	python -m src.evaluation.run_eval --config configs/eval_config.yaml

eval-benchmark:
	python -m src.evaluation.benchmark

# ============================================================
# Optimization & Serving
# ============================================================
optimize:
	python -m src.optimization.quantize --config configs/optimization_config.yaml

serve:
	python -m src.optimization.serve --config configs/serving_config.yaml

# ============================================================
# Development
# ============================================================
test:
	pytest tests/ -v --cov=src

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

clean:
	rm -rf outputs/ runs/ __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
