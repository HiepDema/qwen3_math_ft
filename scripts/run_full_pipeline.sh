#!/bin/bash
# Run the complete ML pipeline end-to-end
set -e

echo "=== Qwen3-VL Math: Full Pipeline ==="
echo "Started at: $(date)"
echo ""

# Parse arguments
SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-false}
SKIP_CPT=${SKIP_CPT:-false}

# Step 1: Data Download
if [ "$SKIP_DOWNLOAD" = "false" ]; then
    echo "[Step 1/7] Downloading datasets..."
    python -m src.data_engineering.download
    echo ""
fi

# Step 2: ETL Pipeline
echo "[Step 2/7] Running ETL pipeline..."
python -m src.data_engineering.etl_pipeline
echo ""

# Step 3: Data Versioning
echo "[Step 3/7] Versioning data with DVC..."
dvc add data/processed/
git add data/processed.dvc
echo ""

# Step 4: Sanity Check
echo "[Step 4/7] Running sanity checks..."
python -m src.model.sanity_check --model Qwen/Qwen3-VL-7B --dataset data/processed/cpt --type cpt
python -m src.model.sanity_check --model Qwen/Qwen3-VL-7B --dataset data/processed/sft --type sft
echo ""

# Step 5: CPT Training
if [ "$SKIP_CPT" = "false" ]; then
    echo "[Step 5/7] Continual Pre-Training..."
    python -m src.model.train_cpt --config configs/cpt_config.yaml
    echo ""
fi

# Step 6: SFT Training
echo "[Step 6/7] Supervised Fine-Tuning..."
python -m src.model.train_sft --config configs/sft_config.yaml
echo ""

# Step 7: Evaluation
echo "[Step 7/7] Running evaluation..."
python -m src.evaluation.run_eval --config configs/eval_config.yaml
echo ""

echo "=== Pipeline Complete ==="
echo "Finished at: $(date)"
echo ""
echo "Results:"
echo "  - Model checkpoint: outputs/sft/final/"
echo "  - Evaluation results: outputs/eval/eval_results.json"
echo "  - MLflow dashboard: http://localhost:5000"
echo ""
echo "Next steps:"
echo "  - Run 'make optimize' to quantize the model"
echo "  - Run 'make serve' to start inference server"
