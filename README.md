# Fine-tuning Qwen3-0.6B for Math Reasoning

So sánh 4 phương pháp fine-tuning **Qwen3-0.6B** trên dataset **LSReasoning-15000**:

| Method | Pipeline | Exact Match |
|--------|----------|:-----------:|
| **CPT + SFT** | Base → CPT → SFT | **85.2%** |
| SFT only | Base → SFT | 83.6% |
| SFT + GRPO (dense) | Base → SFT → GRPO (4 rewards) | 81.8% |
| SFT + GRPO (sparse) | Base → SFT → GRPO (binary) | 79.4% |

Full report: [reports/experiment_report.md](reports/experiment_report.md)

wandb: [lsreasoning-sft-vs-grpo](https://wandb.ai/hiep26-sdf/lsreasoning-sft-vs-grpo)

---

## Quick Start

### 1. Requirements

- Python 3.10+
- NVIDIA GPU with >= 16GB VRAM (tested on A10 24GB)
- CUDA 12.1+

### 2. Installation

```bash
git clone https://github.com/<your-username>/qwen3-vl-math.git
cd qwen3-vl-math

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Install PyTorch (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements_finetune.txt

# Login wandb (for tracking)
wandb login
```

### 3. Run Full Experiment

```bash
# Chạy 4 experiments: CPT+SFT, SFT, GRPO dense, GRPO sparse
python scripts/run_experiment_lsreasoning.py
```

Tự động:
- Download LSReasoning-15000 từ HuggingFace
- Split 80/20 (train/test) với seed=42
- Train & eval từng method
- Log metrics lên wandb
- In kết quả comparison

### 4. Run Individual Steps

```bash
# Chỉ CPT
python scripts/train_cpt_lsreasoning.py --train-file data/lsreasoning_split/train.jsonl

# Chỉ SFT
python scripts/train_sft_lsreasoning_v2.py --train-file data/lsreasoning_split/train.jsonl

# Chỉ GRPO (cần SFT model đã train)
python scripts/train_grpo_lsreasoning_v2.py \
  --train-file data/lsreasoning_split/train.jsonl \
  --sft-path outputs/sft_lsreasoning/final \
  --reward-mode dense

# Evaluate bất kỳ model nào
python scripts/evaluate_lsreasoning_v2.py \
  --model-path outputs/cpt_sft_lsreasoning/final \
  --test-file data/lsreasoning_split/test.jsonl
```

### 5. Skip Steps (nếu đã train)

```bash
# Skip SFT (đã train xong)
python scripts/run_experiment_lsreasoning.py --skip-sft

# Chỉ chạy eval (tất cả model đã train)
python scripts/run_experiment_lsreasoning.py --skip-cpt --skip-sft --skip-grpo-dense --skip-grpo-sparse

# Skip CPT+SFT
python scripts/run_experiment_lsreasoning.py --skip-cpt
```

### 6. Inference Optimization (KV Cache Benchmark)

```bash
python scripts/inference_optimized.py \
  --model-path outputs/cpt_sft_lsreasoning/final \
  --benchmark \
  --num-samples 50 \
  --output-file outputs/benchmark_results.json
```

### 7. Export Charts from wandb

```bash
python scripts/export_wandb_charts.py --entity hiep26-sdf --project lsreasoning-sft-vs-grpo
```

Saves PNG charts to `reports/figures/`.

---

## Project Structure

```
qwen3-vl-math/
├── scripts/
│   ├── run_experiment_lsreasoning.py      # Main pipeline (4 experiments)
│   ├── train_cpt_lsreasoning.py           # CPT training
│   ├── train_sft_lsreasoning_v2.py        # SFT training
│   ├── train_grpo_lsreasoning_v2.py       # GRPO training (dense/sparse)
│   ├── evaluate_lsreasoning_v2.py         # Batch evaluation
│   ├── inference_optimized.py             # KV Cache benchmark
│   ├── export_wandb_charts.py             # Download charts from wandb
│   └── plot_comparison_wandb.py           # Log comparison to wandb
├── data/
│   └── lsreasoning_split/
│       ├── train.jsonl                    # 12,000 samples (auto-generated)
│       └── test.jsonl                     # 3,000 samples (auto-generated)
├── outputs/
│   ├── cpt_lsreasoning/final/            # CPT checkpoint
│   ├── cpt_sft_lsreasoning/final/        # CPT+SFT checkpoint
│   ├── sft_lsreasoning/final/            # SFT checkpoint
│   ├── grpo_lsreasoning_dense/final/     # GRPO dense checkpoint
│   ├── grpo_lsreasoning_sparse/final/    # GRPO sparse checkpoint
│   ├── experiment_results.json            # Final metrics
│   └── benchmark_results.json             # KV Cache results
├── reports/
│   ├── experiment_report.md               # Full report
│   └── figures/                           # Charts (from export script)
├── .github/workflows/
│   └── lsreasoning_experiment.yml         # CI/CD pipeline
├── requirements_finetune.txt              # Dependencies
└── README.md
```

---

## Dataset

**LSReasoning-15000** ([HuggingFace](https://huggingface.co/datasets/DataMuncher-Labs/LSReasoning-15000))

| Property | Value |
|----------|-------|
| Size | 15,000 samples |
| Split | 80% train / 20% test (seed=42) |
| Language | English |
| Types | Arithmetic, linear equations, two-step equations, fractions, exponents, algebra, inequalities |

Sample:
```json
{
  "question": "Solve for x: 3x + 7 = 22",
  "problem": "Solve the linear equation.",
  "how_to_solve": "Subtract 7 from both sides to get 3x = 15, then divide by 3.",
  "answer": "5"
}
```

---

## Training Configuration

| Parameter | CPT | SFT | GRPO |
|-----------|:---:|:---:|:----:|
| Base model | Qwen3-0.6B | Qwen3-0.6B / CPT merged | SFT merged |
| Data | ~12,500 plain text | 12,000 chat pairs | 1,000 prompts × 4 generations |
| Epochs | 2 | 3 | 1 |
| Learning rate | 2e-4 | 1e-4 | 5e-6 |
| LoRA r / alpha | 16 / 32 | 32 / 64 | 16 / 32 |
| Effective batch | 16 | 16 | 8 |
| Quantization | 4-bit QLoRA | 4-bit QLoRA | 4-bit QLoRA |
| Precision | fp16 | fp16 | fp16 |
| Optimizer | adamw_8bit | adamw_8bit | adamw_8bit |

---

## Reward Functions (GRPO)

### Dense Reward
```
R = 0.4 × Correctness + 0.2 × Proximity + 0.2 × Format + 0.2 × Reasoning
```
- **Correctness**: exact answer match (0 or 1)
- **Proximity**: partial credit based on relative error (0 to 1)
- **Format**: has "Answer:", multi-line reasoning, math operators (0 to 1)
- **Reasoning**: keyword overlap with reference approach (0 to 1)

### Sparse Reward
```
R = Correctness (0 or 1)
```

---

## Key Findings

1. **CPT+SFT wins** (85.2%) — domain pre-training improves downstream SFT
2. **GRPO does not help** at 0.6B scale with high SFT baseline (83.6%)
3. **Dense > Sparse** within GRPO, but both underperform SFT
4. **Two-step equations** are the key differentiator between methods
5. **Static KV Cache** gives 1.04x inference speedup

---

## Estimated Training Time (A10 24GB)

| Stage | Time |
|-------|:----:|
| CPT | ~15 min |
| SFT | ~30 min |
| GRPO (×2) | ~15 min each |
| Eval (×4) | ~5 min each |
| **Total** | **~2 hours** |

---

## CI/CD

GitHub Actions workflow at `.github/workflows/lsreasoning_experiment.yml`:
- **validate**: syntax check + reward function unit tests
- **prepare-data**: download + split dataset
- **train**: full experiment on self-hosted GPU runner
- Auto-comments results on PRs

Trigger manually via `workflow_dispatch` or on push to training scripts.

---

## License

MIT
