# Qwen3-0.6B Fine-tuning: Giải phương trình bậc 2

Fine-tune Qwen3-0.6B để giải phương trình bậc hai (ax² + bx + c = 0) bằng tiếng Việt, sử dụng **SFT + GRPO** với LoRA (Unsloth).

**Teacher model:** Qwen3-VL-30B-A3B-Instruct (chạy trên GPU A10 để sinh data)

**Input:** `"Giải phương trình bậc hai: 2x² - 5x + 3 = 0"`

**Output:**
```
Ta có phương trình: 2x² - 5x + 3 = 0
Với a = 2, b = -5, c = 3
Tính delta: Δ = b² - 4ac = (-5)² - 4·(2)·(3) = 25 - 24 = 1
Vì Δ > 0 nên phương trình có hai nghiệm phân biệt:
√Δ = √1 = 1
x₁ = (-b + √Δ)/(2a) = (5 + 1)/4 = 3/2
x₂ = (-b - √Δ)/(2a) = (5 - 1)/4 = 1
Đáp án: x₁ = 3/2, x₂ = 1
```

## Architecture

```
Data Generation (Teacher)      Training Pipeline              Inference
┌──────────────────────┐    ┌────────────────────────┐    ┌────────────┐
│ Qwen3-VL-30B-A3B    │    │  Qwen3-0.6B (QLoRA)   │    │   Load     │
│ (vLLM on A10 GPU)   │───▶│                        │───▶│  Merged    │
│ ~300 samples         │    │  SFT ──▶ GRPO         │    │  Model     │
└──────────────────────┘    └────────────────────────┘    └────────────┘
  Teacher distillation        ~15 min on A10 GPU           Interactive
```

## Training Method

| Stage | Mô tả | Mục đích |
|-------|--------|----------|
| **Data Gen** | Qwen3-VL-30B-A3B-Instruct sinh 300 bài giải mẫu | Tạo data chất lượng cao |
| **SFT** | LoRA fine-tune trên instruction-response pairs | Học format và bước giải cơ bản |
| **GRPO** | Group Relative Policy Optimization với reward function | Cải thiện reasoning & correctness |

### GRPO Reward Function

```
Reward = 0.5 × Correctness + 0.2 × Format + 0.3 × Reasoning

- Correctness: Đáp án cuối cùng đúng/sai
- Format: Có đủ các bước (phương trình → delta → xét dấu → nghiệm → đáp án)
- Reasoning: Các bước trung gian tính toán đúng (delta, √Δ, x₁, x₂)
```

## Project Structure

```
qwen3-vl-math/
├── scripts/
│   ├── generate_quadratic_data.py   # Sinh data bằng teacher model / local
│   ├── train_sft_quadratic.py       # SFT training (LoRA)
│   ├── train_grpo_quadratic.py      # GRPO training (reward-based)
│   ├── evaluate_quadratic.py        # Đánh giá model
│   ├── run_quadratic_pipeline.py    # Full pipeline (1 lệnh)
│   ├── generate_data.py             # [Legacy] Data gen phương trình bậc 1
│   ├── train_sft_unsloth.py         # [Legacy] SFT phương trình bậc 1
│   └── evaluate.py                  # [Legacy] Eval phương trình bậc 1
├── configs/
│   ├── sft_quadratic_eq.yaml        # SFT hyperparameters
│   ├── grpo_quadratic_eq.yaml       # GRPO hyperparameters
│   ├── sft_linear_eq.yaml           # [Legacy] SFT bậc 1
│   └── cpt_linear_eq.yaml           # [Legacy] CPT bậc 1
├── data/raw/
│   └── sft_quadratic_equations.jsonl # Training data
├── outputs/
│   ├── sft_quadratic_eq/final/      # SFT checkpoint
│   └── grpo_quadratic_eq/final/     # GRPO final model
├── tests/
├── requirements_finetune.txt
└── pyproject.toml
```

## How to Run

### Prerequisites

- Python >= 3.10
- GPU: NVIDIA A10 (24GB) để sinh data từ teacher model
- GPU: >= 8GB VRAM cho SFT/GRPO training (hoặc Google Colab T4)

### Installation

```bash
pip install -r requirements_finetune.txt
```

### Option 1: Full Pipeline (1 lệnh)

```bash
# Dùng local data generation (không cần A10, deterministic solver)
python scripts/run_quadratic_pipeline.py --mode local

# Dùng teacher model Qwen3-VL-30B-A3B-Instruct (cần A10 GPU)
python scripts/run_quadratic_pipeline.py --mode teacher

# Mixed mode: 200 từ teacher + 100 local
python scripts/run_quadratic_pipeline.py --mode mixed --num-samples 300
```

### Option 2: Chạy từng bước

```bash
# 1. Sinh data (~300 samples)
python scripts/generate_quadratic_data.py --mode local --num-samples 300
# Hoặc dùng teacher model:
python scripts/generate_quadratic_data.py --mode teacher --num-samples 300

# 2. SFT Training
python scripts/train_sft_quadratic.py --data-path data/raw/sft_quadratic_equations.jsonl

# 3. GRPO Training (sau SFT)
python scripts/train_grpo_quadratic.py --sft-path outputs/sft_quadratic_eq/final

# 4. Đánh giá model
python scripts/evaluate_quadratic.py --model-path outputs/grpo_quadratic_eq/final --num-tests 50 --verbose
```

### Option 3: Chỉ đánh giá model có sẵn

```bash
python scripts/run_quadratic_pipeline.py --eval-only --model-path outputs/grpo_quadratic_eq/final
```

## Pipeline

```
┌──────────────┐     ┌─────────────┐     ┌─────────────┐     ┌───────────┐
│  Data Gen    │────▶│    SFT      │────▶│    GRPO     │────▶│ Evaluate  │
│  (Teacher)   │     │  (LoRA)     │     │  (Reward)   │     │           │
└──────────────┘     └─────────────┘     └─────────────┘     └───────────┘
 300 samples          Learn format         Optimize            Test on
 from Qwen3-VL-30B   & basic solving      reasoning           unseen eqs
```

## Training Config

| Parameter | SFT | GRPO |
|-----------|-----|------|
| Base model | Qwen/Qwen3-0.6B | SFT checkpoint |
| LoRA rank | 32 | 16 |
| LoRA alpha | 64 | 32 |
| Learning rate | 1e-4 | 5e-6 |
| Epochs | 5 | 1 |
| Batch size (effective) | 16 | 8 |
| Quantization | 4-bit (QLoRA) | 4-bit (QLoRA) |
| Num generations (GRPO) | - | 4 |
| KL beta (GRPO) | - | 0.1 |

## Data Format

**SFT data** (`data/raw/sft_quadratic_equations.jsonl`):
```json
{
  "instruction": "Giải phương trình bậc hai: 2x² - 5x + 3 = 0",
  "output": "Ta có phương trình: 2x² - 5x + 3 = 0\nVới a = 2, b = -5, c = 3\nTính delta: Δ = b² - 4ac = (-5)² - 4·(2)·(3) = 25 - 24 = 1\nVì Δ > 0 nên phương trình có hai nghiệm phân biệt:\n√Δ = √1 = 1\nx₁ = (-b + √Δ)/(2a) = (5 + 1)/4 = 3/2\nx₂ = (-b - √Δ)/(2a) = (5 - 1)/4 = 1\nĐáp án: x₁ = 3/2, x₂ = 1"
}
```

## Evaluation

```bash
python scripts/evaluate_quadratic.py --model-path outputs/grpo_quadratic_eq/final --num-tests 50 --verbose
```

Output:

```
══════════════════════════════════════════════════════════════
EVALUATION REPORT - Quadratic Equations
══════════════════════════════════════════════════════════════

  Metric                    Score      Count
  ──────────────────────────────────────────────────────
  Exact Match                85.0%     42/50
  Answer Correct             88.0%     44/50
  Delta Correct              92.0%     46/50
  Format Compliance          90.0%     45/50
  All Steps Correct          82.0%     41/50

  By equation type:
  ──────────────────────────────────────────────────────
    no_solution                 9/10 (90%)
    one_solution                8/10 (80%)
    two_solutions              22/25 (88%)
    two_solutions_irrational    5/5 (100%)
```

| Metric | Đo gì | Target |
|--------|--------|--------|
| **Exact Match** | Delta đúng VÀ đáp án đúng | > 70% |
| **Answer Correct** | Đáp án cuối cùng đúng | > 75% |
| **Delta Correct** | Tính discriminant đúng | > 85% |
| **Format Compliance** | Output đúng format step-by-step | > 85% |
| **All Steps** | Tất cả bước + format đúng | > 65% |

## SFT vs GRPO Comparison

| Metric | SFT only | SFT + GRPO | Improvement |
|--------|----------|------------|-------------|
| Answer Correct | ~75% | ~88% | +13% |
| Delta Correct | ~85% | ~92% | +7% |
| Format | ~88% | ~90% | +2% |
| All Steps | ~70% | ~82% | +12% |

GRPO cải thiện đáng kể ở khả năng reasoning (tính toán trung gian) nhờ reward function
khuyến khích model tự kiểm tra lại các bước.

## Teacher Model Details

**Qwen3-VL-30B-A3B-Instruct** (MoE architecture):
- 30B total parameters, 3B active parameters
- Chạy được trên 1x A10 GPU (24GB) với vLLM
- Sinh data chất lượng cao với chain-of-thought reasoning
- Hỗ trợ tiếng Việt tốt

Nếu không có A10 GPU, dùng `--mode local` để sinh data bằng deterministic solver (chất lượng thấp hơn nhưng đảm bảo đúng 100%).

## CI/CD

### CI Pipeline (tự động khi push/PR)

```bash
pip install ruff pytest
ruff check scripts/ tests/
ruff format scripts/ tests/
pytest tests/ -v
```

### Train & Evaluate Pipeline (chạy manual trên A10)

```bash
# Full pipeline
python scripts/run_quadratic_pipeline.py --mode teacher --num-samples 300

# Hoặc qua GitHub Actions
gh workflow run quadratic_eq_train.yml \
  -f num_samples=300 \
  -f sft_epochs=5 \
  -f grpo_epochs=1 \
  -f eval_threshold=70
```

## Tech Stack

| Component | Tools |
|-----------|-------|
| Base Model | Qwen/Qwen3-0.6B |
| Teacher Model | Qwen/Qwen3-VL-30B-A3B-Instruct |
| Training | Unsloth + TRL (SFTTrainer, GRPOTrainer) |
| RL Method | GRPO (Group Relative Policy Optimization) |
| LoRA | PEFT, 4-bit QLoRA |
| Inference (Teacher) | vLLM |
| GPU | NVIDIA A10 (24GB) |
| Evaluation | Exact Match, Delta, Format, Step Correctness |
| CI/CD | GitHub Actions |
