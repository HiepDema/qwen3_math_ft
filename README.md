# Qwen3-0.6B Fine-tuning: Giải phương trình bậc 2

Fine-tune Qwen3-0.6B để giải phương trình bậc hai (ax² + bx + c = 0) bằng tiếng Việt, so sánh **SFT vs GRPO** (dense reward vs sparse reward).

## Mục tiêu

So sánh hiệu quả của các phương pháp training:
| Method | Mô tả |
|--------|--------|
| **SFT only** | Supervised fine-tuning trên data có sẵn |
| **GRPO only (dense)** | RL với multi-signal reward (correctness + delta + format + reasoning) |
| **GRPO only (sparse)** | RL với chỉ 1 signal (đúng/sai) |
| **SFT → GRPO** | SFT trước, rồi GRPO để cải thiện reasoning |

## Data Sources

| Source | Loại | Số lượng | Mô tả |
|--------|------|----------|--------|
| VietJack crawl | Phương trình bậc nhất | ~30-50 clean pairs | Data thực từ web, đã clean |
| Local generator | Phương trình bậc hai | 300 samples | Deterministic solver, 100% đúng |

**Input:** `"Giải phương trình bậc hai: 2x² - 5x + 3 = 0"`

**Output:**
```
Ta có phương trình: 2x² - 5x + 3 = 0
Với a = 2, b = -5, c = 3
Tính delta: Δ = b² - 4ac = (-5)² - 4·(2)·(3) = 25 - (24) = 1
Vì Δ > 0 nên phương trình có hai nghiệm phân biệt:
√Δ = √1 = 1
x₁ = (-b + √Δ)/(2a) = (5 + 1)/4 = 3/2
x₂ = (-b - √Δ)/(2a) = (5 - 1)/4 = 1
Đáp án: x₁ = 3/2, x₂ = 1
```

## GRPO Dense Reward

```
Dense Reward = 0.4 × Correctness + 0.2 × Delta + 0.2 × Format + 0.2 × Reasoning

- Correctness: đáp án cuối cùng đúng/sai (0 hoặc 1)
- Delta: tính discriminant đúng (0 hoặc 1)
- Format: có đủ các phần (Ta có / a,b,c / Δ / Đáp án / >= 5 dòng)
- Reasoning: bước trung gian hợp lệ (b², 4ac, xét dấu, 2a)

vs Sparse Reward = chỉ Correctness (0 hoặc 1)
```

## Project Structure

```
qwen3-vl-math/
├── scripts/
│   ├── prepare_data.py              # Clean VietJack + sinh quadratic data
│   ├── generate_quadratic_data.py   # Sinh quadratic data (standalone)
│   ├── train_sft_quadratic.py       # SFT training (LoRA)
│   ├── train_grpo_quadratic.py      # GRPO training (dense/sparse reward)
│   ├── evaluate_quadratic.py        # Đánh giá model
│   └── run_quadratic_pipeline.py    # Full pipeline + compare experiments
├── configs/
│   ├── sft_quadratic_eq.yaml        # SFT config
│   └── grpo_quadratic_eq.yaml       # GRPO config
├── data/raw/
│   ├── cpt_vietjack_crawled.jsonl   # Raw VietJack data
│   ├── sft_vietjack_clean.jsonl     # Cleaned → SFT format
│   ├── sft_quadratic_equations.jsonl # Generated quadratic data
│   └── sft_combined.jsonl           # All data combined
├── outputs/
│   ├── sft_quadratic_eq/final/      # SFT model
│   ├── grpo_quadratic_eq/final/     # GRPO model
│   └── exp_*/final/                 # Experiment models (compare mode)
├── requirements_finetune.txt
└── pyproject.toml
```

## How to Run

### Prerequisites

- Python >= 3.10
- GPU with CUDA >= 8GB VRAM (A10 recommended, T4 OK)

### Installation

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_finetune.txt
```

### Option 1: Full Pipeline (SFT → GRPO)

```bash
python scripts/run_quadratic_pipeline.py
```

### Option 2: So sánh tất cả methods

```bash
python scripts/run_quadratic_pipeline.py --method compare
```

Sẽ chạy 4 experiments và eval từng cái:
1. SFT only
2. GRPO dense only
3. GRPO sparse only
4. SFT → GRPO dense

### Option 3: Chạy từng bước

```bash
# 1. Chuẩn bị data (VietJack clean + quadratic gen)
python scripts/prepare_data.py --num-quadratic 300

# 2a. SFT
python scripts/train_sft_quadratic.py

# 2b. GRPO (dense reward, sau SFT)
python scripts/train_grpo_quadratic.py --sft-path outputs/sft_quadratic_eq/final --reward-mode dense

# 2c. Hoặc GRPO only (từ base model, không SFT)
python scripts/train_grpo_quadratic.py --reward-mode dense

# 3. Evaluate
python scripts/evaluate_quadratic.py --model-path outputs/grpo_quadratic_eq/final --verbose
```

### Option 4: Chỉ 1 method cụ thể

```bash
# SFT only
python scripts/run_quadratic_pipeline.py --method sft

# GRPO only (dense)
python scripts/run_quadratic_pipeline.py --method grpo --reward-mode dense

# GRPO only (sparse) 
python scripts/run_quadratic_pipeline.py --method grpo --reward-mode sparse
```

## Training Config

| Parameter | SFT | GRPO |
|-----------|-----|------|
| Base model | Qwen/Qwen3-0.6B | SFT checkpoint hoặc base |
| LoRA rank | 32 | 16 |
| LoRA alpha | 64 | 32 |
| Learning rate | 1e-4 | 5e-6 |
| Epochs | 5 | 1 |
| Batch size | 16 (effective) | 8 (effective) |
| Quantization | 4-bit QLoRA | 4-bit QLoRA |
| Num generations | - | 4 per prompt |
| KL beta | - | 0.1 |

## Evaluation Metrics

```bash
python scripts/evaluate_quadratic.py --model-path outputs/grpo_quadratic_eq/final --num-tests 50
```

| Metric | Đo gì | Target |
|--------|--------|--------|
| **Exact Match** | Delta đúng + đáp án đúng | > 70% |
| **Answer Correct** | Đáp án cuối đúng | > 75% |
| **Delta Correct** | Tính Δ đúng | > 85% |
| **Format Compliance** | Đúng format step-by-step | > 85% |
| **All Steps Correct** | Tất cả đúng | > 65% |

## Expected Results

| Method | Answer Correct | Delta | Format |
|--------|---------------|-------|--------|
| SFT only | ~75% | ~85% | ~88% |
| GRPO sparse only | ~60% | ~70% | ~50% |
| GRPO dense only | ~70% | ~80% | ~75% |
| **SFT → GRPO dense** | **~88%** | **~92%** | **~90%** |

Dense reward > sparse reward vì cung cấp gradient signal phong phú hơn cho model.
SFT → GRPO > GRPO only vì SFT cho base format/knowledge, GRPO refine reasoning.

## Tech Stack

| Component | Tools |
|-----------|-------|
| Base Model | Qwen/Qwen3-0.6B |
| Training | Unsloth + TRL (SFTTrainer, GRPOTrainer) |
| RL Method | GRPO (dense reward function) |
| LoRA | PEFT, 4-bit QLoRA |
| Data | VietJack crawl + local deterministic solver |
| GPU | NVIDIA A10 (24GB) / T4 (16GB) |
