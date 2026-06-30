# Qwen3-0.6B Fine-tuning: Math Reasoning (SFT vs GRPO)

Fine-tune Qwen3-0.6B cho bài toán giải phương trình, so sánh **SFT vs GRPO** với **dense reward vs sparse reward**.

## Experiments

### Experiment 1: Quadratic Equations (Phương trình bậc 2)

Data tự sinh bằng deterministic solver + VietJack crawl (phương trình bậc nhất).

| Method | Mô tả |
|--------|--------|
| SFT only | Supervised fine-tuning trên 300 samples |
| GRPO dense | RL với multi-signal reward |
| GRPO sparse | RL chỉ dùng correctness |
| SFT → GRPO | SFT trước, GRPO refine |

### Experiment 2: LSReasoning-15000

Dataset từ HuggingFace: `DataMuncher-Labs/LSReasoning-15000` (arithmetic, algebra, fractions, word problems).

| Method | Mô tả |
|--------|--------|
| SFT only | Train trên 5000 samples |
| SFT → GRPO (dense) | 4 reward signals |
| SFT → GRPO (sparse) | Chỉ đúng/sai |

## Dense vs Sparse Reward

```
Dense Reward = 0.4 × Correctness + 0.2 × Proximity + 0.2 × Format + 0.2 × Reasoning

- Correctness: đáp án đúng/sai (0 hoặc 1)
- Proximity: đáp án gần đúng được partial credit
- Format: có "Answer:", có reasoning steps
- Reasoning: các bước tính toán hợp lệ

Sparse Reward = chỉ Correctness (0 hoặc 1)
```

Dense cho gradient signal phong phú hơn → model học nhanh hơn, đặc biệt ở giai đoạn đầu khi chưa ra được đáp án đúng.

## Data Sources

| Source | Loại | Mô tả |
|--------|------|--------|
| Local generator | Phương trình bậc 2 | 300 samples, deterministic solver |
| VietJack crawl | Phương trình bậc 1 | Cleaned từ web |
| LSReasoning-15000 | Math reasoning (EN) | Arithmetic, algebra, fractions, word problems |

## Project Structure

```
qwen3-vl-math/
├── scripts/
│   ├── prepare_data.py                # Clean VietJack + sinh quadratic data
│   ├── generate_quadratic_data.py     # Sinh data phương trình bậc 2
│   ├── train_sft_quadratic.py         # SFT cho quadratic
│   ├── train_grpo_quadratic.py        # GRPO cho quadratic (dense/sparse)
│   ├── evaluate_quadratic.py          # Eval quadratic
│   ├── run_quadratic_pipeline.py      # Pipeline quadratic
│   ├── train_sft_lsreasoning.py       # SFT cho LSReasoning-15000
│   ├── train_grpo_lsreasoning.py      # GRPO cho LSReasoning (dense/sparse)
│   ├── evaluate_lsreasoning.py        # Eval LSReasoning
│   └── run_lsreasoning_pipeline.py    # Pipeline LSReasoning (3 experiments)
├── configs/
│   ├── sft_quadratic_eq.yaml
│   └── grpo_quadratic_eq.yaml
├── data/raw/
│   ├── cpt_vietjack_crawled.jsonl     # Raw VietJack
│   ├── sft_vietjack_clean.jsonl       # Cleaned
│   ├── sft_quadratic_equations.jsonl  # Generated
│   └── sft_combined.jsonl             # All combined
├── outputs/
│   ├── sft_quadratic_eq/final/
│   ├── grpo_quadratic_eq/final/
│   ├── sft_lsreasoning/final/
│   ├── grpo_lsreasoning_dense/final/
│   └── grpo_lsreasoning_sparse/final/
├── requirements_finetune.txt
└── pyproject.toml
```

## How to Run

### Installation

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_finetune.txt
```

### Pipeline 1: Quadratic Equations

```bash
# So sánh tất cả methods (SFT / GRPO dense / GRPO sparse / SFT+GRPO)
python scripts/run_quadratic_pipeline.py --method compare

# Hoặc chỉ SFT → GRPO
python scripts/run_quadratic_pipeline.py
```

### Pipeline 2: LSReasoning-15000 (Recommended)

```bash
# Chạy full: SFT → eval → GRPO dense → eval → GRPO sparse → eval
python scripts/run_lsreasoning_pipeline.py

# Tuỳ chỉnh
python scripts/run_lsreasoning_pipeline.py --max-samples 5000 --max-prompts 3000 --num-tests 200
```

### Chạy từng bước (LSReasoning)

```bash
# 1. SFT
python scripts/train_sft_lsreasoning.py --max-samples 5000 --epochs 3

# 2. GRPO dense reward
python scripts/train_grpo_lsreasoning.py --sft-path outputs/sft_lsreasoning/final --reward-mode dense

# 3. GRPO sparse reward
python scripts/train_grpo_lsreasoning.py --sft-path outputs/sft_lsreasoning/final --reward-mode sparse

# 4. Evaluate
python scripts/evaluate_lsreasoning.py --model-path outputs/sft_lsreasoning/final --verbose
python scripts/evaluate_lsreasoning.py --model-path outputs/grpo_lsreasoning_dense/final --verbose
python scripts/evaluate_lsreasoning.py --model-path outputs/grpo_lsreasoning_sparse/final --verbose
```

## Training Config

| Parameter | SFT | GRPO |
|-----------|-----|------|
| Base model | Qwen/Qwen3-0.6B | SFT checkpoint |
| LoRA rank | 32 | 16 |
| LoRA alpha | 64 | 32 |
| Learning rate | 1e-4 | 5e-6 |
| Epochs | 3 (LSR) / 5 (quad) | 1 |
| Effective batch size | 16 | 8 |
| Quantization | 4-bit QLoRA | 4-bit QLoRA |
| Num generations | - | 4 per prompt |
| KL beta | - | 0.1 |
| Packing | Yes (SFT) | N/A |

## Expected Results

### LSReasoning-15000

| Method | Exact Match | Format | Notes |
|--------|-------------|--------|-------|
| SFT only | ~45-55% | ~70% | Baseline |
| SFT → GRPO (sparse) | ~50-60% | ~60% | Chỉ reward đúng/sai |
| SFT → GRPO (dense) | ~60-70% | ~85% | Multi-signal reward |

### Quadratic Equations

| Method | Answer Correct | Delta | Format |
|--------|---------------|-------|--------|
| SFT only | ~75% | ~85% | ~88% |
| GRPO sparse only | ~60% | ~70% | ~50% |
| GRPO dense only | ~70% | ~80% | ~75% |
| SFT → GRPO dense | ~88% | ~92% | ~90% |

### Key Findings

- **Dense > Sparse**: Dense reward cho partial credit, model nhận feedback ngay cả khi chưa đúng hoàn toàn
- **SFT → GRPO > GRPO alone**: SFT cung cấp format & knowledge base, GRPO refine reasoning
- **Format reward quan trọng**: Sparse GRPO có thể làm giảm format quality vì chỉ optimize cho đáp án đúng

## Tech Stack

| Component | Tools |
|-----------|-------|
| Base Model | Qwen/Qwen3-0.6B |
| Dataset | LSReasoning-15000, VietJack, Local Gen |
| Training | Unsloth + TRL (SFTTrainer, GRPOTrainer) |
| RL Method | GRPO (dense / sparse reward) |
| LoRA | PEFT, 4-bit QLoRA |
| GPU | NVIDIA A10 (24GB) / T4 (16GB) |
| Eval | Exact Match, Format Score, by Problem Type |
