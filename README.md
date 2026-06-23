# Qwen3-0.6B Fine-tuning: Giải phương trình bậc nhất

Fine-tune Qwen3-0.6B để giải phương trình bậc nhất một ẩn (tiếng Việt) sử dụng CPT + SFT với LoRA (Unsloth).

**Input:** `"Giải phương trình: 3x - 5 = 16"`

**Output:**
```
Ta có:
3x - 5 = 16
3x = 21
x = 7
Đáp án: x = 7
```

## Architecture

```
Data Generation          Training (LoRA + Unsloth)           Inference
┌────────────┐     ┌──────────────────────────────┐     ┌────────────┐
│ Gemini API │     │  Qwen3-0.6B (4-bit QLoRA)    │     │   Load     │
│ OpenAI API │────▶│                              │────▶│  Merged    │
│ Local Gen  │     │  CPT ──▶ Merge ──▶ SFT      │     │  Model     │
└────────────┘     └──────────────────────────────┘     └────────────┘
 ~150 samples        ~10 min on T4 GPU                   Interactive
```

## Project Structure

```
qwen3-vl-math/
├── scripts/
│   ├── generate_data.py          # Sinh data (Gemini/OpenAI/Local)
│   ├── train_cpt_unsloth.py      # CPT training
│   ├── train_sft_unsloth.py      # SFT training
│   ├── evaluate.py               # Đánh giá model (Exact Match, Format, Steps)
│   ├── inference.py              # Test model
│   └── run_pipeline.py           # Full pipeline (1 lệnh)
├── configs/
│   ├── cpt_linear_eq.yaml        # CPT hyperparameters
│   └── sft_linear_eq.yaml        # SFT hyperparameters
├── notebooks/
│   └── finetune_qwen3_linear_eq.py  # Colab notebook (all-in-one)
├── data/raw/                     # Generated training data
├── outputs/
│   ├── cpt_linear_eq/final/      # CPT checkpoint
│   └── sft_linear_eq/final/      # SFT final model
├── requirements_finetune.txt
└── pyproject.toml
```

## How to Run

### Prerequisites

- Python >= 3.10
- GPU with CUDA (recommend >= 8GB VRAM, hoặc dùng Google Colab T4 free)

### Installation

```bash
pip install -r requirements_finetune.txt
```

### Option 1: Full Pipeline (1 lệnh duy nhất)

```bash
# Dùng data sinh local (không cần API key)
python scripts/run_pipeline.py --api local

# Dùng Gemini API để sinh data chất lượng hơn
python scripts/run_pipeline.py --api gemini --api-key YOUR_GEMINI_KEY

# Dùng OpenAI API
python scripts/run_pipeline.py --api openai --api-key YOUR_OPENAI_KEY
```

### Option 2: Chạy từng bước

```bash
# 1. Sinh data (150 CPT + 150 SFT samples)
python scripts/generate_data.py --api local
# Hoặc với Gemini:
python scripts/generate_data.py --api gemini --api-key YOUR_KEY

# 2. CPT - Continual Pre-Training (học ngôn ngữ toán học)
python scripts/train_cpt_unsloth.py

# 3. SFT - Supervised Fine-Tuning (học giải bài theo format)
python scripts/train_sft_unsloth.py --cpt-path outputs/cpt_linear_eq/final

# 4. Đánh giá model
python scripts/evaluate.py --model-path outputs/sft_linear_eq/final --num-tests 50

# 5. Test model
python scripts/inference.py --model-path outputs/sft_linear_eq/final

# 6. Interactive mode
python scripts/inference.py --model-path outputs/sft_linear_eq/final --interactive
```

### Option 3: Google Colab Notebook

Mở file `notebooks/finetune_qwen3_linear_eq.py` trong Colab - tất cả đã self-contained trong 1 file.

## Pipeline

```
┌──────────────┐     ┌─────────────┐     ┌─────────────┐     ┌───────────┐
│  Data Gen    │────▶│    CPT      │────▶│    SFT      │────▶│ Inference │
│  (API/Local) │     │  (LoRA)     │     │  (LoRA)     │     │           │
└──────────────┘     └─────────────┘     └─────────────┘     └───────────┘
 150 CPT samples      Learn math          Learn format         Solve
 150 SFT samples      vocabulary          step-by-step         equations!
```

| Stage | Mô tả | Output |
|-------|--------|--------|
| Data Gen | Sinh data bằng Gemini/OpenAI/Local | `data/raw/*.jsonl` |
| CPT | LoRA train trên corpus toán (next-token prediction) | `outputs/cpt_linear_eq/final` |
| SFT | LoRA train trên instruction-response pairs | `outputs/sft_linear_eq/final` |
| Evaluate | Đánh giá Exact Match, Format, Step Correctness | Report |
| Inference | Load model và giải phương trình | Kết quả step-by-step |

## Training Config

| Parameter | CPT | SFT |
|-----------|-----|-----|
| Base model | Qwen/Qwen3-0.6B | Qwen3-0.6B + CPT adapter |
| LoRA rank | 32 | 32 |
| LoRA alpha | 64 | 64 |
| Learning rate | 2e-4 | 1e-4 |
| Epochs | 3 | 5 |
| Batch size (effective) | 16 | 16 |
| Quantization | 4-bit (QLoRA) | 4-bit (QLoRA) |
| Packing | Yes | No |

## Data Format

**CPT data** (`data/raw/cpt_linear_equations.jsonl`):
```json
{"text": "Xét phương trình 3x - 5 = 16. Chuyển -5 sang vế phải ta được 3x = 21. Chia cả hai vế cho 3, ta được x = 7."}
```

**SFT data** (`data/raw/sft_linear_equations.jsonl`):
```json
{
  "instruction": "Giải phương trình: 3x - 5 = 16",
  "output": "Ta có:\n3x - 5 = 16\n3x = 21\nx = 7\nĐáp án: x = 7"
}
```

## Evaluation

Chạy evaluation trên 50 phương trình test (sinh ngẫu nhiên, khác seed với training data):

```bash
python scripts/evaluate.py --model-path outputs/sft_linear_eq/final --num-tests 50 --verbose
```

Output:

```
══════════════════════════════════════════════════════════════
EVALUATION REPORT
══════════════════════════════════════════════════════════════

  Metric                    Score      Count
  ──────────────────────────────────────────────────────
  Exact Match                92.0%     46/50
  Format Compliance          96.0%     48/50
  Step 1 (chuyển vế)         90.0%     45/50
  Step 2 (tìm x)            92.0%     46/50
  All Steps Correct          88.0%     44/50
```

| Metric | Đo gì | Target |
|--------|--------|--------|
| **Exact Match** | Đáp án cuối cùng đúng/sai | > 80% |
| **Format Compliance** | Output đúng format (Ta có:... Đáp án:...) | > 90% |
| **Step 1** | Bước chuyển vế đúng (ax = c-b) | > 70% |
| **Step 2** | Bước tìm x đúng (x = kết quả) | > 80% |
| **All Steps** | Tất cả bước trung gian đúng | > 70% |

## Tech Stack

| Component | Tools |
|-----------|-------|
| Base Model | Qwen/Qwen3-0.6B |
| Training Framework | Unsloth, TRL (SFTTrainer) |
| LoRA | PEFT, 4-bit QLoRA |
| Data Generation | Gemini API / OpenAI API / Local |
| Evaluation | Exact Match, Format Check, Step Correctness |
| Quantization | bitsandbytes (4-bit NF4) |