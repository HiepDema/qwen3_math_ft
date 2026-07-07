# Experiment Report: Fine-tuning Qwen3-0.6B for Math Reasoning
## Comparing CPT+SFT, SFT, SFT+GRPO (Dense), SFT+GRPO (Sparse)

---

## 1. Introduction

This report presents results from a comparative study of four fine-tuning strategies applied to **Qwen3-0.6B** (600M parameters) for elementary math reasoning tasks. The goal is to determine the most effective training pipeline among:

1. **CPT + SFT** — Continued Pre-Training on domain text, then Supervised Fine-Tuning
2. **SFT only** — Direct Supervised Fine-Tuning from base model
3. **SFT + GRPO (Dense Reward)** — SFT followed by RL with multi-signal reward
4. **SFT + GRPO (Sparse Reward)** — SFT followed by RL with binary correctness reward

All experiments use the same dataset split and evaluation protocol for fair comparison.

---

## 2. Dataset

**Source**: `DataMuncher-Labs/LSReasoning-15000` (HuggingFace)

| Property | Value |
|----------|-------|
| Total samples | 15,000 |
| Train split | 12,000 (80%) |
| Test split | 3,000 (20%) |
| Eval subset | 500 samples |
| Split seed | 42 |
| Language | English |

**Problem types**: Arithmetic (addition, subtraction, multiplication, division), linear equations, two-step equations, algebra, fractions, integer exponents, inequalities.

**Fields per sample**: `question`, `problem` (type), `how_to_solve` (approach), `answer` (ground truth).

### 2.1 Data Structure Example

Mỗi sample trong dataset có cấu trúc:

```json
{
  "question": "Solve for x: 3x + 7 = 22",
  "problem": "Solve the linear equation.",
  "how_to_solve": "Subtract 7 from both sides to get 3x = 15, then divide by 3.",
  "answer": "5"
}
```

```json
{
  "question": "What is 48 / 6?",
  "problem": "Divide evenly.",
  "how_to_solve": "Divide 48 by 6.",
  "answer": "8"
}
```

```json
{
  "question": "Reduce 18/24 to simplest form.",
  "problem": "Reduce the fraction to simplest form.",
  "how_to_solve": "Find GCD of 18 and 24, which is 6. Divide both by 6.",
  "answer": "3/4"
}
```

### 2.2 CPT Data Generation

CPT corpus được sinh từ 3 nguồn, tổng ~12,500 passages dạng **plain text** (không có chat template):

**Nguồn 1: Từ training data (~12,000 passages)**

Trích xuất `how_to_solve`, `question`, `answer` và format lại thành đoạn văn tự nhiên:

```text
Problem: Solve for x: 3x + 7 = 22
Solution approach: Subtract 7 from both sides to get 3x = 15, then divide by 3.
The answer is 5.
```

**Nguồn 2: Math theory passages (15 passages)**

Các đoạn lý thuyết toán viết sẵn, bao gồm:
- Arithmetic operations (addition, subtraction, multiplication, division)
- Order of operations (PEMDAS)
- Linear equation solving methods
- Fraction operations (add, subtract, multiply, divide, simplify)
- Word problem translation strategies
- Sign rules for negative numbers

Ví dụ:
```text
A linear equation has the form ax + b = c. To solve: isolate x by performing
inverse operations. Subtract b from both sides, then divide by a.
```

**Nguồn 3: Generated computation examples (500 passages)**

Sinh ngẫu nhiên 500 ví dụ tính toán theo 4 loại:

```text
Computing 25 + 17: the sum equals 42.
```
```text
Solving the equation 5x + 12 = 47: subtract 12 from both sides to get 5x = 35,
then divide by 5 to get x = 7.
```
```text
Simplifying 18/6: the GCD of 18 and 6 is 6, so 18/6 = 3/1.
```
```text
If you buy 7 items at $12 each, the total cost is 7 × $12 = $84.
```

Toàn bộ corpus được shuffle trước khi train, split 95/5 cho train/validation.

---

## 3. Methodology

### 3.1 Base Model

| Property | Value |
|----------|-------|
| Model | Qwen/Qwen3-0.6B |
| Parameters | 600M |
| Architecture | Transformer decoder-only, 28 layers, 14 attention heads |
| Vocabulary size | 151,936 tokens |
| Context length | 32,768 tokens (max), 1024 (used) |
| Quantization | 4-bit QLoRA (NF4, float16 compute) |
| Framework | Unsloth + TRL + PEFT |
| GPU | NVIDIA A10 (24GB VRAM) |

### 3.2 Configuration Comparison (4 Methods)

| Parameter | CPT+SFT | SFT only | SFT+GRPO Dense | SFT+GRPO Sparse |
|-----------|:-------:|:--------:|:--------------:|:---------------:|
| **Stage 1** | CPT | - | - | - |
| Data | ~12,500 plain text | - | - | - |
| Epochs | 2 | - | - | - |
| LR | 2e-4 | - | - | - |
| LoRA r/alpha | 16/32 | - | - | - |
| **Stage 2** | SFT | SFT | SFT | SFT |
| Base model | CPT merged | Qwen3-0.6B | Qwen3-0.6B | Qwen3-0.6B |
| Data | 12,000 chat pairs | 12,000 chat pairs | 12,000 chat pairs | 12,000 chat pairs |
| Epochs | 3 | 3 | 3 | 3 |
| LR | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| LoRA r/alpha | 32/64 | 32/64 | 32/64 | 32/64 |
| Batch size (eff.) | 16 | 16 | 16 | 16 |
| **Stage 3** | - | - | GRPO | GRPO |
| Base model | - | - | SFT merged | SFT merged |
| Prompts | - | - | 1,000 | 1,000 |
| Generations/prompt | - | - | 4 | 4 |
| Reward | - | - | Dense (4 signals) | Sparse (binary) |
| Epochs | - | - | 1 | 1 |
| LR | - | - | 5e-6 | 5e-6 |
| LoRA r/alpha | - | - | 16/32 | 16/32 |
| KL beta | - | - | 0.1 | 0.1 |
| Batch size (eff.) | - | - | 8 | 8 |
| **Common** | | | | |
| Optimizer | adamw_8bit | adamw_8bit | adamw_8bit | adamw_8bit |
| LR scheduler | cosine | cosine | cosine | cosine |
| Warmup ratio | 0.1 | 0.1 | 0.1 | 0.1 |
| Weight decay | 0.01 | 0.01 | 0.01 | 0.01 |
| Max seq length | 1024 | 1024 | 1024 | 1024 |
| Precision | fp16 | fp16 | fp16 | fp16 |
| Quantization | 4-bit QLoRA | 4-bit QLoRA | 4-bit QLoRA | 4-bit QLoRA |

### 3.3 Training Stages (Detail)

#### CPT (Continued Pre-Training)

Adapts the base model to the math domain using plain text (no instruction format).

| Hyperparameter | Value |
|----------------|-------|
| Corpus size | ~12,500 passages |
| Epochs | 2 |
| Learning rate | 2e-4 |
| Effective batch size | 16 |
| LoRA r / alpha | 16 / 32 |
| Max seq length | 1024 |
| Packing | Yes |

**Corpus composition**:
- ~12,000 passages derived from training data (`how_to_solve` + question + answer)
- 15 curated math theory passages (arithmetic, algebra, fractions, word problems)
- 500 programmatically generated computation examples

#### SFT (Supervised Fine-Tuning)

Trains the model to follow instructions using chat-formatted conversations.

| Hyperparameter | Value |
|----------------|-------|
| Training samples | 12,000 |
| Epochs | 3 |
| Learning rate | 1e-4 |
| Effective batch size | 16 |
| LoRA r / alpha | 32 / 64 |
| LoRA dropout | 0.05 |
| Max seq length | 1024 |
| Packing | Yes |
| Early stopping | Best eval_loss |

**Chat template**:
```
System: You are a math tutor. Solve the problem step by step, show your reasoning clearly, then give the final answer.
User: {question}
Assistant: Approach: {how_to_solve}
Solving step by step:
{question}
Answer: {answer}
```

#### GRPO (Group Relative Policy Optimization)

Reinforcement learning stage that generates multiple completions per prompt and updates based on relative reward.

| Hyperparameter | Value |
|----------------|-------|
| Training prompts | 1,000 |
| Generations per prompt | 4 |
| Epochs | 1 |
| Learning rate | 5e-6 |
| Effective batch size | 8 |
| KL penalty (beta) | 0.1 |
| LoRA r / alpha | 16 / 32 |
| Max completion length | 256 |

### 3.4 Reward Functions

#### Dense Reward (Multi-Signal)

```
R_dense = 0.4 × Correctness + 0.2 × Proximity + 0.2 × Format + 0.2 × Reasoning
```

| Component | Description | Range |
|-----------|-------------|-------|
| Correctness | Exact match of extracted answer | {0, 1} |
| Proximity | Partial credit based on relative error | [0, 1] |
| Format | Has "Answer:", multi-line, reasoning keywords, math operators | [0, 1] |
| Reasoning | Keyword overlap with reference approach + math expressions | [0, 1] |

#### Sparse Reward (Binary)

```
R_sparse = Correctness (0 or 1)
```

### 3.5 Design Patterns

- **Decreasing learning rate** across stages: CPT (2e-4) > SFT (1e-4) > GRPO (5e-6)
- **Decreasing LoRA rank** for later stages: SFT (r=32) > CPT/GRPO (r=16)
- **Model merging** between stages: LoRA adapters merged to float16 before next stage
- **Consistent dtype**: All stages use float16 to avoid Half/BFloat16 conflicts with 4-bit dequantization

### 3.6 Evaluation Protocol

#### Inference Setup

| Parameter | Value |
|-----------|-------|
| Test set | 500 samples (from 20% holdout) |
| Batch size | 16 |
| Max new tokens | 256 |
| Decoding | Greedy (do_sample=False) |
| Quantization | 4-bit (same as training) |

#### Metrics

**Exact Match**: Trích xuất số từ output model, so sánh với ground truth.

Logic trích xuất:
1. Tìm pattern `Answer: <number>` trong response
2. Nếu không có, lấy số cuối cùng trong response
3. Hỗ trợ số nguyên, thập phân, phân số (ví dụ `3/4` → 0.75)
4. So sánh: `|predicted - true| < 1e-6` → exact match

**Close Match**: Tương tự Exact Match nhưng cho phép sai số 1%:
- `|predicted - true| / |true| < 0.01` → close match

**Format Score**: Tỷ lệ response có chứa "Answer:" hoặc "answer:" — đánh giá model có tuân thủ format output không.

**Reasoning Score**: Tỷ lệ response có ≥ 3 dòng non-empty — đánh giá model có show reasoning steps không.

#### Example Evaluation Flow

```
Input:  "Solve for x: 5x - 3 = 12"
Output: "Approach: Add 3 to both sides, then divide by 5.
         Solving step by step:
         5x - 3 = 12
         5x = 15
         x = 3
         Answer: 3"

Extract number: 3
Ground truth: 3
|3 - 3| < 1e-6 → Exact Match ✓
Contains "Answer:" → Format ✓
≥ 3 lines → Reasoning ✓
```

#### Batch Inference

Evaluation sử dụng **batch inference** (batch_size=16) thay vì sequential generation:
- Left-padding với pad_token = eos_token
- `torch.no_grad()` để giảm memory
- Progress log mỗi 5 batches
- ~500 samples hoàn thành trong ~5 phút trên A10

---

## 4. Training Curves

### 4.1 CPT Training Loss

![CPT Training Loss](figures/cpt_train_loss.png)

CPT loss giảm nhanh trong 2 epochs trên domain text — model nhanh chóng absorb math vocabulary và patterns.

### 4.2 SFT Training Loss

![SFT Training Loss Comparison](figures/sft_train_loss.png)

So sánh SFT loss giữa 2 pipelines:
- **CPT+SFT** (green): Bắt đầu từ CPT checkpoint, loss khởi đầu thấp hơn do đã adapted domain
- **SFT only** (blue): Bắt đầu từ base model, loss khởi đầu cao hơn

### 4.3 GRPO Training Loss & Reward

![GRPO Training Loss and Reward](figures/grpo_train_loss_reward.png)

- **Left**: GRPO training loss cho Dense vs Sparse reward
- **Right**: Mean reward progression — Dense reward cung cấp gradient phong phú hơn, reward tăng nhanh hơn

---

## 5. Results

### 5.1 Overall Performance

| Method | Exact Match | Close Match | Format | Reasoning |
|--------|:-----------:|:-----------:|:------:|:---------:|
| **CPT + SFT** | **85.2%** | **85.4%** | 100% | 100% |
| SFT only | 83.6% | 83.8% | 100% | 100% |
| SFT + GRPO (dense) | 81.8% | 82.0% | 100% | 100% |
| SFT + GRPO (sparse) | 79.4% | 79.8% | 100% | 100% |

**Winner: CPT + SFT** with 85.2% exact match, outperforming all other methods.

![Evaluation Comparison](figures/eval_comparison.png)

### 5.2 Performance by Problem Type

![Accuracy by Problem Type](figures/accuracy_by_type.png)

| Problem Type | CPT+SFT | SFT | GRPO Dense | GRPO Sparse |
|-------------|:-------:|:---:|:----------:|:-----------:|
| Add two numbers | 100% | 100% | 100% | 100% |
| Subtract numbers | 100% | 100% | 100% | 100% |
| Multiply numbers | 96.1% | 94.1% | 96.1% | 96.1% |
| Divide evenly | 100% | 100% | 97.8% | 97.8% |
| Reduce fractions | 100% | 100% | 100% | 100% |
| Integer exponents | 100% | 100% | 100% | 97.7% |
| Linear equations | **94.4%** | 87.0% | 85.2% | 85.2% |
| Two-step equations | **63.5%** | 58.7% | 49.2% | 28.6% |
| Solve using algebra | 100% | 100% | 94.1% | 100% |
| Solve inequalities | 0% | 0% | 0% | 0% |

### 5.3 Key Observations

**1. CPT + SFT is the best method (+1.6% over SFT, +3.4% over GRPO dense)**

The continued pre-training stage provides domain adaptation that improves downstream fine-tuning. The gains are most pronounced on complex problem types:
- Linear equations: CPT+SFT 94.4% vs SFT 87.0% (+7.4%)
- Two-step equations: CPT+SFT 63.5% vs SFT 58.7% (+4.8%)

**2. GRPO does NOT improve over SFT in this setting**

Contrary to expectations, both GRPO variants performed worse than SFT alone:
- SFT+GRPO (dense): -1.8% vs SFT
- SFT+GRPO (sparse): -4.2% vs SFT

**3. Dense reward > Sparse reward (within GRPO)**

Dense GRPO (81.8%) outperforms Sparse GRPO (79.4%), confirming that multi-signal feedback provides more useful gradient signal than binary correctness alone.

**4. Two-step equations are the differentiator**

This is the hardest non-trivial type and shows the clearest separation:
- CPT+SFT: 63.5% → SFT: 58.7% → GRPO dense: 49.2% → GRPO sparse: 28.6%

GRPO sparse catastrophically degrades on multi-step reasoning — the binary reward provides no learning signal for partial progress.

**5. Format and reasoning are saturated**

All methods achieve 100% format and reasoning scores, indicating that even basic SFT is sufficient to teach output structure for this dataset.

**6. Inequalities remain unsolved**

All methods score 0% on inequalities — this problem type likely requires fundamentally different reasoning patterns not present in the training data.

---

## 6. Analysis: Why GRPO Underperforms

The GRPO results contradict findings from larger-scale RL papers. Several factors explain this:

### 6.1 Small Model Capacity (0.6B)

A 600M parameter model has limited capacity for the explore-exploit trade-off in RL. GRPO requires generating diverse completions to discover better solutions, but a small model may not have sufficient representational power to benefit from this.

### 6.2 Already-High SFT Baseline

The SFT model already achieves 83.6% — most "easy" problems are solved. GRPO's exploration primarily affects the remaining 16.4% of hard problems, but the generation quality for these problems may be too poor to provide meaningful reward signal.

### 6.3 Limited GRPO Training (1000 prompts)

With only 1000 prompts × 4 generations × 1 epoch, the RL stage may not have sufficient training signal to improve without overfitting. The model may be "unlearning" some SFT knowledge during the RL phase.

### 6.4 Reward Function Limitations

The dense reward's proximity and reasoning components may introduce noise. For example, a wrong answer that is numerically close to the correct one receives partial credit, potentially reinforcing incorrect reasoning paths.

---

## 7. Inference Optimization

### KV Cache Benchmark

Tested on the best model (CPT+SFT) with 50 samples, batch_size=8:

| Strategy | Time (s) | Throughput (tok/s) | Speedup |
|----------|:--------:|:-----------------:|:-------:|
| Dynamic KV Cache (baseline) | 17.53 | 730.1 | 1.00x |
| Static KV Cache | 16.80 | 761.7 | **1.04x** |

The Static KV Cache provides a modest 4% speedup by pre-allocating cache memory and avoiding dynamic reallocation overhead. The improvement is relatively small because:
- The 0.6B model has a small KV cache to begin with
- Batch inference already amortizes overhead effectively
- The bottleneck is generation (autoregressive decoding), not cache management

For production deployment, additional optimizations to consider:
- **torch.compile**: JIT compilation of forward pass (additional 10-20% speedup after warmup)
- **vLLM/TGI**: Dedicated serving frameworks with continuous batching
- **Quantized inference**: INT8/INT4 without LoRA overhead (merge weights permanently)

---

## 8. Conclusions & Recommendations

### Key Findings

1. **CPT + SFT is the most effective pipeline** for adapting a small LLM to a specific math domain. Domain pre-training on plain text passages before instruction tuning provides measurable gains (+1.6% overall, +7.4% on linear equations).

2. **GRPO is not beneficial at this scale/setting**. For a 0.6B model with high SFT baseline and limited RL budget, reinforcement learning degrades performance rather than improving it.

3. **Dense reward mitigates GRPO damage** but cannot overcome the fundamental scaling limitation. The multi-signal reward still outperforms binary correctness within the GRPO framework.

4. **Output format is easy to learn** — all methods saturate format/reasoning scores, suggesting this is not a meaningful differentiator for this dataset.

### Recommended Pipeline

For production deployment of Qwen3-0.6B on math reasoning:

```
Qwen3-0.6B → CPT (2 epochs, 2e-4 LR) → SFT (3 epochs, 1e-4 LR)
```

Skip GRPO unless:
- Using a larger model (>=7B) where RL exploration is more effective
- Having significantly more RL training budget (>5000 prompts, multiple epochs)
- The SFT baseline is low (<70%), leaving more room for RL improvement

### Future Work

- Test CPT+SFT+GRPO pipeline (GRPO on top of CPT+SFT model)
- Increase GRPO budget (5000+ prompts, 2-3 epochs) to test if longer training helps
- Try DPO (Direct Preference Optimization) as a simpler RL alternative
- Scale to larger models (Qwen3-1.7B, 4B) where GRPO may be more effective
- Address inequality solving with targeted data augmentation

---

## 9. Reproducibility

### Environment

| Component | Version |
|-----------|---------|
| GPU | NVIDIA A10 (24GB) |
| Framework | Unsloth + TRL |
| Model | Qwen/Qwen3-0.6B |
| Quantization | 4-bit QLoRA (float16) |
| Python | 3.10 |

### Commands

```bash
# Full experiment (4 methods)
python scripts/run_experiment_lsreasoning.py

# Individual steps
python scripts/train_cpt_lsreasoning.py --train-file data/lsreasoning_split/train.jsonl
python scripts/train_sft_lsreasoning_v2.py --train-file data/lsreasoning_split/train.jsonl
python scripts/train_grpo_lsreasoning_v2.py --reward-mode dense --sft-path outputs/sft_lsreasoning/final
python scripts/evaluate_lsreasoning_v2.py --model-path outputs/cpt_sft_lsreasoning/final --test-file data/lsreasoning_split/test.jsonl

# Inference benchmark
python scripts/inference_optimized.py --model-path outputs/cpt_sft_lsreasoning/final --benchmark
```

### Tracking

- **wandb project**: [lsreasoning-sft-vs-grpo](https://wandb.ai/hiep26-sdf/lsreasoning-sft-vs-grpo)
- **Data split**: `data/lsreasoning_split/` (seed=42, 80/20)
- **Results**: `outputs/experiment_results.json`

---

## Appendix: Training Time

| Stage | Duration (est.) |
|-------|:---------:|
| CPT | ~15 min |
| SFT (CPT+SFT) | ~30 min |
| SFT (standalone) | ~30 min |
| GRPO dense (1000 prompts) | ~15 min |
| GRPO sparse (1000 prompts) | ~15 min |
| Evaluation (4 models × 500 samples) | ~20 min |
| **Total** | **~2 hours** |
