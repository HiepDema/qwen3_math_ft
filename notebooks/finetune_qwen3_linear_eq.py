# %% [markdown]
# # Fine-tune Qwen3-0.6B: Giải phương trình bậc nhất
#
# Pipeline: **Data Generation → CPT → SFT → Inference**
#
# - Model: Qwen/Qwen3-0.6B
# - Method: LoRA via Unsloth
# - Task: Giải phương trình bậc nhất 1 ẩn (tiếng Việt)

# %% [markdown]
# ## 1. Install Dependencies

# %%
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install torch transformers datasets trl peft accelerate bitsandbytes xformers
# !pip install google-generativeai  # for data generation

# %% [markdown]
# ## 2. Generate Training Data

# %%
import json
import random
from fractions import Fraction
from pathlib import Path

random.seed(42)


def generate_linear_equation():
    a = random.choice([i for i in range(-10, 11) if i != 0])
    b = random.randint(-20, 20)
    c = random.randint(-30, 30)
    return a, b, c


def format_equation(a, b, c):
    if a == 1:
        left = "x"
    elif a == -1:
        left = "-x"
    else:
        left = f"{a}x"

    if b > 0:
        left += f" + {b}"
    elif b < 0:
        left += f" - {abs(b)}"

    return f"{left} = {c}"


def solve_equation_steps(a, b, c):
    eq_str = format_equation(a, b, c)
    steps = [f"Ta có:\n{eq_str}"]

    rhs = c - b
    if a == 1:
        lhs = "x"
    elif a == -1:
        lhs = "-x"
    else:
        lhs = f"{a}x"
    steps.append(f"{lhs} = {rhs}")

    if a != 1 and a != -1:
        if rhs % a == 0:
            x = rhs // a
        else:
            frac = Fraction(rhs, a)
            x = f"{frac.numerator}/{frac.denominator}"
        steps.append(f"x = {rhs}/{a}")
        if isinstance(x, int):
            steps.append(f"x = {x}")
        else:
            steps.append(f"x = {x}")
    elif a == -1:
        x = -rhs
        steps.append(f"x = {x}")
    else:
        x = rhs

    return "\n".join(steps) + f"\nĐáp án: x = {x}"


# Generate CPT data
cpt_theory = [
    "Phương trình bậc nhất một ẩn có dạng tổng quát ax + b = 0, trong đó a và b là các hằng số, a khác 0, và x là ẩn số cần tìm. Nghiệm của phương trình là x = -b/a.",
    "Để giải phương trình bậc nhất, ta thực hiện các bước: chuyển vế các hạng tử chứa ẩn sang một vế, các hằng số sang vế kia, sau đó chia cả hai vế cho hệ số của ẩn.",
    "Phương trình bậc nhất một ẩn luôn có đúng một nghiệm duy nhất khi hệ số a khác 0.",
    "Khi giải phương trình, ta có thể cộng hoặc trừ cùng một số vào hai vế mà không thay đổi tập nghiệm.",
    "Quy tắc chuyển vế: Khi chuyển một hạng tử từ vế này sang vế kia của phương trình, ta phải đổi dấu hạng tử đó.",
    "Quy tắc nhân: Ta có thể nhân cả hai vế của phương trình với cùng một số khác 0 mà không thay đổi nghiệm.",
    "Phương trình bậc nhất có ứng dụng rộng rãi trong thực tế: tính tuổi, tính quãng đường, tính giá tiền.",
    "Nghiệm của phương trình bậc nhất ax + b = c là x = (c - b)/a. Điều kiện để phương trình có nghiệm duy nhất là a ≠ 0.",
    "Phương trình tương đương là các phương trình có cùng tập nghiệm.",
    "Trong toán học, phương trình bậc nhất là nền tảng để học các loại phương trình phức tạp hơn.",
]

cpt_data = []
for _ in range(150):
    choice = random.random()
    if choice < 0.15:
        cpt_data.append({"text": random.choice(cpt_theory)})
    elif choice < 0.6:
        a, b, c = generate_linear_equation()
        eq_str = format_equation(a, b, c)
        rhs = c - b
        if rhs % a == 0:
            x = rhs // a
            cpt_data.append({"text":
                f"Xét phương trình {eq_str}. "
                f"Chuyển {b} sang vế phải ta được {a}x = {rhs}. "
                f"Chia cả hai vế cho {a}, ta được x = {x}. "
                f"Vậy phương trình có nghiệm duy nhất x = {x}."
            })
        else:
            frac = Fraction(rhs, a)
            cpt_data.append({"text":
                f"Xét phương trình {eq_str}. "
                f"Chuyển {b} sang vế phải ta được {a}x = {rhs}. "
                f"Chia cả hai vế cho {a}, ta được x = {frac}. "
                f"Vậy nghiệm là x = {frac}."
            })
    else:
        a, b, c = generate_linear_equation()
        eq_str = format_equation(a, b, c)
        rhs = c - b
        cpt_data.append({"text":
            f"Phương pháp giải phương trình bậc nhất: "
            f"Cho phương trình {eq_str}. "
            f"Bước 1: Chuyển hạng tử tự do sang vế phải: {a}x = {rhs}. "
            f"Bước 2: Chia hai vế cho hệ số của x. "
            f"Kết quả: x = {rhs // a if rhs % a == 0 else Fraction(rhs, a)}."
        })

# Generate SFT data
sft_data = []
seen = set()
while len(sft_data) < 150:
    a, b, c = generate_linear_equation()
    key = (a, b, c)
    if key in seen:
        continue
    seen.add(key)
    sft_data.append({
        "instruction": f"Giải phương trình: {format_equation(a, b, c)}",
        "output": solve_equation_steps(a, b, c),
    })

print(f"Generated {len(cpt_data)} CPT samples")
print(f"Generated {len(sft_data)} SFT samples")

# Show samples
print("\n--- CPT Sample ---")
print(cpt_data[5]["text"])
print("\n--- SFT Sample ---")
print(f"Instruction: {sft_data[0]['instruction']}")
print(f"Output:\n{sft_data[0]['output']}")

# %% [markdown]
# ## 3. CPT - Continual Pre-Training

# %%
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_SEQ_LENGTH = 1024

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,
)

# Apply LoRA for CPT
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Prepare CPT dataset
cpt_dataset = Dataset.from_list(cpt_data)
print(f"CPT dataset: {len(cpt_dataset)} samples")

# %%
# Train CPT
cpt_trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=cpt_dataset,
    args=SFTConfig(
        output_dir="outputs/cpt_linear_eq",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=True,
        seed=42,
        optim="adamw_8bit",
    ),
)

print("Starting CPT...")
cpt_result = cpt_trainer.train()
print(f"CPT done! Loss: {cpt_result.metrics['train_loss']:.4f}")

# Save CPT checkpoint
model.save_pretrained("outputs/cpt_linear_eq/final")
tokenizer.save_pretrained("outputs/cpt_linear_eq/final")

# %% [markdown]
# ## 4. SFT - Supervised Fine-Tuning

# %%
# Reload model fresh and merge CPT adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,
)

# Load and merge CPT adapter
from peft import PeftModel
model = PeftModel.from_pretrained(model, "outputs/cpt_linear_eq/final")
model = model.merge_and_unload()
print("CPT adapter merged!")

# Apply new LoRA for SFT
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# %%
# Format SFT data as chat conversations
SYSTEM_PROMPT = "Bạn là trợ lý toán học. Hãy giải phương trình bậc nhất theo từng bước."

sft_formatted = []
for item in sft_data:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item["instruction"]},
        {"role": "assistant", "content": item["output"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    sft_formatted.append({"text": text})

sft_dataset = Dataset.from_list(sft_formatted)

# Split train/eval
split = sft_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
print(f"SFT Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# %%
# Train SFT
sft_trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=SFTConfig(
        output_dir="outputs/sft_linear_eq",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        eval_steps=25,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
        seed=42,
        optim="adamw_8bit",
    ),
)

print("Starting SFT...")
sft_result = sft_trainer.train()
eval_metrics = sft_trainer.evaluate()
print(f"SFT done! Train loss: {sft_result.metrics['train_loss']:.4f}, Eval loss: {eval_metrics['eval_loss']:.4f}")

# Save final model
model.save_pretrained("outputs/sft_linear_eq/final")
tokenizer.save_pretrained("outputs/sft_linear_eq/final")

# %% [markdown]
# ## 5. Inference - Test the Model

# %%
# Load for inference
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/sft_linear_eq/final",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,
)
FastLanguageModel.for_inference(model)

# %%
def solve_equation(equation_str: str) -> str:
    """Solve a linear equation using the fine-tuned model."""
    if not equation_str.startswith("Giải"):
        equation_str = f"Giải phương trình: {equation_str}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": equation_str},
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
    )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


# Test!
test_equations = [
    "Giải phương trình: 3x - 5 = 16",
    "Giải phương trình: 2x + 7 = 15",
    "Giải phương trình: -4x + 3 = -9",
    "Giải phương trình: 5x - 10 = 25",
    "Giải phương trình: 7x + 2 = 23",
]

print("=" * 50)
print("MODEL INFERENCE RESULTS")
print("=" * 50)

for eq in test_equations:
    print(f"\n{'─' * 40}")
    print(f"Input: {eq}")
    print(f"{'─' * 40}")
    result = solve_equation(eq)
    print(result)

# %% [markdown]
# ## 6. (Optional) Push to HuggingFace Hub
#
# ```python
# model.push_to_hub("your-username/qwen3-0.6b-linear-eq-solver", token="hf_...")
# tokenizer.push_to_hub("your-username/qwen3-0.6b-linear-eq-solver", token="hf_...")
# ```

# %% [markdown]
# ## 7. (Optional) Generate data with Gemini API
#
# If you want higher quality data, uncomment and run the cell below:
#
# ```python
# import google.generativeai as genai
#
# genai.configure(api_key="YOUR_GEMINI_API_KEY")
# gemini = genai.GenerativeModel("gemini-2.0-flash")
#
# # Generate better SFT data
# prompt = """Tạo 10 bài giải phương trình bậc nhất. Format:
# [{"instruction": "Giải phương trình: ...", "output": "Ta có:\n...\nĐáp án: x = ..."}]
# Trả về JSON only."""
#
# response = gemini.generate_content(prompt)
# extra_data = json.loads(response.text)
# sft_data.extend(extra_data)
# ```
