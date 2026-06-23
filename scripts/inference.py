"""Inference script to test the fine-tuned model on linear equations.

Usage:
    python scripts/inference.py
    python scripts/inference.py --model-path outputs/sft_linear_eq/final
    python scripts/inference.py --interactive
"""

import argparse

from unsloth import FastLanguageModel


SYSTEM_PROMPT = "Bạn là trợ lý toán học. Hãy giải phương trình bậc nhất theo từng bước."

TEST_EQUATIONS = [
    "Giải phương trình: 3x - 5 = 16",
    "Giải phương trình: 2x + 7 = 15",
    "Giải phương trình: -4x + 3 = -9",
    "Giải phương trình: 5x - 10 = 25",
    "Giải phương trình: 7x + 2 = 23",
    "Giải phương trình: -2x + 8 = 0",
    "Giải phương trình: 6x - 18 = 0",
    "Giải phương trình: 4x + 12 = 4",
]


def load_model(model_path: str, max_seq_length: int = 1024):
    """Load the fine-tuned model."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_response(model, tokenizer, question: str, max_new_tokens: int = 256):
    """Generate a response for a given question."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="outputs/sft_linear_eq/final")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = load_model(args.model_path, args.max_seq_length)
    print("Model loaded!\n")

    if args.interactive:
        print("=" * 60)
        print("Interactive Mode - Nhập phương trình để giải (quit để thoát)")
        print("=" * 60)
        while True:
            question = input("\nNhập phương trình (vd: 3x - 5 = 16): ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question.startswith("Giải"):
                question = f"Giải phương trình: {question}"
            response = generate_response(model, tokenizer, question)
            print(f"\n{response}")
    else:
        print("=" * 60)
        print("Testing model on sample equations")
        print("=" * 60)
        for eq in TEST_EQUATIONS:
            print(f"\n{'─' * 40}")
            print(f"Input: {eq}")
            print(f"{'─' * 40}")
            response = generate_response(model, tokenizer, eq)
            print(f"Output:\n{response}")


if __name__ == "__main__":
    main()
