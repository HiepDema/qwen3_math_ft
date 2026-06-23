"""Generate training data for linear equation solving using Gemini API.

Generates two datasets:
- CPT data: Plain text about linear equations (math knowledge corpus)
- SFT data: Instruction-response pairs for solving linear equations

Usage:
    python scripts/generate_data.py --api gemini --api-key YOUR_KEY
    python scripts/generate_data.py --api openai --api-key YOUR_KEY
"""

import argparse
import json
import random
import time
from pathlib import Path


def generate_linear_equation():
    """Generate a random linear equation ax + b = c or ax - b = c."""
    a = random.choice([i for i in range(-10, 11) if i != 0])
    b = random.randint(-20, 20)
    c = random.randint(-30, 30)
    return a, b, c


def format_equation(a, b, c):
    """Format equation as string."""
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
    """Generate step-by-step solution for ax + b = c."""
    eq_str = format_equation(a, b, c)

    steps = [f"Ta có:\n{eq_str}"]

    # Step: move b to the right side
    rhs = c - b
    if a == 1:
        lhs = "x"
    elif a == -1:
        lhs = "-x"
    else:
        lhs = f"{a}x"

    steps.append(f"{lhs} = {rhs}")

    # Step: divide by a
    if rhs % a == 0:
        x = rhs // a
    else:
        x = rhs / a

    if a != 1 and a != -1:
        steps.append(f"x = {rhs}/{a}")
        if isinstance(x, int) or x == int(x):
            x = int(x)
            steps.append(f"x = {x}")
        else:
            # Keep as fraction for simplicity
            from fractions import Fraction
            frac = Fraction(rhs, a)
            x = f"{frac.numerator}/{frac.denominator}"
            steps.append(f"x = {x}")
    elif a == -1:
        x = -rhs
        steps.append(f"x = {x}")
    else:
        x = rhs
        # already have x = rhs

    solution = "\n".join(steps)
    answer = f"\nĐáp án: x = {x}"

    return solution + answer


def generate_sft_sample(a, b, c):
    """Generate an SFT training sample."""
    prompt = f"Giải phương trình: {format_equation(a, b, c)}"
    response = solve_equation_steps(a, b, c)
    return {"instruction": prompt, "output": response}


def generate_cpt_text_local(num_samples=150):
    """Generate CPT corpus text about linear equations (no API needed)."""
    texts = []

    # Type 1: Definitions and theory
    theory_templates = [
        "Phương trình bậc nhất một ẩn có dạng tổng quát ax + b = 0, trong đó a và b là các hằng số, a khác 0, và x là ẩn số cần tìm. Nghiệm của phương trình là x = -b/a.",
        "Để giải phương trình bậc nhất, ta thực hiện các bước: chuyển vế các hạng tử chứa ẩn sang một vế, các hằng số sang vế kia, sau đó chia cả hai vế cho hệ số của ẩn.",
        "Phương trình bậc nhất một ẩn luôn có đúng một nghiệm duy nhất khi hệ số a khác 0. Đây là tính chất quan trọng nhất của phương trình bậc nhất.",
        "Khi giải phương trình, ta có thể cộng hoặc trừ cùng một số vào hai vế mà không thay đổi tập nghiệm. Tương tự, ta có thể nhân hoặc chia hai vế cho cùng một số khác 0.",
        "Phương trình tương đương là các phương trình có cùng tập nghiệm. Trong quá trình giải phương trình, ta biến đổi phương trình đã cho thành các phương trình tương đương đơn giản hơn.",
        "Quy tắc chuyển vế: Khi chuyển một hạng tử từ vế này sang vế kia của phương trình, ta phải đổi dấu hạng tử đó. Ví dụ: nếu 2x + 3 = 7, chuyển 3 sang vế phải ta được 2x = 7 - 3 = 4.",
        "Quy tắc nhân: Ta có thể nhân cả hai vế của phương trình với cùng một số khác 0 mà không thay đổi nghiệm. Ví dụ: x/2 = 3 nhân hai vế với 2 ta được x = 6.",
        "Phương trình bậc nhất có ứng dụng rộng rãi trong thực tế: tính tuổi, tính quãng đường, tính giá tiền, chia tỷ lệ, và nhiều bài toán khác.",
        "Trong toán học, phương trình bậc nhất là nền tảng để học các loại phương trình phức tạp hơn như phương trình bậc hai, hệ phương trình, và bất phương trình.",
        "Nghiệm của phương trình bậc nhất ax + b = c là x = (c - b)/a. Điều kiện để phương trình có nghiệm duy nhất là a ≠ 0.",
    ]

    # Type 2: Worked examples as prose
    for _ in range(num_samples):
        choice = random.random()

        if choice < 0.15:
            texts.append(random.choice(theory_templates))
        elif choice < 0.5:
            # Example with explanation
            a, b, c = generate_linear_equation()
            eq_str = format_equation(a, b, c)
            rhs = c - b
            if rhs % a == 0:
                x = rhs // a
                texts.append(
                    f"Xét phương trình {eq_str}. "
                    f"Chuyển {b} sang vế phải ta được {a}x = {c} - {b} = {rhs}. "
                    f"Chia cả hai vế cho {a}, ta được x = {rhs}/{a} = {x}. "
                    f"Vậy phương trình có nghiệm duy nhất x = {x}."
                )
            else:
                from fractions import Fraction
                frac = Fraction(rhs, a)
                texts.append(
                    f"Xét phương trình {eq_str}. "
                    f"Chuyển {b} sang vế phải ta được {a}x = {rhs}. "
                    f"Chia cả hai vế cho {a}, ta được x = {frac}. "
                    f"Vậy nghiệm của phương trình là x = {frac}."
                )
        elif choice < 0.75:
            # Multiple equations in one paragraph
            eqs = []
            for _ in range(random.randint(2, 4)):
                a, b, c = generate_linear_equation()
                eq_str = format_equation(a, b, c)
                rhs = c - b
                if rhs % a == 0:
                    x = rhs // a
                    eqs.append(f"{eq_str} có nghiệm x = {x}")
            texts.append(
                "Giải các phương trình bậc nhất sau: " + "; ".join(eqs) + "."
            )
        else:
            # Method description with specific example
            a, b, c = generate_linear_equation()
            eq_str = format_equation(a, b, c)
            rhs = c - b
            texts.append(
                f"Phương pháp giải phương trình bậc nhất: "
                f"Cho phương trình {eq_str}. "
                f"Bước 1: Chuyển hạng tử tự do sang vế phải: {a}x = {rhs}. "
                f"Bước 2: Chia hai vế cho hệ số của x. "
                f"Kết quả: x = {rhs}/{a}"
                + (f" = {rhs // a}." if rhs % a == 0 else ".")
            )

    return texts


def generate_sft_data_local(num_samples=150):
    """Generate SFT data locally (no API needed)."""
    samples = []
    seen = set()

    while len(samples) < num_samples:
        a, b, c = generate_linear_equation()
        key = (a, b, c)
        if key in seen:
            continue
        seen.add(key)
        samples.append(generate_sft_sample(a, b, c))

    return samples


def generate_with_gemini(api_key, num_cpt=100, num_sft=100):
    """Generate data using Gemini API for higher quality/diversity."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    cpt_texts = []
    sft_samples = []

    # Generate CPT data
    print("Generating CPT data with Gemini...")
    cpt_prompt = """Hãy viết 10 đoạn văn ngắn (mỗi đoạn 2-4 câu) bằng tiếng Việt về chủ đề giải phương trình bậc nhất một ẩn.
Mỗi đoạn nên bao gồm:
- Lý thuyết về phương trình bậc nhất (dạng ax + b = c)
- Ví dụ minh họa với lời giải
- Các quy tắc biến đổi phương trình

Trả về dạng JSON array of strings. Ví dụ:
["đoạn 1...", "đoạn 2...", ...]

CHỈ trả về JSON, không có text thừa."""

    for i in range(num_cpt // 10):
        try:
            response = model.generate_content(cpt_prompt)
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            paragraphs = json.loads(text.strip())
            cpt_texts.extend(paragraphs)
            print(f"  Batch {i+1}: got {len(paragraphs)} paragraphs")
        except Exception as e:
            print(f"  Batch {i+1} failed: {e}")
        time.sleep(1)

    # Generate SFT data
    print("Generating SFT data with Gemini...")
    sft_prompt = """Hãy tạo 10 bài toán giải phương trình bậc nhất một ẩn với lời giải theo format sau:

Mỗi bài có:
- instruction: "Giải phương trình: [phương trình]"
- output: lời giải theo format:
  "Ta có:
  [phương trình gốc]
  [bước chuyển vế]
  [bước chia]
  x = [kết quả]
  Đáp án: x = [kết quả]"

Đa dạng các dạng: ax + b = c, ax - b = c, b - ax = c, với a,b,c là các số nguyên.

Trả về JSON array:
[{"instruction": "...", "output": "..."}, ...]

CHỈ trả về JSON, không có text thừa."""

    for i in range(num_sft // 10):
        try:
            response = model.generate_content(sft_prompt)
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            batch = json.loads(text.strip())
            sft_samples.extend(batch)
            print(f"  Batch {i+1}: got {len(batch)} samples")
        except Exception as e:
            print(f"  Batch {i+1} failed: {e}")
        time.sleep(1)

    return cpt_texts, sft_samples


def generate_with_openai(api_key, num_cpt=100, num_sft=100):
    """Generate data using OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    cpt_texts = []
    sft_samples = []

    # Generate CPT data
    print("Generating CPT data with OpenAI...")
    cpt_prompt = """Hãy viết 10 đoạn văn ngắn (mỗi đoạn 2-4 câu) bằng tiếng Việt về chủ đề giải phương trình bậc nhất một ẩn.
Mỗi đoạn nên bao gồm lý thuyết, ví dụ minh họa, hoặc quy tắc biến đổi phương trình.

Trả về dạng JSON array of strings. CHỈ trả về JSON."""

    for i in range(num_cpt // 10):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": cpt_prompt}],
                temperature=0.9,
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            paragraphs = json.loads(text.strip())
            cpt_texts.extend(paragraphs)
            print(f"  Batch {i+1}: got {len(paragraphs)} paragraphs")
        except Exception as e:
            print(f"  Batch {i+1} failed: {e}")
        time.sleep(0.5)

    # Generate SFT data
    print("Generating SFT data with OpenAI...")
    sft_prompt = """Tạo 10 bài toán giải phương trình bậc nhất một ẩn. Mỗi bài:
- instruction: "Giải phương trình: [phương trình]"
- output: "Ta có:\n[phương trình gốc]\n[các bước giải]\nx = [kết quả]\nĐáp án: x = [kết quả]"

Đa dạng dạng: ax + b = c, ax - b = c, etc. Trả về JSON array. CHỈ trả về JSON."""

    for i in range(num_sft // 10):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": sft_prompt}],
                temperature=0.9,
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            batch = json.loads(text.strip())
            sft_samples.extend(batch)
            print(f"  Batch {i+1}: got {len(batch)} samples")
        except Exception as e:
            print(f"  Batch {i+1} failed: {e}")
        time.sleep(0.5)

    return cpt_texts, sft_samples


def main():
    parser = argparse.ArgumentParser(description="Generate training data for linear equation solving")
    parser.add_argument("--api", choices=["gemini", "openai", "local"], default="local",
                        help="API to use for data generation")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--num-cpt", type=int, default=150, help="Number of CPT samples")
    parser.add_argument("--num-sft", type=int, default=150, help="Number of SFT samples")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.api == "gemini":
        assert args.api_key, "Gemini API key required (--api-key)"
        cpt_texts, sft_samples = generate_with_gemini(args.api_key, args.num_cpt, args.num_sft)
        # Supplement with local data if API didn't return enough
        if len(cpt_texts) < args.num_cpt:
            local_cpt = generate_cpt_text_local(args.num_cpt - len(cpt_texts))
            cpt_texts.extend(local_cpt)
        if len(sft_samples) < args.num_sft:
            local_sft = generate_sft_data_local(args.num_sft - len(sft_samples))
            sft_samples.extend(local_sft)
    elif args.api == "openai":
        assert args.api_key, "OpenAI API key required (--api-key)"
        cpt_texts, sft_samples = generate_with_openai(args.api_key, args.num_cpt, args.num_sft)
        if len(cpt_texts) < args.num_cpt:
            local_cpt = generate_cpt_text_local(args.num_cpt - len(cpt_texts))
            cpt_texts.extend(local_cpt)
        if len(sft_samples) < args.num_sft:
            local_sft = generate_sft_data_local(args.num_sft - len(sft_samples))
            sft_samples.extend(local_sft)
    else:
        print("Generating data locally (no API)...")
        cpt_texts = generate_cpt_text_local(args.num_cpt)
        sft_samples = generate_sft_data_local(args.num_sft)

    # Save CPT data
    cpt_path = output_dir / "cpt_linear_equations.jsonl"
    with open(cpt_path, "w", encoding="utf-8") as f:
        for text in cpt_texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"Saved {len(cpt_texts)} CPT samples to {cpt_path}")

    # Save SFT data
    sft_path = output_dir / "sft_linear_equations.jsonl"
    with open(sft_path, "w", encoding="utf-8") as f:
        for sample in sft_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved {len(sft_samples)} SFT samples to {sft_path}")

    # Print samples
    print("\n" + "=" * 60)
    print("SAMPLE CPT DATA:")
    print("=" * 60)
    for text in cpt_texts[:3]:
        print(f"\n{text}\n---")

    print("\n" + "=" * 60)
    print("SAMPLE SFT DATA:")
    print("=" * 60)
    for sample in sft_samples[:3]:
        print(f"\nInstruction: {sample['instruction']}")
        print(f"Output:\n{sample['output']}\n---")


if __name__ == "__main__":
    main()
