"""Prepare training data from VietJack crawl + local quadratic generation.

Sources:
1. VietJack crawled data (linear equations, needs cleaning)
2. Local deterministic generator (quadratic equations)

Output: unified SFT format JSONL files ready for training.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --num-quadratic 300 --output-dir data/raw
"""

import argparse
import json
import math
import random
import re
from fractions import Fraction
from pathlib import Path


# ============================================================
# Part 1: Clean VietJack data → SFT format
# ============================================================

NOISE_PATTERNS = [
    r"Quảng cáo",
    r"\(\d+k\) Xem Khóa học.*",
    r"HOT.*Shopee.*",
    r"Trang trước.*Trang sau",
    r"Xem thêm các.*",
    r"Giải bài tập lớp \d+.*",
    r"Dịch vụ nổi bật.*",
    r"CÔNG TY TNHH.*",
    r"Tầng \d+.*Hà Nội",
    r"Phone:.*",
    r"Email:.*",
    r"hotro@vietjack.*",
    r"Theo dõi chúng tôi.*",
    r"Đã có app VietJack.*",
    r"Nếu thấy hay.*",
    r"Loạt bài.*",
    r"Lý thuyết.*Bài tập.*",
    r"Giải sgk.*",
    r"Soạn văn.*",
    r"Lớp \d+ - (Kết nối|Chân trời|Cánh diều).*",
    r"Đề thi.*tài liệu.*",
    r"Bài giảng Powerpoint.*",
    r"Giáo án word.*",
    r"Chuyên đề dạy thêm.*",
    r"Trắc nghiệm đúng sai.*",
    r"xem tất cả",
    r"TÀI LIỆU CLC.*",
    r"Mã giảm giá.*",
    r"Combo \d+ khóa.*",
    r"Phòng luyện.*",
    r"Sale \d+%.*",
    r"2015 © All Rights Reserved.*",
    r"Người đại diện:.*",
    r"Số giấy chứng nhận.*",
    r"\d+ Đề thi cuối kì.*",
    r"Chính sách.*",
    r"Hình thức thanh toán",
    r"Liên hệ.*",
    r"Trang web chia sẻ.*",
    r"VietJack Official",
    r"Tổng đài hỗ trợ.*",
]


def clean_text(text):
    """Remove ads, navigation, and website noise from VietJack text."""
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_equation_pairs(text):
    """Extract (instruction, solution) pairs from VietJack text."""
    pairs = []
    text = clean_text(text)

    if not text or len(text) < 20:
        return pairs

    pattern = r"(?:Giải.*?phương trình[^:]*:|Phương trình\s+)(.*?)(?=\n)"
    eq_matches = re.finditer(
        r"([^\n]*?(?:\d+x|x\s*[+\-=])[^\n]*=\s*[^\n]+)",
        text
    )

    solution_pattern = re.compile(
        r"((?:.*?⇔.*?)+(?:x\s*=\s*[^\n]+|S\s*=\s*\{[^}]+\}|nghiệm[^\n]*))",
        re.DOTALL
    )

    blocks = re.split(r"(?=(?:Bài \d+|Ví dụ \d+|[a-d][,\)]))", text)

    for block in blocks:
        block = block.strip()
        if not block or len(block) < 10:
            continue

        eq_match = re.search(
            r"(?:Giải.*?phương trình.*?:?\s*\n?|[a-d][,\)]\s*)([\dx\s+\-=().]+(?:=\s*[\dx\s+\-().]+))",
            block
        )
        if not eq_match:
            continue

        equation = eq_match.group(1).strip()
        if "x" not in equation or "=" not in equation:
            continue

        solution_match = re.search(
            r"(?:Lời giải:?\s*\n?|⇔)(.*?)(?:Vậy|Phương trình có)",
            block, re.DOTALL
        )

        answer_match = re.search(
            r"(?:x\s*=\s*([^\n,⇔]+?)(?:\s*$|\s*\n)|S\s*=\s*\{\s*([^}]+)\})",
            block, re.MULTILINE
        )

        if answer_match:
            answer = (answer_match.group(1) or answer_match.group(2) or "").strip()
            if answer and len(answer) < 20:
                full_solution = ""
                steps = re.findall(r"⇔\s*([^\n⇔]+)", block)
                if steps:
                    full_solution = f"Ta có: {equation}\n"
                    full_solution += "\n".join(f"⇔ {s.strip()}" for s in steps)
                    full_solution += f"\nĐáp án: x = {answer}"

                    pairs.append({
                        "instruction": f"Giải phương trình: {equation}",
                        "output": full_solution,
                    })

    return pairs


def process_vietjack_data(input_path):
    """Process all VietJack data into SFT format."""
    all_pairs = []
    seen_instructions = set()

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                text = item.get("text", "")
                pairs = extract_equation_pairs(text)
                for pair in pairs:
                    key = pair["instruction"].lower().strip()
                    if key not in seen_instructions and len(pair["output"]) > 20:
                        seen_instructions.add(key)
                        all_pairs.append(pair)
            except json.JSONDecodeError:
                continue

    return all_pairs


# ============================================================
# Part 2: Generate quadratic equation data (local, no teacher)
# ============================================================

def is_perfect_square(n):
    if n < 0:
        return False
    root = math.isqrt(n)
    return root * root == n


def format_quadratic(a, b, c):
    """Format ax² + bx + c = 0."""
    parts = []
    if a == 1:
        parts.append("x²")
    elif a == -1:
        parts.append("-x²")
    else:
        parts.append(f"{a}x²")
    if b > 0:
        parts.append(f" + {b}x")
    elif b < 0:
        parts.append(f" - {abs(b)}x")
    if c > 0:
        parts.append(f" + {c}")
    elif c < 0:
        parts.append(f" - {abs(c)}")
    return "".join(parts) + " = 0"


def solve_quadratic(a, b, c):
    """Generate step-by-step solution for ax² + bx + c = 0."""
    eq_str = format_quadratic(a, b, c)
    delta = b * b - 4 * a * c

    steps = [f"Ta có phương trình: {eq_str}"]
    steps.append(f"Với a = {a}, b = {b}, c = {c}")
    steps.append(f"Tính delta: Δ = b² - 4ac = ({b})² - 4·({a})·({c}) = {b*b} - ({4*a*c}) = {delta}")

    if delta < 0:
        steps.append("Vì Δ < 0 nên phương trình vô nghiệm.")
        answer = "Phương trình vô nghiệm"
    elif delta == 0:
        x = Fraction(-b, 2 * a)
        steps.append("Vì Δ = 0 nên phương trình có nghiệm kép:")
        steps.append(f"x = -b/(2a) = -({b})/(2·{a}) = {-b}/{2*a} = {x}")
        answer = f"x = {x}"
    else:
        sqrt_delta = math.isqrt(delta) if is_perfect_square(delta) else None
        if sqrt_delta is not None:
            steps.append("Vì Δ > 0 nên phương trình có hai nghiệm phân biệt:")
            steps.append(f"√Δ = √{delta} = {sqrt_delta}")
            x1 = Fraction(-b + sqrt_delta, 2 * a)
            x2 = Fraction(-b - sqrt_delta, 2 * a)
            steps.append(f"x₁ = (-b + √Δ)/(2a) = ({-b} + {sqrt_delta})/{2*a} = {x1}")
            steps.append(f"x₂ = (-b - √Δ)/(2a) = ({-b} - {sqrt_delta})/{2*a} = {x2}")
            answer = f"x₁ = {x1}, x₂ = {x2}"
        else:
            steps.append("Vì Δ > 0 nên phương trình có hai nghiệm phân biệt:")
            steps.append(f"x₁ = ({-b} + √{delta})/{2*a}")
            steps.append(f"x₂ = ({-b} - √{delta})/{2*a}")
            answer = f"x₁ = ({-b} + √{delta})/{2*a}, x₂ = ({-b} - √{delta})/{2*a}"

    solution = "\n".join(steps) + f"\nĐáp án: {answer}"
    return solution


def generate_quadratic_data(num_samples=300, seed=42):
    """Generate quadratic equation SFT data."""
    random.seed(seed)
    samples = []
    seen = set()

    while len(samples) < num_samples:
        a = random.choice([i for i in range(-8, 9) if i != 0])
        b = random.randint(-20, 20)
        c = random.randint(-30, 30)
        key = (a, b, c)
        if key in seen:
            continue
        seen.add(key)

        eq_str = format_quadratic(a, b, c)
        solution = solve_quadratic(a, b, c)
        samples.append({
            "instruction": f"Giải phương trình bậc hai: {eq_str}",
            "output": solution,
        })

    return samples


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--vietjack-path", type=str,
                        default="data/raw/cpt_vietjack_crawled.jsonl")
    parser.add_argument("--num-quadratic", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default="data/raw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Process VietJack data
    vietjack_path = Path(args.vietjack_path)
    vietjack_samples = []
    if vietjack_path.exists():
        print(f"Processing VietJack data from: {vietjack_path}")
        vietjack_samples = process_vietjack_data(vietjack_path)
        print(f"  Extracted {len(vietjack_samples)} valid equation pairs")
    else:
        print(f"Warning: VietJack file not found at {vietjack_path}")

    # 2. Generate quadratic data
    print(f"\nGenerating {args.num_quadratic} quadratic equation samples...")
    quadratic_samples = generate_quadratic_data(args.num_quadratic, seed=args.seed)
    print(f"  Generated {len(quadratic_samples)} samples")

    # 3. Save separately
    if vietjack_samples:
        vj_path = output_dir / "sft_vietjack_clean.jsonl"
        with open(vj_path, "w", encoding="utf-8") as f:
            for s in vietjack_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"\nSaved VietJack data: {vj_path} ({len(vietjack_samples)} samples)")

    quad_path = output_dir / "sft_quadratic_equations.jsonl"
    with open(quad_path, "w", encoding="utf-8") as f:
        for s in quadratic_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Saved quadratic data: {quad_path} ({len(quadratic_samples)} samples)")

    # 4. Save combined
    all_samples = vietjack_samples + quadratic_samples
    random.seed(args.seed)
    random.shuffle(all_samples)

    combined_path = output_dir / "sft_combined.jsonl"
    with open(combined_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Saved combined data: {combined_path} ({len(all_samples)} samples)")

    # 5. Print stats
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"  VietJack (linear eq):     {len(vietjack_samples)} samples")
    print(f"  Quadratic (generated):    {len(quadratic_samples)} samples")
    print(f"  Combined total:           {len(all_samples)} samples")
    print()

    print("SAMPLES:")
    print("-" * 40)
    for s in all_samples[:3]:
        print(f"  Q: {s['instruction']}")
        print(f"  A: {s['output'][:100]}...")
        print()


if __name__ == "__main__":
    main()
