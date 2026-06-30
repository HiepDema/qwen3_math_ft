"""Crawl math exercises from VietJack for linear equation training data.

Crawls multiple pages about solving linear equations, extracts exercises
and solutions, then saves as JSONL compatible with the training pipeline.

Usage:
    python scripts/crawl_vietjack.py
    python scripts/crawl_vietjack.py --output-dir data/raw --delay 2
"""

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


URLS = [
    "https://vietjack.com/toan-lop-8/cach-giai-phuong-trinh-bac-nhat-mot-an-cuc-hay-co-dap-an.jsp",
    "https://vietjack.com/toan-lop-8/cach-giai-phuong-trinh-dua-duoc-ve-dang-ax-b-0-cuc-hay.jsp",
    "https://vietjack.com/toan-lop-8/cach-giai-phuong-trinh-tich-cuc-hay-co-dap-an.jsp",
    "https://vietjack.com/toan-lop-8/chuong-3-phuong-trinh-bac-nhat-mot-an.jsp",
    "https://vietjack.com/toan-lop-8/bai-tap-phuong-trinh-dua-duoc-ve-dang-ax-b-0.jsp",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
}


def fetch_page(url, delay=2):
    """Fetch a page with retry logic."""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            time.sleep(delay)
            return resp.text
        except requests.RequestException as e:
            print(f"  Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < 2:
                time.sleep(delay * 2)
    return None


def clean_text(text):
    """Clean extracted text."""
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" ", " ")
    return text


def extract_math_content(soup):
    """Extract math exercises and solutions from a VietJack page."""
    content_div = soup.find("div", class_="content-detail") or soup.find("div", id="content")
    if not content_div:
        content_div = soup.find("article") or soup.find("div", class_="post-content")
    if not content_div:
        content_div = soup.body

    exercises = []
    current_exercise = None
    current_solution = []

    elements = content_div.find_all(["p", "div", "h2", "h3", "h4", "strong", "span", "table"])

    for elem in elements:
        text = elem.get_text(separator=" ", strip=True)
        if not text:
            continue

        is_exercise_header = bool(
            re.search(r"(Bài\s*\d+|Ví\s*dụ\s*\d+|Câu\s*\d+|Bài\s*tập\s*\d+)", text, re.IGNORECASE)
        )
        has_equation = bool(
            re.search(r"[0-9]*x\s*[\+\-\=]|phương trình|giải", text, re.IGNORECASE)
        )

        if is_exercise_header or (has_equation and "Giải" in text and len(text) < 200):
            if current_exercise and current_solution:
                exercises.append({
                    "question": current_exercise,
                    "solution": "\n".join(current_solution),
                })
            current_exercise = text
            current_solution = []
        elif current_exercise:
            current_solution.append(text)

    if current_exercise and current_solution:
        exercises.append({
            "question": current_exercise,
            "solution": "\n".join(current_solution),
        })

    return exercises


def extract_equations_from_text(text):
    """Extract individual equations from text using regex patterns."""
    patterns = [
        r"(\-?\d*x\s*[\+\-]\s*\d+\s*=\s*\-?\d+)",
        r"(\-?\d+\s*[\+\-]\s*\d*x\s*=\s*\-?\d+)",
        r"(\-?\d*x\s*[\+\-]\s*\d*x\s*[\+\-]?\s*\d*\s*=\s*\-?\d*x?\s*[\+\-]?\s*\d*)",
        r"(\-?\d*x\s*=\s*\-?\d+)",
    ]
    equations = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        equations.extend(matches)
    return equations


def parse_exercises_structured(html_content):
    """Parse exercises with a more structured approach for VietJack pages."""
    soup = BeautifulSoup(html_content, "html.parser")
    exercises = []

    content = soup.find("div", class_="content-detail") or soup.find("div", id="content")
    if not content:
        content = soup.body
    if not content:
        return exercises

    full_text = content.get_text(separator="\n", strip=True)
    blocks = re.split(r"(Bài\s*\d+[.:)]|Ví\s*dụ\s*\d+[.:)]|Câu\s*\d+[.:)])", full_text)

    i = 1
    while i < len(blocks):
        header = blocks[i].strip() if i < len(blocks) else ""
        body = blocks[i + 1].strip() if i + 1 < len(blocks) else ""
        i += 2

        if not body:
            continue

        lines = body.split("\n")
        question_lines = []
        solution_lines = []
        in_solution = False

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if re.search(r"(Lời giải|Hướng dẫn|Giải|Ta có|Đáp án)", line, re.IGNORECASE):
                in_solution = True
            if in_solution:
                solution_lines.append(line)
            else:
                question_lines.append(line)

        if not question_lines and solution_lines:
            question_lines = solution_lines[:1]
            solution_lines = solution_lines[1:]

        question = f"{header} " + " ".join(question_lines)
        solution = "\n".join(solution_lines)

        if question and solution:
            exercises.append({
                "question": clean_text(question),
                "solution": solution.strip(),
            })

    return exercises


def clean_ads_and_noise(text):
    """Remove ad text, noise, and fix common issues."""
    text = re.sub(r"Quảng cáo\s*", "", text)
    text = re.sub(r"Xem thêm.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"Click để xem.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"Tham khảo.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def convert_to_sft_format(exercise):
    """Convert a crawled exercise to SFT instruction-output format."""
    question = exercise["question"]
    solution = exercise["solution"]

    question = clean_ads_and_noise(question)
    solution = clean_ads_and_noise(solution)

    question = re.sub(r"^(Bài\s*\d+[.:)]\s*|Ví\s*dụ\s*\d+[.:)]\s*|Câu\s*\d+[.:)]\s*)", "", question)

    if "phương trình" in question.lower() or "giải" in question.lower():
        instruction = question
    else:
        instruction = f"Giải phương trình: {question}"

    instruction = clean_text(instruction)
    if not instruction.endswith("?") and not instruction.endswith("."):
        instruction = instruction.rstrip()

    return {
        "instruction": instruction,
        "output": solution,
    }


def convert_to_cpt_format(exercise):
    """Convert a crawled exercise to CPT text format."""
    question = exercise["question"]
    solution = exercise["solution"]
    return f"{question}\n{solution}"


def crawl_all(urls, delay=2):
    """Crawl all URLs and extract exercises."""
    all_exercises = []

    for i, url in enumerate(urls):
        print(f"[{i+1}/{len(urls)}] Crawling: {url}")
        html = fetch_page(url, delay=delay)
        if not html:
            print(f"  FAILED to fetch")
            continue

        exercises_structured = parse_exercises_structured(html)

        soup = BeautifulSoup(html, "html.parser")
        exercises_content = extract_math_content(soup)

        seen_questions = set()
        page_exercises = []

        for ex in exercises_structured + exercises_content:
            q_key = re.sub(r"\s+", "", ex["question"])[:80]
            if q_key not in seen_questions and len(ex["solution"]) > 10:
                seen_questions.add(q_key)
                page_exercises.append(ex)

        all_exercises.extend(page_exercises)
        print(f"  Extracted {len(page_exercises)} exercises")

    return all_exercises


def main():
    parser = argparse.ArgumentParser(description="Crawl VietJack math exercises")
    parser.add_argument("--output-dir", type=str, default="data/raw",
                        help="Output directory for crawled data")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Delay between requests (seconds)")
    parser.add_argument("--urls-file", type=str, default=None,
                        help="File with additional URLs to crawl (one per line)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    urls = list(URLS)
    if args.urls_file:
        with open(args.urls_file) as f:
            extra_urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            urls.extend(extra_urls)

    print(f"Crawling {len(urls)} pages from VietJack...")
    print("=" * 60)

    exercises = crawl_all(urls, delay=args.delay)

    print("=" * 60)
    print(f"Total exercises extracted: {len(exercises)}")

    sft_samples = []
    cpt_texts = []

    for ex in exercises:
        sft = convert_to_sft_format(ex)
        if len(sft["instruction"]) > 5 and len(sft["output"]) > 10:
            sft_samples.append(sft)
            cpt_texts.append(convert_to_cpt_format(ex))

    sft_path = output_dir / "sft_vietjack_crawled.jsonl"
    with open(sft_path, "w", encoding="utf-8") as f:
        for sample in sft_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(sft_samples)} SFT samples to {sft_path}")

    cpt_path = output_dir / "cpt_vietjack_crawled.jsonl"
    with open(cpt_path, "w", encoding="utf-8") as f:
        for text in cpt_texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"Saved {len(cpt_texts)} CPT samples to {cpt_path}")

    raw_path = output_dir / "vietjack_raw_exercises.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(exercises, f, ensure_ascii=False, indent=2)
    print(f"Saved raw data to {raw_path}")

    if sft_samples:
        print("\n" + "=" * 60)
        print("SAMPLE SFT DATA:")
        print("=" * 60)
        for sample in sft_samples[:5]:
            instr = sample['instruction'].encode('ascii', 'replace').decode()
            out = sample['output'][:200].encode('ascii', 'replace').decode()
            print(f"\n[Instruction]: {instr}")
            print(f"[Output]: {out}...")
            print("---")


if __name__ == "__main__":
    main()
