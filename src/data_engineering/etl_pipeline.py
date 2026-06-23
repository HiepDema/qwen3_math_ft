"""ETL Pipeline: Extract, Transform, Load for math datasets."""

import logging
import re
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_from_disk

from .data_quality import DataQualityChecker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


class MathDataCleaner:
    """Clean and normalize mathematical text data."""

    def __init__(self):
        self.latex_pattern = re.compile(r"\$[^$]+\$|\\\[.*?\\\]|\\\(.*?\\\)", re.DOTALL)

    def clean_text(self, text: str) -> str | None:
        if not text or len(text.strip()) < 50:
            return None

        text = text.strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"[^\x00-\x7F-ɏ∀-⋿℀-⅏]+", "", text)

        return text

    def has_math_content(self, text: str) -> bool:
        """Check if text contains mathematical expressions."""
        math_indicators = [
            self.latex_pattern,
            re.compile(r"\d+\s*[+\-*/=<>≤≥]\s*\d+"),
            re.compile(r"(theorem|proof|lemma|equation|formula|calculate|solve)", re.IGNORECASE),
        ]
        return any(p.search(text) for p in math_indicators)

    def estimate_quality(self, text: str) -> float:
        """Score text quality 0-1 based on math content density."""
        if not text:
            return 0.0

        score = 0.0
        latex_matches = self.latex_pattern.findall(text)
        score += min(len(latex_matches) / 5.0, 0.4)

        words = text.split()
        if 100 <= len(words) <= 5000:
            score += 0.2
        elif 50 <= len(words) < 100:
            score += 0.1

        if self.has_math_content(text):
            score += 0.3

        paragraphs = text.split("\n\n")
        if len(paragraphs) >= 2:
            score += 0.1

        return min(score, 1.0)


class CPTDataProcessor:
    """Process data for Continual Pre-Training."""

    def __init__(self, min_quality: float = 0.3, max_length: int = 4096):
        self.cleaner = MathDataCleaner()
        self.min_quality = min_quality
        self.max_length = max_length

    def process(self, dataset: Dataset) -> Dataset:
        logger.info(f"Processing CPT dataset: {len(dataset)} samples")

        processed = dataset.map(self._process_sample, num_proc=4, remove_columns=dataset.column_names)
        processed = processed.filter(lambda x: x["text"] is not None and x["quality_score"] >= self.min_quality)

        logger.info(f"After filtering: {len(processed)} samples")
        return processed

    def _process_sample(self, sample: dict) -> dict:
        text = sample.get("text", "")
        cleaned = self.cleaner.clean_text(text)

        if cleaned and len(cleaned) > self.max_length * 4:
            cleaned = cleaned[: self.max_length * 4]

        quality = self.cleaner.estimate_quality(cleaned) if cleaned else 0.0

        return {"text": cleaned, "quality_score": quality}


class SFTDataProcessor:
    """Process data for Supervised Fine-Tuning."""

    SYSTEM_PROMPT = (
        "You are a mathematical reasoning assistant. Solve problems step by step, "
        "showing clear chain-of-thought reasoning. Always verify your answer."
    )

    def __init__(self, max_length: int = 2048):
        self.max_length = max_length
        self.cleaner = MathDataCleaner()

    def process(self, dataset: Dataset) -> Dataset:
        logger.info(f"Processing SFT dataset: {len(dataset)} samples")

        processed = dataset.map(self._process_sample, num_proc=4, remove_columns=dataset.column_names)
        processed = processed.filter(lambda x: x["conversations"] is not None)

        logger.info(f"After filtering: {len(processed)} samples")
        return processed

    def _process_sample(self, sample: dict) -> dict:
        problem = sample.get("problem", sample.get("question", ""))
        solution = sample.get("solution", sample.get("answer", ""))

        if not problem or not solution:
            return {"conversations": None}

        problem = problem.strip()
        solution = solution.strip()

        if len(problem) < 10 or len(solution) < 20:
            return {"conversations": None}

        conversations = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution},
        ]

        return {"conversations": conversations}


def run_etl():
    """Run full ETL pipeline."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    quality_checker = DataQualityChecker()

    # Process CPT data
    logger.info("=" * 50)
    logger.info("Processing CPT data (OpenWebMath)")
    logger.info("=" * 50)

    cpt_raw = load_from_disk(str(RAW_DIR / "openwebmath"))
    cpt_processor = CPTDataProcessor()
    cpt_processed = cpt_processor.process(cpt_raw)

    cpt_report = quality_checker.check_dataset(cpt_processed, dataset_type="cpt")
    logger.info(f"CPT Quality Report: {cpt_report}")

    cpt_processed.save_to_disk(str(PROCESSED_DIR / "cpt"))
    logger.info(f"CPT data saved: {len(cpt_processed)} samples")

    # Process SFT data
    logger.info("=" * 50)
    logger.info("Processing SFT data (NuminaMath-CoT)")
    logger.info("=" * 50)

    sft_raw = load_from_disk(str(RAW_DIR / "numinamath_cot"))
    sft_processor = SFTDataProcessor()
    sft_processed = sft_processor.process(sft_raw)

    sft_report = quality_checker.check_dataset(sft_processed, dataset_type="sft")
    logger.info(f"SFT Quality Report: {sft_report}")

    sft_processed.save_to_disk(str(PROCESSED_DIR / "sft"))
    logger.info(f"SFT data saved: {len(sft_processed)} samples")

    # Save processing stats
    stats = {
        "cpt": {"raw_count": len(cpt_raw), "processed_count": len(cpt_processed), "quality_report": cpt_report},
        "sft": {"raw_count": len(sft_raw), "processed_count": len(sft_processed), "quality_report": sft_report},
    }

    import json
    with open(PROCESSED_DIR / "processing_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("ETL pipeline completed!")


if __name__ == "__main__":
    run_etl()
