"""Data Quality Framework: validation, profiling, and monitoring."""

import logging
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    dataset_name: str
    total_samples: int
    checks: dict = field(default_factory=dict)
    passed: bool = True
    warnings: list = field(default_factory=list)

    def __repr__(self):
        status = "PASSED" if self.passed else "FAILED"
        failed = [k for k, v in self.checks.items() if not v["passed"]]
        return f"QualityReport({status}, samples={self.total_samples}, failed_checks={failed})"


class DataQualityChecker:
    """Comprehensive data quality validation framework."""

    def check_dataset(self, dataset: Dataset, dataset_type: str = "cpt") -> dict:
        report = QualityReport(dataset_name=dataset_type, total_samples=len(dataset))

        report.checks["completeness"] = self._check_completeness(dataset, dataset_type)
        report.checks["length_distribution"] = self._check_length_distribution(dataset, dataset_type)
        report.checks["duplicates"] = self._check_duplicates(dataset, dataset_type)
        report.checks["format_validity"] = self._check_format(dataset, dataset_type)

        report.passed = all(c["passed"] for c in report.checks.values())
        return report.__dict__

    def _check_completeness(self, dataset: Dataset, dataset_type: str) -> dict:
        """Check for null/empty values."""
        if dataset_type == "cpt":
            null_count = sum(1 for x in dataset if not x.get("text"))
        else:
            null_count = sum(1 for x in dataset if not x.get("conversations"))

        null_ratio = null_count / max(len(dataset), 1)
        return {
            "passed": null_ratio < 0.01,
            "null_count": null_count,
            "null_ratio": round(null_ratio, 4),
        }

    def _check_length_distribution(self, dataset: Dataset, dataset_type: str) -> dict:
        """Validate text length distribution."""
        if dataset_type == "cpt":
            lengths = [len(x["text"].split()) for x in dataset if x.get("text")]
        else:
            lengths = [
                sum(len(m["content"].split()) for m in x["conversations"])
                for x in dataset
                if x.get("conversations")
            ]

        if not lengths:
            return {"passed": False, "error": "no valid samples"}

        lengths = np.array(lengths)
        stats = {
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "p5": float(np.percentile(lengths, 5)),
            "p95": float(np.percentile(lengths, 95)),
            "std": float(np.std(lengths)),
            "too_short": int(np.sum(lengths < 20)),
            "too_long": int(np.sum(lengths > 10000)),
        }

        too_short_ratio = stats["too_short"] / len(lengths)
        too_long_ratio = stats["too_long"] / len(lengths)

        stats["passed"] = too_short_ratio < 0.05 and too_long_ratio < 0.05
        return stats

    def _check_duplicates(self, dataset: Dataset, dataset_type: str) -> dict:
        """Check for near-duplicate samples."""
        if dataset_type == "cpt":
            fingerprints = [hash(x["text"][:200]) for x in dataset if x.get("text")]
        else:
            fingerprints = [
                hash(x["conversations"][1]["content"][:200])
                for x in dataset
                if x.get("conversations") and len(x["conversations"]) > 1
            ]

        counts = Counter(fingerprints)
        duplicates = sum(v - 1 for v in counts.values() if v > 1)
        dup_ratio = duplicates / max(len(fingerprints), 1)

        return {
            "passed": dup_ratio < 0.05,
            "duplicate_count": duplicates,
            "duplicate_ratio": round(dup_ratio, 4),
            "unique_count": len(counts),
        }

    def _check_format(self, dataset: Dataset, dataset_type: str) -> dict:
        """Validate data format/schema."""
        errors = []

        if dataset_type == "sft":
            for i, x in enumerate(dataset):
                conv = x.get("conversations")
                if not conv:
                    continue
                if not isinstance(conv, list):
                    errors.append(f"Sample {i}: conversations is not a list")
                    continue
                if len(conv) < 2:
                    errors.append(f"Sample {i}: conversations has < 2 messages")
                    continue
                roles = [m.get("role") for m in conv]
                if "user" not in roles or "assistant" not in roles:
                    errors.append(f"Sample {i}: missing user/assistant role")

                if len(errors) > 100:
                    break
        else:
            for i, x in enumerate(dataset):
                if x.get("text") and not isinstance(x["text"], str):
                    errors.append(f"Sample {i}: text is not a string")
                if len(errors) > 100:
                    break

        return {
            "passed": len(errors) == 0,
            "error_count": len(errors),
            "sample_errors": errors[:10],
        }
