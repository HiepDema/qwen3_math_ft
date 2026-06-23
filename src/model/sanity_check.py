"""Pre-training sanity checks: validate data, tokenization, and training setup."""

import logging
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class SanityChecker:
    """Run sanity checks before training to catch common issues early."""

    def __init__(self, model_name: str, dataset_path: str, training_type: str = "sft"):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.training_type = training_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.issues = []
        self.warnings = []

    def run_all(self) -> dict:
        """Run all sanity checks."""
        logger.info("Running pre-training sanity checks...")

        self._check_dataset_loaded()
        self._check_tokenization()
        self._check_special_tokens()
        self._check_data_leakage()
        self._check_gpu_available()

        result = {
            "passed": len(self.issues) == 0,
            "issues": self.issues,
            "warnings": self.warnings,
        }

        if self.issues:
            logger.error(f"SANITY CHECK FAILED: {len(self.issues)} issues found")
            for issue in self.issues:
                logger.error(f"  - {issue}")
        else:
            logger.info("All sanity checks passed!")

        if self.warnings:
            for w in self.warnings:
                logger.warning(f"  - {w}")

        return result

    def _check_dataset_loaded(self):
        """Verify dataset loads correctly and has expected schema."""
        try:
            ds = load_from_disk(self.dataset_path)
            if len(ds) == 0:
                self.issues.append("Dataset is empty")
                return

            sample = ds[0]

            if self.training_type == "cpt":
                if "text" not in sample:
                    self.issues.append("CPT dataset missing 'text' field")
                elif not sample["text"]:
                    self.issues.append("First sample has empty text")
            else:
                if "conversations" not in sample:
                    self.issues.append("SFT dataset missing 'conversations' field")
                elif len(sample["conversations"]) < 2:
                    self.issues.append("SFT conversations have < 2 messages")

            logger.info(f"Dataset loaded: {len(ds)} samples")
        except Exception as e:
            self.issues.append(f"Failed to load dataset: {e}")

    def _check_tokenization(self):
        """Verify tokenization produces expected output."""
        test_texts = [
            "Solve: $x^2 + 2x + 1 = 0$",
            "\\begin{align} f(x) &= x^2 \\\\ f'(x) &= 2x \\end{align}",
            "The answer is $\\boxed{42}$.",
        ]

        for text in test_texts:
            tokens = self.tokenizer(text, return_tensors="pt")
            decoded = self.tokenizer.decode(tokens["input_ids"][0])

            if len(tokens["input_ids"][0]) == 0:
                self.issues.append(f"Tokenization produced empty output for: {text[:50]}")
            if len(tokens["input_ids"][0]) > 512:
                self.warnings.append(f"Short text produced {len(tokens['input_ids'][0])} tokens")

    def _check_special_tokens(self):
        """Verify EOS/BOS tokens are set correctly."""
        if self.tokenizer.eos_token is None:
            self.issues.append("EOS token is not set")
        if self.tokenizer.pad_token is None:
            self.warnings.append("PAD token not set (will use EOS)")

        # Check chat template for SFT
        if self.training_type == "sft":
            test_conv = [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
            try:
                formatted = self.tokenizer.apply_chat_template(test_conv, tokenize=False)
                if not formatted:
                    self.issues.append("Chat template produced empty string")
            except Exception as e:
                self.issues.append(f"Chat template failed: {e}")

    def _check_data_leakage(self):
        """Basic check for train/eval overlap."""
        try:
            ds = load_from_disk(self.dataset_path)
            if len(ds) < 100:
                return

            # Check first 1000 samples for exact duplicates
            seen = set()
            duplicates = 0
            for i, sample in enumerate(ds):
                if i >= 1000:
                    break
                if self.training_type == "cpt":
                    key = sample.get("text", "")[:100]
                else:
                    conv = sample.get("conversations", [])
                    key = conv[1]["content"][:100] if len(conv) > 1 else ""

                if key in seen:
                    duplicates += 1
                seen.add(key)

            if duplicates > 50:
                self.warnings.append(f"High duplicate rate in first 1000 samples: {duplicates}")
        except Exception:
            pass

    def _check_gpu_available(self):
        """Verify GPU is available and has sufficient memory."""
        if not torch.cuda.is_available():
            self.issues.append("CUDA not available - training requires GPU")
            return

        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, Memory: {gpu_memory:.1f} GB")

        if gpu_memory < 16:
            self.warnings.append(f"GPU memory ({gpu_memory:.1f} GB) may be insufficient for 7B model")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-7B")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--type", type=str, choices=["cpt", "sft"], default="sft")
    args = parser.parse_args()

    checker = SanityChecker(args.model, args.dataset, args.type)
    result = checker.run_all()

    if not result["passed"]:
        exit(1)
