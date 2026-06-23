"""Tests for ETL pipeline components."""

import pytest
from src.data_engineering.etl_pipeline import MathDataCleaner, CPTDataProcessor, SFTDataProcessor
from datasets import Dataset


@pytest.fixture
def cleaner():
    return MathDataCleaner()


class TestMathDataCleaner:
    def test_clean_text_basic(self, cleaner):
        text = "  This is a test about $x^2 + 1$.  \n\n\n\n  More text here.  "
        result = cleaner.clean_text(text)
        assert result is not None
        assert "\n\n\n" not in result

    def test_clean_text_too_short(self, cleaner):
        assert cleaner.clean_text("short") is None
        assert cleaner.clean_text("") is None
        assert cleaner.clean_text(None) is None

    def test_has_math_content(self, cleaner):
        assert cleaner.has_math_content("The equation $x^2 = 4$ has two solutions.")
        assert cleaner.has_math_content("By the theorem, we have 2 + 3 = 5.")
        assert not cleaner.has_math_content("The cat sat on the mat.")

    def test_quality_score(self, cleaner):
        math_text = (
            "Consider the equation $x^2 + 2x + 1 = 0$. "
            "By the quadratic formula $x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$, "
            "we get $x = -1$. This can be verified by substitution.\n\n"
            "Furthermore, $f(x) = (x+1)^2 \\geq 0$ for all $x$."
        )
        score = cleaner.estimate_quality(math_text)
        assert score > 0.5

        plain_text = "Hello world. " * 50
        plain_score = cleaner.estimate_quality(plain_text)
        assert plain_score < score


class TestCPTDataProcessor:
    def test_process_filters_low_quality(self):
        dataset = Dataset.from_dict({
            "text": [
                "Good math: $\\int_0^1 x dx = 1/2$. The proof follows from FTC. " * 5,
                "short",
                "",
                "No math content here, just regular English text about cooking recipes. " * 3,
            ]
        })
        processor = CPTDataProcessor(min_quality=0.3)
        result = processor.process(dataset)
        assert len(result) <= len(dataset)


class TestSFTDataProcessor:
    def test_process_valid(self):
        dataset = Dataset.from_dict({
            "problem": ["What is 2+2?", "Solve x^2=4"],
            "solution": [
                "Step 1: Add 2 and 2. Step 2: 2+2=4. The answer is 4.",
                "Step 1: Take square root. x = ±2. The solutions are x=2 and x=-2.",
            ],
        })
        processor = SFTDataProcessor()
        result = processor.process(dataset)
        assert len(result) == 2
        assert "conversations" in result.column_names

    def test_process_filters_empty(self):
        dataset = Dataset.from_dict({
            "problem": ["Valid question?", ""],
            "solution": ["Valid detailed answer here for testing.", ""],
        })
        processor = SFTDataProcessor()
        result = processor.process(dataset)
        assert len(result) <= 2
