"""Tests for data quality framework."""

import pytest
from datasets import Dataset

from src.data_engineering.data_quality import DataQualityChecker


@pytest.fixture
def quality_checker():
    return DataQualityChecker()


@pytest.fixture
def sample_cpt_dataset():
    return Dataset.from_dict({
        "text": [
            "This is a math problem about calculus. $f(x) = x^2 + 2x + 1$. "
            "The derivative is $f'(x) = 2x + 2$. Setting this equal to zero gives $x = -1$.",
            "Consider the integral $\\int_0^1 x^2 dx = \\frac{1}{3}$. "
            "This can be verified by the fundamental theorem of calculus.",
            "Solve the equation $x^2 - 5x + 6 = 0$. Using the quadratic formula: "
            "$x = \\frac{5 \\pm \\sqrt{25-24}}{2} = \\frac{5 \\pm 1}{2}$. So $x=3$ or $x=2$.",
        ] * 20,
        "quality_score": [0.8, 0.7, 0.9] * 20,
    })


@pytest.fixture
def sample_sft_dataset():
    return Dataset.from_dict({
        "conversations": [
            [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "What is the derivative of x^3?"},
                {"role": "assistant", "content": "The derivative of x^3 is 3x^2."},
            ],
            [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "Solve 2x + 3 = 7"},
                {"role": "assistant", "content": "Subtract 3: 2x = 4. Divide by 2: x = 2."},
            ],
        ] * 30,
    })


def test_cpt_completeness(quality_checker, sample_cpt_dataset):
    result = quality_checker._check_completeness(sample_cpt_dataset, "cpt")
    assert result["passed"] is True
    assert result["null_count"] == 0


def test_sft_completeness(quality_checker, sample_sft_dataset):
    result = quality_checker._check_completeness(sample_sft_dataset, "sft")
    assert result["passed"] is True


def test_length_distribution(quality_checker, sample_cpt_dataset):
    result = quality_checker._check_length_distribution(sample_cpt_dataset, "cpt")
    assert "mean" in result
    assert "median" in result
    assert result["too_short"] == 0


def test_duplicates_detection(quality_checker, sample_cpt_dataset):
    result = quality_checker._check_duplicates(sample_cpt_dataset, "cpt")
    # Our fixture has duplicates (repeated 20x)
    assert result["duplicate_count"] > 0


def test_format_check_sft(quality_checker, sample_sft_dataset):
    result = quality_checker._check_format(sample_sft_dataset, "sft")
    assert result["passed"] is True
    assert result["error_count"] == 0


def test_format_check_invalid():
    checker = DataQualityChecker()
    bad_dataset = Dataset.from_dict({
        "conversations": [
            [{"role": "user", "content": "hi"}],  # Missing assistant
            "not a list",  # Wrong type
        ],
    })
    result = checker._check_format(bad_dataset, "sft")
    assert result["passed"] is False


def test_full_quality_check(quality_checker, sample_cpt_dataset):
    report = quality_checker.check_dataset(sample_cpt_dataset, "cpt")
    assert "checks" in report
    assert "completeness" in report["checks"]
    assert "length_distribution" in report["checks"]
    assert "duplicates" in report["checks"]
