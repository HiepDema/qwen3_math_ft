"""Tests for evaluation utilities."""

import pytest
from src.evaluation.run_eval import extract_answer, normalize_answer


class TestExtractAnswer:
    def test_boxed_answer(self):
        text = "Therefore, $x = \\boxed{42}$."
        assert extract_answer(text) == "42"

    def test_boxed_fraction(self):
        text = "The answer is $\\boxed{\\frac{1}{2}}$."
        assert extract_answer(text) == "\\frac{1}{2}"

    def test_answer_is_pattern(self):
        text = "After simplification, the answer is 7."
        assert extract_answer(text) == "7"

    def test_final_answer_pattern(self):
        text = "Computing step by step...\nFinal answer: 15"
        assert extract_answer(text) == "15"

    def test_last_number_fallback(self):
        text = "We compute 3 + 4 = 7 and then 7 * 2 = 14"
        assert extract_answer(text) == "14"

    def test_negative_number(self):
        text = "The solution is $\\boxed{-3}$."
        assert extract_answer(text) == "-3"

    def test_multiple_boxed(self):
        text = "First we get $\\boxed{5}$ but actually $\\boxed{10}$."
        assert extract_answer(text) == "10"


class TestNormalizeAnswer:
    def test_numeric(self):
        assert normalize_answer("42") == "42.0"
        assert normalize_answer("42.0") == "42.0"
        assert normalize_answer(" 42 ") == "42.0"

    def test_strips_symbols(self):
        assert normalize_answer("$42") == "42.0"
        assert normalize_answer("42%") == "42.0"
        assert normalize_answer("1,000") == "1000.0"

    def test_text_answer(self):
        assert normalize_answer("yes") == "yes"
        assert normalize_answer("  YES  ") == "yes"

    def test_comparison(self):
        assert normalize_answer("42") == normalize_answer("42.0")
        assert normalize_answer("$100") == normalize_answer("100")
