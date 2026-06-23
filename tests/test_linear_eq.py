"""Unit tests for linear equation data generation and evaluation."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.generate_data import (
    format_equation,
    generate_linear_equation,
    generate_sft_sample,
    solve_equation_steps,
    generate_cpt_text_local,
    generate_sft_data_local,
)
from scripts.evaluate import extract_answer, check_format, check_steps


class TestFormatEquation:
    def test_positive_a_positive_b(self):
        assert format_equation(3, 5, 16) == "3x + 5 = 16"

    def test_positive_a_negative_b(self):
        assert format_equation(3, -5, 16) == "3x - 5 = 16"

    def test_a_is_one(self):
        assert format_equation(1, 3, 7) == "x + 3 = 7"

    def test_a_is_negative_one(self):
        assert format_equation(-1, 2, 5) == "-x + 2 = 5"

    def test_b_is_zero(self):
        assert format_equation(4, 0, 12) == "4x = 12"

    def test_negative_a(self):
        assert format_equation(-3, 5, 10) == "-3x + 5 = 10"


class TestSolveEquationSteps:
    def test_simple_case(self):
        result = solve_equation_steps(3, -5, 16)
        assert "Ta có:" in result
        assert "3x - 5 = 16" in result
        assert "3x = 21" in result
        assert "x = 7" in result
        assert "Đáp án: x = 7" in result

    def test_negative_answer(self):
        result = solve_equation_steps(2, 10, 4)
        assert "Đáp án: x = -3" in result

    def test_a_is_one(self):
        result = solve_equation_steps(1, 3, 10)
        assert "Đáp án: x = 7" in result

    def test_a_is_negative_one(self):
        result = solve_equation_steps(-1, 0, -5)
        assert "Đáp án: x = 5" in result


class TestGenerateSFTSample:
    def test_has_required_keys(self):
        sample = generate_sft_sample(3, -5, 16)
        assert "instruction" in sample
        assert "output" in sample

    def test_instruction_format(self):
        sample = generate_sft_sample(3, -5, 16)
        assert sample["instruction"] == "Giải phương trình: 3x - 5 = 16"

    def test_output_has_answer(self):
        sample = generate_sft_sample(2, 4, 10)
        assert "Đáp án: x = 3" in sample["output"]


class TestExtractAnswer:
    def test_standard_format(self):
        text = "Ta có:\n3x - 5 = 16\n3x = 21\nx = 7\nĐáp án: x = 7"
        assert extract_answer(text) == "7"

    def test_negative_answer(self):
        text = "Ta có:\n2x + 10 = 4\n2x = -6\nx = -3\nĐáp án: x = -3"
        assert extract_answer(text) == "-3"

    def test_fraction_answer(self):
        text = "Ta có:\n3x + 1 = 4\n3x = 3\nx = 1\nĐáp án: x = 1"
        assert extract_answer(text) == "1"

    def test_no_dap_an_fallback(self):
        text = "3x = 21\nx = 7"
        assert extract_answer(text) == "7"

    def test_no_answer(self):
        text = "some random text"
        assert extract_answer(text) is None


class TestCheckFormat:
    def test_valid_format(self):
        output = "Ta có:\n3x - 5 = 16\n3x = 21\nx = 7\nĐáp án: x = 7"
        result = check_format(output)
        assert result["has_ta_co"] is True
        assert result["has_dap_an"] is True
        assert result["has_enough_steps"] is True
        assert result["all_ok"] is True

    def test_missing_ta_co(self):
        output = "3x - 5 = 16\n3x = 21\nx = 7\nĐáp án: x = 7"
        result = check_format(output)
        assert result["has_ta_co"] is False
        assert result["all_ok"] is False

    def test_missing_dap_an(self):
        output = "Ta có:\n3x - 5 = 16\n3x = 21\nx = 7"
        result = check_format(output)
        assert result["has_dap_an"] is False
        assert result["all_ok"] is False

    def test_too_few_lines(self):
        output = "Ta có:\nĐáp án: x = 7"
        result = check_format(output)
        assert result["has_enough_steps"] is False


class TestCheckSteps:
    def test_all_steps_correct(self):
        output = "Ta có:\n3x - 5 = 16\n3x = 21\nx = 7\nĐáp án: x = 7"
        result = check_steps(output, 3, -5, 16)
        assert result["step1_correct"] is True
        assert result["step2_correct"] is True
        assert result["all_ok"] is True

    def test_step1_wrong(self):
        output = "Ta có:\n3x - 5 = 16\n3x = 20\nx = 7\nĐáp án: x = 7"
        result = check_steps(output, 3, -5, 16)
        assert result["step1_correct"] is False

    def test_step2_wrong(self):
        output = "Ta có:\n3x - 5 = 16\n3x = 21\nx = 8\nĐáp án: x = 8"
        result = check_steps(output, 3, -5, 16)
        assert result["step2_correct"] is False

    def test_negative_coefficient(self):
        output = "Ta có:\n-2x + 4 = 10\n-2x = 6\nx = -3\nĐáp án: x = -3"
        result = check_steps(output, -2, 4, 10)
        assert result["all_ok"] is True


class TestDataGeneration:
    def test_cpt_generates_correct_count(self):
        data = generate_cpt_text_local(20)
        assert len(data) == 20

    def test_cpt_data_not_empty(self):
        data = generate_cpt_text_local(10)
        for text in data:
            assert len(text) > 10

    def test_sft_generates_correct_count(self):
        data = generate_sft_data_local(20)
        assert len(data) == 20

    def test_sft_data_has_correct_format(self):
        data = generate_sft_data_local(10)
        for sample in data:
            assert "instruction" in sample
            assert "output" in sample
            assert "Giải phương trình:" in sample["instruction"]
            assert "Đáp án:" in sample["output"]

    def test_sft_data_no_duplicates(self):
        data = generate_sft_data_local(50)
        instructions = [d["instruction"] for d in data]
        assert len(instructions) == len(set(instructions))


class TestDataFileFormat:
    """Test that saved data files have correct format."""

    def test_cpt_jsonl_format(self, tmp_path):
        data = generate_cpt_text_local(5)
        filepath = tmp_path / "cpt.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for text in data:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

        # Read back and validate
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                assert "text" in item
                assert isinstance(item["text"], str)
                assert len(item["text"]) > 0

    def test_sft_jsonl_format(self, tmp_path):
        data = generate_sft_data_local(5)
        filepath = tmp_path / "sft.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for sample in data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                assert "instruction" in item
                assert "output" in item
                assert item["instruction"].startswith("Giải phương trình:")
                assert "Đáp án:" in item["output"]
