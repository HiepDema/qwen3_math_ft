"""Benchmark framework: standardized evaluation across multiple dimensions."""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

from deepeval import evaluate as deepeval_evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    benchmark: str
    model: str
    accuracy: float
    reasoning_score: float
    total_samples: int
    metadata: dict


class MathBenchmark:
    """Standardized math evaluation benchmark."""

    def __init__(self, model_path: str, judge_model: str = "gpt-4o-mini"):
        self.model_path = model_path
        self.judge_model = judge_model

    def evaluate_reasoning_quality(self, predictions: list[dict]) -> float:
        """Use LLM-as-Judge to evaluate reasoning quality."""
        correctness_metric = GEval(
            name="Mathematical Correctness",
            criteria=(
                "Evaluate whether the mathematical solution is correct, complete, "
                "and follows a logical step-by-step approach."
            ),
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            model=self.judge_model,
        )

        test_cases = []
        for pred in predictions[:100]:  # Limit for cost
            test_case = LLMTestCase(
                input=pred["question"],
                actual_output=pred.get("full_response", pred.get("predicted", "")),
                expected_output=pred.get("gold", ""),
            )
            test_cases.append(test_case)

        results = deepeval_evaluate(test_cases, [correctness_metric])

        scores = [tc.metrics_data[0].score for tc in results.test_results if tc.metrics_data]
        avg_score = sum(scores) / max(len(scores), 1)

        return avg_score

    def evaluate_step_completeness(self, predictions: list[dict]) -> float:
        """Evaluate completeness of reasoning steps."""
        step_metric = GEval(
            name="Step Completeness",
            criteria=(
                "Evaluate whether the solution breaks down the problem into clear, "
                "logical steps. Each step should be justified and build on previous steps. "
                "The final answer should clearly follow from the reasoning chain."
            ),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            model=self.judge_model,
        )

        test_cases = [
            LLMTestCase(
                input=pred["question"],
                actual_output=pred.get("full_response", ""),
            )
            for pred in predictions[:50]
        ]

        results = deepeval_evaluate(test_cases, [step_metric])
        scores = [tc.metrics_data[0].score for tc in results.test_results if tc.metrics_data]
        return sum(scores) / max(len(scores), 1)

    def run_full_benchmark(self, predictions: dict[str, list]) -> dict:
        """Run comprehensive benchmark evaluation."""
        results = {}

        for benchmark_name, preds in predictions.items():
            logger.info(f"Running LLM-Judge evaluation for {benchmark_name}...")

            reasoning_score = self.evaluate_reasoning_quality(preds)
            step_score = self.evaluate_step_completeness(preds)

            accuracy = sum(1 for p in preds if p.get("correct")) / max(len(preds), 1)

            results[benchmark_name] = BenchmarkResult(
                benchmark=benchmark_name,
                model=self.model_path,
                accuracy=accuracy,
                reasoning_score=reasoning_score,
                total_samples=len(preds),
                metadata={"step_completeness": step_score},
            )

            logger.info(
                f"  {benchmark_name}: accuracy={accuracy:.4f}, "
                f"reasoning={reasoning_score:.4f}, steps={step_score:.4f}"
            )

        return {k: asdict(v) for k, v in results.items()}


if __name__ == "__main__":
    output_dir = Path("outputs/eval")
    results_file = output_dir / "eval_results.json"

    if not results_file.exists():
        logger.error("Run evaluation first: make eval")
        exit(1)

    with open(results_file) as f:
        eval_results = json.load(f)

    predictions = {name: data["predictions"] for name, data in eval_results.items()}

    benchmark = MathBenchmark(model_path="outputs/sft/final")
    full_results = benchmark.run_full_benchmark(predictions)

    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(full_results, f, indent=2)

    logger.info("Benchmark evaluation complete!")
