"""Generate comparison charts to wandb after all experiments complete.

Creates:
1. Bar chart: Exact Match / Close Match / Format / Reasoning across methods
2. Bar chart: Accuracy by problem type per method
3. Summary table with all metrics

Run after all experiments are done:
    python scripts/plot_comparison_wandb.py

Or pass results directly:
    python scripts/plot_comparison_wandb.py --results-file outputs/experiment_results.json
"""

import argparse
import json
from pathlib import Path

import wandb


RESULTS_FILE = "outputs/experiment_results.json"


def load_results(results_file: str) -> dict:
    """Load saved experiment results."""
    with open(results_file, "r") as f:
        return json.load(f)


def plot_comparison(results: dict):
    """Create wandb comparison charts from experiment results."""
    wandb.init(
        project="lsreasoning-sft-vs-grpo",
        name="comparison_report",
        reinit=True,
    )

    methods = list(results.keys())
    metrics = ["exact_match", "close_match", "format_score", "reasoning_score"]
    metric_labels = ["Exact Match", "Close Match", "Format", "Reasoning"]

    # 1. Overall comparison bar chart
    table = wandb.Table(columns=["method"] + metric_labels)
    for method in methods:
        row = [method]
        for m in metrics:
            row.append(results[method].get(m, 0))
        table.add_data(*row)

    wandb.log({
        "comparison/overall": wandb.plot.bar(
            table, "method", "Exact Match",
            title="Exact Match by Method"
        ),
    })

    # Log as grouped bar chart via custom table
    bar_table = wandb.Table(columns=["method", "metric", "score"])
    for method in methods:
        for m, label in zip(metrics, metric_labels):
            bar_table.add_data(method, label, results[method].get(m, 0))
    wandb.log({"comparison/all_metrics": bar_table})

    # 2. By problem type comparison
    all_types = set()
    for method in methods:
        by_type = results[method].get("by_type", {})
        all_types.update(by_type.keys())

    if all_types:
        type_table = wandb.Table(columns=["problem_type"] + methods)
        for ptype in sorted(all_types):
            row = [ptype]
            for method in methods:
                row.append(results[method].get("by_type", {}).get(ptype, 0))
            type_table.add_data(*row)
        wandb.log({"comparison/by_problem_type": type_table})

    # 3. Summary
    summary_table = wandb.Table(
        columns=["Method", "Exact Match", "Close Match", "Format", "Reasoning", "Test Size"]
    )
    for method in methods:
        r = results[method]
        summary_table.add_data(
            method,
            f"{r.get('exact_match', 0):.1%}",
            f"{r.get('close_match', 0):.1%}",
            f"{r.get('format_score', 0):.1%}",
            f"{r.get('reasoning_score', 0):.1%}",
            r.get("total", 0),
        )
    wandb.log({"comparison/summary": summary_table})

    # Also log scalar summaries for easy dashboard
    for method in methods:
        for m in metrics:
            wandb.summary[f"{method}/{m}"] = results[method].get(m, 0)

    wandb.finish()
    print("\nComparison charts logged to wandb!")
    print("  Project: lsreasoning-sft-vs-grpo")
    print("  Run: comparison_report")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", type=str, default=RESULTS_FILE)
    args = parser.parse_args()

    if not Path(args.results_file).exists():
        print(f"Results file not found: {args.results_file}")
        print("Run the experiment pipeline first, or provide --results-file")
        return

    results = load_results(args.results_file)
    print(f"Loaded results for: {list(results.keys())}")
    plot_comparison(results)


if __name__ == "__main__":
    main()
