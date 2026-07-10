"""Export training loss charts from wandb and save as PNG.

Downloads loss curves for each training run and creates comparison plots.
Saves images to reports/figures/ for embedding in the report.

Requirements:
    pip install wandb matplotlib

Usage:
    python scripts/export_wandb_charts.py
    python scripts/export_wandb_charts.py --project lsreasoning-sft-vs-grpo --entity hiep26-sdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import wandb


FIGURES_DIR = Path("reports/figures")


def fetch_run_history(api, entity, project, run_name, metrics):
    """Fetch metric history for a specific run."""
    runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
    if not runs:
        print(f"  WARNING: Run '{run_name}' not found")
        return None
    run = runs[0]
    history = run.history(keys=metrics, pandas=True)
    return history


def plot_sft_loss(api, entity, project):
    """Plot SFT training loss (standalone vs CPT+SFT)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    runs_to_plot = [
        ("sft_lsreasoning", "SFT only", "#2196F3"),
        ("cpt_sft", "CPT + SFT", "#4CAF50"),
    ]

    for run_name, label, color in runs_to_plot:
        history = fetch_run_history(api, entity, project, run_name, ["train/loss"])
        if history is not None and "train/loss" in history.columns:
            data = history["train/loss"].dropna()
            ax.plot(data.index, data.values, label=label, color=color, linewidth=1.5)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("SFT Training Loss Comparison", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = FIGURES_DIR / "sft_train_loss.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_cpt_loss(api, entity, project):
    """Plot CPT training loss."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    history = fetch_run_history(api, entity, project, "cpt_lsreasoning", ["train/loss"])
    if history is not None and "train/loss" in history.columns:
        data = history["train/loss"].dropna()
        ax.plot(data.index, data.values, color="#FF9800", linewidth=1.5)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("CPT Training Loss", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = FIGURES_DIR / "cpt_train_loss.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_grpo_loss(api, entity, project):
    """Plot GRPO training loss (dense vs sparse) for SFT-based GRPO."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    runs_to_plot = [
        ("grpo_dense", "SFT+GRPO Dense", "#E91E63"),
        ("grpo_sparse", "SFT+GRPO Sparse", "#9C27B0"),
    ]

    for run_name, label, color in runs_to_plot:
        history = fetch_run_history(api, entity, project, run_name, ["train/loss"])
        if history is not None and "train/loss" in history.columns:
            data = history["train/loss"].dropna()
            ax.plot(data.index, data.values, label=label, color=color, linewidth=1.5)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("SFT+GRPO Training Loss", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for run_name, label, color in runs_to_plot:
        history = fetch_run_history(api, entity, project, run_name, ["train/reward"])
        if history is not None and "train/reward" in history.columns:
            data = history["train/reward"].dropna()
            ax.plot(data.index, data.values, label=label, color=color, linewidth=1.5)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Mean Reward", fontsize=12)
    ax.set_title("SFT+GRPO Mean Reward", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "grpo_sft_train_loss_reward.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_cpt_grpo_loss(api, entity, project):
    """Plot GRPO training loss (dense vs sparse) for CPT+SFT-based GRPO."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    runs_to_plot = [
        ("grpo_dense", "CPT+SFT+GRPO Dense", "#FF5722"),
        ("grpo_sparse", "CPT+SFT+GRPO Sparse", "#795548"),
    ]

    # CPT+SFT GRPO runs may have different names in wandb
    # Try alternative names if the standard ones don't work
    alt_names = [
        ("cpt_grpo_dense", "CPT+SFT+GRPO Dense", "#FF5722"),
        ("cpt_grpo_sparse", "CPT+SFT+GRPO Sparse", "#795548"),
    ]

    found = False
    for run_name, label, color in alt_names:
        history = fetch_run_history(api, entity, project, run_name, ["train/loss"])
        if history is not None and "train/loss" in history.columns:
            data = history["train/loss"].dropna()
            ax.plot(data.index, data.values, label=label, color=color, linewidth=1.5)
            found = True

    if not found:
        for run_name, label, color in runs_to_plot:
            history = fetch_run_history(api, entity, project, run_name, ["train/loss"])
            if history is not None and "train/loss" in history.columns:
                data = history["train/loss"].dropna()
                ax.plot(data.index, data.values, label="CPT+" + label, color=color, linewidth=1.5)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("CPT+SFT+GRPO Training Loss", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for run_name, label, color in alt_names:
        history = fetch_run_history(api, entity, project, run_name, ["train/reward"])
        if history is not None and "train/reward" in history.columns:
            data = history["train/reward"].dropna()
            ax.plot(data.index, data.values, label=label, color=color, linewidth=1.5)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Mean Reward", fontsize=12)
    ax.set_title("CPT+SFT+GRPO Mean Reward", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "grpo_cpt_sft_train_loss_reward.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_eval_comparison():
    """Plot final evaluation comparison bar chart."""
    import json

    results_file = Path("outputs/experiment_results.json")
    if not results_file.exists():
        print("  WARNING: experiment_results.json not found")
        return

    with open(results_file) as f:
        results = json.load(f)

    methods = list(results.keys())
    exact_match = [results[m]["exact_match"] * 100 for m in methods]
    close_match = [results[m]["close_match"] * 100 for m in methods]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    x = range(len(methods))
    width = 0.35
    bars1 = ax.bar([i - width/2 for i in x], exact_match, width, label="Exact Match", color="#2196F3")
    bars2 = ax.bar([i + width/2 for i in x], close_match, width, label="Close Match", color="#4CAF50")

    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Evaluation Results Comparison (6 Methods)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9, rotation=15, ha="right")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(70, 90)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / "eval_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_by_problem_type():
    """Plot accuracy by problem type for each method."""
    import json

    results_file = Path("outputs/experiment_results.json")
    if not results_file.exists():
        return

    with open(results_file) as f:
        results = json.load(f)

    methods = list(results.keys())
    all_types = set()
    for m in methods:
        all_types.update(results[m].get("by_type", {}).keys())

    # Filter to types with <100% accuracy (interesting ones)
    interesting_types = []
    for t in sorted(all_types):
        scores = [results[m].get("by_type", {}).get(t, 0) for m in methods]
        if min(scores) < 0.99:
            interesting_types.append(t)

    if not interesting_types:
        return

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    x = range(len(interesting_types))
    n_methods = len(methods)
    width = 0.8 / n_methods
    colors = ["#4CAF50", "#2196F3", "#E91E63", "#9C27B0", "#FF5722", "#795548"]

    for i, method in enumerate(methods):
        color = colors[i % len(colors)]
        scores = [results[method].get("by_type", {}).get(t, 0) * 100 for t in interesting_types]
        ax.bar([xi + i * width for xi in x], scores, width, label=method, color=color)

    ax.set_xlabel("Problem Type", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy by Problem Type (6 Methods)", fontsize=14)
    ax.set_xticks([xi + width * (n_methods - 1) / 2 for xi in x])
    short_labels = [t.replace("Solve the ", "").replace("Solve a ", "").replace(".", "") for t in interesting_types]
    ax.set_xticklabels(short_labels, fontsize=9, rotation=15, ha="right")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = FIGURES_DIR / "accuracy_by_type.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, default="hiep26-sdf")
    parser.add_argument("--project", type=str, default="lsreasoning-sft-vs-grpo")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Exporting charts from wandb...")
    print(f"  Entity: {args.entity}")
    print(f"  Project: {args.project}")
    print()

    api = wandb.Api()

    print("[1/6] CPT loss curve...")
    plot_cpt_loss(api, args.entity, args.project)

    print("[2/6] SFT loss curves...")
    plot_sft_loss(api, args.entity, args.project)

    print("[3/6] SFT+GRPO loss + reward curves...")
    plot_grpo_loss(api, args.entity, args.project)

    print("[4/6] CPT+SFT+GRPO loss + reward curves...")
    plot_cpt_grpo_loss(api, args.entity, args.project)

    print("[5/6] Evaluation comparison (6 methods)...")
    plot_eval_comparison()

    print("[6/6] Accuracy by problem type (6 methods)...")
    plot_by_problem_type()

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print("Embed in report with: ![caption](figures/filename.png)")


if __name__ == "__main__":
    main()
