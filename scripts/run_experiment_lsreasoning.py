"""Experiment: SFT vs GRPO (dense/sparse) on LSReasoning-15000.

Uses a single 80/20 train/test split for all methods:
1. SFT only
2. SFT -> GRPO (dense reward)
3. SFT -> GRPO (sparse reward)

All methods train on the same 80% and evaluate on the same 20%.

Usage:
    python scripts/run_experiment_lsreasoning.py
    python scripts/run_experiment_lsreasoning.py --skip-sft --skip-grpo-dense
"""

import argparse
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset


DATA_DIR = Path("data/lsreasoning_split")
TRAIN_FILE = DATA_DIR / "train.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"


def prepare_data_split(dataset_name: str, seed: int = 42):
    """Download LSReasoning-15000 and split 80/20."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if TRAIN_FILE.exists() and TEST_FILE.exists():
        train_count = sum(1 for _ in open(TRAIN_FILE))
        test_count = sum(1 for _ in open(TEST_FILE))
        print(f"Data split already exists: train={train_count}, test={test_count}")
        return

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    print(f"  Total samples: {len(dataset)}")

    split = dataset.train_test_split(test_size=0.2, seed=seed)
    train_data = split["train"]
    test_data = split["test"]

    print(f"  Train: {len(train_data)} (80%)")
    print(f"  Test:  {len(test_data)} (20%)")

    for path, data in [(TRAIN_FILE, train_data), (TEST_FILE, test_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  Saved to {DATA_DIR}/")


def run_step(description: str, func, *args, **kwargs):
    """Run a training/eval step with logging."""
    print("\n" + "=" * 60)
    print(f"STEP: {description}")
    print("=" * 60 + "\n")
    result = func(*args, **kwargs)
    print(f"\n[OK] {description}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str,
                        default="DataMuncher-Labs/LSReasoning-15000")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--sft-epochs", type=int, default=3)
    parser.add_argument("--sft-lr", type=float, default=1e-4)
    parser.add_argument("--sft-batch-size", type=int, default=4)
    parser.add_argument("--sft-grad-accum", type=int, default=4)
    parser.add_argument("--grpo-epochs", type=int, default=1)
    parser.add_argument("--grpo-lr", type=float, default=5e-6)
    parser.add_argument("--grpo-batch-size", type=int, default=4)
    parser.add_argument("--grpo-num-generations", type=int, default=4)
    parser.add_argument("--grpo-beta", type=float, default=0.1)
    parser.add_argument("--num-eval", type=int, default=None,
                        help="Number of test samples to evaluate (default: all)")
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--skip-grpo-dense", action="store_true")
    parser.add_argument("--skip-grpo-sparse", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("WANDB_PROJECT", "lsreasoning-sft-vs-grpo")

    print("=" * 60)
    print("LSReasoning-15000: SFT vs GRPO (Dense vs Sparse)")
    print("=" * 60)
    print(f"  Model: {args.model_name}")
    print(f"  Data split: 80% train / 20% test")
    print(f"  Seed: {args.seed}")
    print(f"  wandb project: lsreasoning-sft-vs-grpo")
    print()

    # Step 0: Prepare data
    prepare_data_split(args.dataset_name, seed=args.seed)

    # Import training/eval modules (same directory)
    sys.path.insert(0, str(Path(__file__).parent))
    from train_sft_lsreasoning_v2 import train_sft
    from train_grpo_lsreasoning_v2 import train_grpo
    from evaluate_lsreasoning_v2 import evaluate_model

    results = {}

    # Experiment 1: SFT only
    sft_output = "outputs/sft_lsreasoning"
    if not args.skip_sft:
        run_step("SFT Training", train_sft,
                 model_name=args.model_name,
                 train_file=str(TRAIN_FILE),
                 output_dir=sft_output,
                 max_seq_length=args.max_seq_length,
                 epochs=args.sft_epochs,
                 lr=args.sft_lr,
                 batch_size=args.sft_batch_size,
                 grad_accum=args.sft_grad_accum,
                 seed=args.seed)

    results["SFT"] = run_step("Evaluate SFT", evaluate_model,
                              model_path=f"{sft_output}/final",
                              test_file=str(TEST_FILE),
                              num_eval=args.num_eval,
                              verbose=args.verbose)

    # Experiment 2: SFT -> GRPO (dense)
    grpo_dense_output = "outputs/grpo_lsreasoning_dense"
    if not args.skip_grpo_dense:
        run_step("GRPO Training (Dense Reward)", train_grpo,
                 sft_path=f"{sft_output}/final",
                 model_name=args.model_name,
                 train_file=str(TRAIN_FILE),
                 output_dir=grpo_dense_output,
                 reward_mode="dense",
                 max_seq_length=args.max_seq_length,
                 epochs=args.grpo_epochs,
                 lr=args.grpo_lr,
                 batch_size=args.grpo_batch_size,
                 num_generations=args.grpo_num_generations,
                 beta=args.grpo_beta,
                 seed=args.seed)

    results["GRPO_dense"] = run_step("Evaluate GRPO (Dense)", evaluate_model,
                                     model_path=f"{grpo_dense_output}/final",
                                     test_file=str(TEST_FILE),
                                     num_eval=args.num_eval,
                                     verbose=args.verbose)

    # Experiment 3: SFT -> GRPO (sparse)
    grpo_sparse_output = "outputs/grpo_lsreasoning_sparse"
    if not args.skip_grpo_sparse:
        run_step("GRPO Training (Sparse Reward)", train_grpo,
                 sft_path=f"{sft_output}/final",
                 model_name=args.model_name,
                 train_file=str(TRAIN_FILE),
                 output_dir=grpo_sparse_output,
                 reward_mode="sparse",
                 max_seq_length=args.max_seq_length,
                 epochs=args.grpo_epochs,
                 lr=args.grpo_lr,
                 batch_size=args.grpo_batch_size,
                 num_generations=args.grpo_num_generations,
                 beta=args.grpo_beta,
                 seed=args.seed)

    results["GRPO_sparse"] = run_step("Evaluate GRPO (Sparse)", evaluate_model,
                                      model_path=f"{grpo_sparse_output}/final",
                                      test_file=str(TEST_FILE),
                                      num_eval=args.num_eval,
                                      verbose=args.verbose)

    # Save results to JSON for plotting
    os.makedirs("outputs", exist_ok=True)
    results_file = "outputs/experiment_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Generate comparison charts on wandb
    from plot_comparison_wandb import plot_comparison
    plot_comparison(results)

    # Final comparison (console)
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"\n  {'Method':<25} {'Exact Match':<15} {'Close Match':<15} {'Format':<10}")
    print(f"  {'_' * 65}")
    for method, res in results.items():
        if res:
            em = res.get("exact_match", 0)
            cm = res.get("close_match", 0)
            fmt = res.get("format_score", 0)
            print(f"  {method:<25} {em:>6.1%}         {cm:>6.1%}         {fmt:>6.1%}")
    print()
    print("  Expected: SFT+GRPO(dense) > SFT+GRPO(sparse) > SFT only")
    print("=" * 60)


if __name__ == "__main__":
    main()
