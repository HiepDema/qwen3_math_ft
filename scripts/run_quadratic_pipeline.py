"""Full pipeline for quadratic equation fine-tuning.

Supports 3 experiment modes to compare approaches:
1. SFT only
2. GRPO only (from base model, dense reward)
3. SFT → GRPO (recommended)

Usage:
    # Default: SFT → GRPO (dense reward)
    python scripts/run_quadratic_pipeline.py

    # SFT only (for comparison)
    python scripts/run_quadratic_pipeline.py --method sft

    # GRPO only with dense reward (no SFT, for comparison)
    python scripts/run_quadratic_pipeline.py --method grpo

    # GRPO with sparse reward (only correctness signal)
    python scripts/run_quadratic_pipeline.py --method grpo --reward-mode sparse

    # Compare all methods
    python scripts/run_quadratic_pipeline.py --method compare
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print status."""
    print("\n" + "=" * 60)
    print(f"STEP: {description}")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
    if result.returncode != 0:
        print(f"\nERROR: {description} failed with code {result.returncode}")
        sys.exit(1)
    print(f"\n✓ {description} completed")


def run_data_prep(args):
    """Step 1: Prepare data."""
    run_command(
        [sys.executable, "scripts/prepare_data.py",
         "--num-quadratic", str(args.num_samples),
         "--seed", "42"],
        "Prepare Training Data (VietJack + Quadratic)"
    )


def run_sft(args, output_dir="outputs/sft_quadratic_eq"):
    """Step 2a: SFT training."""
    run_command(
        [sys.executable, "scripts/train_sft_quadratic.py",
         "--model-name", args.model_name,
         "--data-path", "data/raw/sft_quadratic_equations.jsonl",
         "--output-dir", output_dir,
         "--epochs", str(args.sft_epochs)],
        "SFT Training"
    )


def run_grpo(args, sft_path=None, output_dir="outputs/grpo_quadratic_eq", reward_mode="dense"):
    """Step 2b: GRPO training."""
    cmd = [
        sys.executable, "scripts/train_grpo_quadratic.py",
        "--model-name", args.model_name,
        "--output-dir", output_dir,
        "--num-prompts", str(args.num_samples),
        "--epochs", str(args.grpo_epochs),
        "--reward-mode", reward_mode,
    ]
    if sft_path:
        cmd.extend(["--sft-path", sft_path])

    label = f"GRPO Training ({reward_mode} reward"
    if sft_path:
        label += ", after SFT"
    label += ")"
    run_command(cmd, label)


def run_eval(args, model_path, label=""):
    """Step 3: Evaluate."""
    run_command(
        [sys.executable, "scripts/evaluate_quadratic.py",
         "--model-path", model_path,
         "--num-tests", str(args.num_tests),
         "--verbose"],
        f"Evaluate{' - ' + label if label else ''}"
    )


def main():
    parser = argparse.ArgumentParser(description="Quadratic equation fine-tuning pipeline")
    parser.add_argument("--method", choices=["sft", "grpo", "sft_grpo", "compare"],
                        default="sft_grpo",
                        help="Training method (sft / grpo / sft_grpo / compare)")
    parser.add_argument("--reward-mode", choices=["dense", "sparse"], default="dense")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--sft-epochs", type=int, default=5)
    parser.add_argument("--grpo-epochs", type=int, default=1)
    parser.add_argument("--num-tests", type=int, default=50)
    parser.add_argument("--skip-datagen", action="store_true")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Quadratic Equation Fine-tuning Pipeline                ║")
    print(f"║  Method: {args.method:<47}║")
    print(f"║  Reward: {args.reward_mode:<47}║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Step 1: Data
    if not args.skip_datagen:
        run_data_prep(args)

    # Step 2+3: Train & Eval based on method
    if args.method == "sft":
        run_sft(args)
        run_eval(args, "outputs/sft_quadratic_eq/final", "SFT only")

    elif args.method == "grpo":
        run_grpo(args, sft_path=None, reward_mode=args.reward_mode)
        run_eval(args, "outputs/grpo_quadratic_eq/final", f"GRPO only ({args.reward_mode})")

    elif args.method == "sft_grpo":
        run_sft(args)
        run_grpo(args, sft_path="outputs/sft_quadratic_eq/final", reward_mode=args.reward_mode)
        run_eval(args, "outputs/grpo_quadratic_eq/final", f"SFT + GRPO ({args.reward_mode})")

    elif args.method == "compare":
        # Run all 3 methods and compare
        print("\n" + "▓" * 60)
        print("  EXPERIMENT 1: SFT only")
        print("▓" * 60)
        run_sft(args, output_dir="outputs/exp_sft_only")
        run_eval(args, "outputs/exp_sft_only/final", "SFT only")

        print("\n" + "▓" * 60)
        print("  EXPERIMENT 2: GRPO only (dense reward)")
        print("▓" * 60)
        run_grpo(args, sft_path=None, output_dir="outputs/exp_grpo_dense", reward_mode="dense")
        run_eval(args, "outputs/exp_grpo_dense/final", "GRPO dense only")

        print("\n" + "▓" * 60)
        print("  EXPERIMENT 3: GRPO only (sparse reward)")
        print("▓" * 60)
        run_grpo(args, sft_path=None, output_dir="outputs/exp_grpo_sparse", reward_mode="sparse")
        run_eval(args, "outputs/exp_grpo_sparse/final", "GRPO sparse only")

        print("\n" + "▓" * 60)
        print("  EXPERIMENT 4: SFT → GRPO (dense)")
        print("▓" * 60)
        run_grpo(args, sft_path="outputs/exp_sft_only/final",
                 output_dir="outputs/exp_sft_grpo_dense", reward_mode="dense")
        run_eval(args, "outputs/exp_sft_grpo_dense/final", "SFT + GRPO dense")

        print("\n" + "═" * 60)
        print("ALL EXPERIMENTS COMPLETE")
        print("═" * 60)
        print("  outputs/exp_sft_only/final       → SFT only")
        print("  outputs/exp_grpo_dense/final     → GRPO dense only")
        print("  outputs/exp_grpo_sparse/final    → GRPO sparse only")
        print("  outputs/exp_sft_grpo_dense/final → SFT + GRPO dense")
        print("\nRe-run eval on each to get comparison numbers.")

    print("\n✓ Pipeline complete!")


if __name__ == "__main__":
    main()
