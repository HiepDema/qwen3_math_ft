"""Pipeline: SFT + GRPO (dense vs sparse) on LSReasoning-15000.

Runs 3 experiments and evaluates each:
1. SFT only
2. SFT → GRPO (dense reward)
3. SFT → GRPO (sparse reward)

Usage:
    python scripts/run_lsreasoning_pipeline.py
    python scripts/run_lsreasoning_pipeline.py --max-samples 5000 --max-prompts 3000
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, description):
    print("\n" + "=" * 60)
    print(f"STEP: {description}")
    print("=" * 60)
    print(f"$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
    if result.returncode != 0:
        print(f"\nFAILED: {description} (exit code {result.returncode})")
        sys.exit(1)
    print(f"\n✓ {description}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="SFT training samples")
    parser.add_argument("--max-prompts", type=int, default=3000,
                        help="GRPO training prompts")
    parser.add_argument("--sft-epochs", type=int, default=3)
    parser.add_argument("--grpo-epochs", type=int, default=1)
    parser.add_argument("--num-tests", type=int, default=200)
    parser.add_argument("--skip-sft", action="store_true",
                        help="Skip SFT if already trained")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  LSReasoning-15000 Pipeline                             ║")
    print("║  Compare: SFT vs SFT+GRPO(dense) vs SFT+GRPO(sparse)  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ──────────────────────────────────────────────
    # Experiment 1: SFT only
    # ──────────────────────────────────────────────
    if not args.skip_sft:
        run_cmd(
            [sys.executable, "scripts/train_sft_lsreasoning.py",
             "--model-name", args.model_name,
             "--max-samples", str(args.max_samples),
             "--epochs", str(args.sft_epochs),
             "--output-dir", "outputs/sft_lsreasoning"],
            "Experiment 1: SFT Training"
        )

    run_cmd(
        [sys.executable, "scripts/evaluate_lsreasoning.py",
         "--model-path", "outputs/sft_lsreasoning/final",
         "--num-tests", str(args.num_tests)],
        "Evaluate: SFT only"
    )

    # ──────────────────────────────────────────────
    # Experiment 2: SFT → GRPO (dense reward)
    # ──────────────────────────────────────────────
    run_cmd(
        [sys.executable, "scripts/train_grpo_lsreasoning.py",
         "--sft-path", "outputs/sft_lsreasoning/final",
         "--reward-mode", "dense",
         "--max-prompts", str(args.max_prompts),
         "--epochs", str(args.grpo_epochs),
         "--output-dir", "outputs/grpo_lsreasoning_dense"],
        "Experiment 2: GRPO (dense reward)"
    )

    run_cmd(
        [sys.executable, "scripts/evaluate_lsreasoning.py",
         "--model-path", "outputs/grpo_lsreasoning_dense/final",
         "--num-tests", str(args.num_tests)],
        "Evaluate: SFT + GRPO (dense)"
    )

    # ──────────────────────────────────────────────
    # Experiment 3: SFT → GRPO (sparse reward)
    # ──────────────────────────────────────────────
    run_cmd(
        [sys.executable, "scripts/train_grpo_lsreasoning.py",
         "--sft-path", "outputs/sft_lsreasoning/final",
         "--reward-mode", "sparse",
         "--max-prompts", str(args.max_prompts),
         "--epochs", str(args.grpo_epochs),
         "--output-dir", "outputs/grpo_lsreasoning_sparse"],
        "Experiment 3: GRPO (sparse reward)"
    )

    run_cmd(
        [sys.executable, "scripts/evaluate_lsreasoning.py",
         "--model-path", "outputs/grpo_lsreasoning_sparse/final",
         "--num-tests", str(args.num_tests)],
        "Evaluate: SFT + GRPO (sparse)"
    )

    # ──────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("═" * 60)
    print()
    print("  Models saved:")
    print("    outputs/sft_lsreasoning/final           → SFT only")
    print("    outputs/grpo_lsreasoning_dense/final    → SFT + GRPO (dense)")
    print("    outputs/grpo_lsreasoning_sparse/final   → SFT + GRPO (sparse)")
    print()
    print("  Re-run evaluation individually:")
    print("    python scripts/evaluate_lsreasoning.py --model-path <path> --verbose")
    print()
    print("  Expected: dense reward > sparse reward > SFT only")
    print("  Dense provides richer learning signal (format + proximity + reasoning)")
    print("═" * 60)


if __name__ == "__main__":
    main()
