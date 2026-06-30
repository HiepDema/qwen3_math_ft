"""Full pipeline for quadratic equation fine-tuning.

Pipeline: Generate Data → SFT → GRPO → Evaluate

Usage:
    # Full pipeline with local data generation
    python scripts/run_quadratic_pipeline.py --mode local

    # Full pipeline with teacher model (requires A10 GPU)
    python scripts/run_quadratic_pipeline.py --mode teacher

    # Skip data generation (data already exists)
    python scripts/run_quadratic_pipeline.py --skip-datagen

    # Only evaluate existing model
    python scripts/run_quadratic_pipeline.py --eval-only --model-path outputs/grpo_quadratic_eq/final
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
    print(f"\n✓ {description} completed successfully")


def main():
    parser = argparse.ArgumentParser(description="Full quadratic equation fine-tuning pipeline")
    parser.add_argument("--mode", choices=["teacher", "local", "mixed"], default="local")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher-model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--sft-epochs", type=int, default=5)
    parser.add_argument("--grpo-epochs", type=int, default=1)
    parser.add_argument("--skip-datagen", action="store_true")
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--skip-grpo", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--num-tests", type=int, default=50)
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Quadratic Equation Fine-tuning Pipeline                ║")
    print("║  Model: Qwen3-0.6B + LoRA                              ║")
    print("║  Teacher: Qwen3-VL-30B-A3B-Instruct                    ║")
    print("║  Method: SFT → GRPO                                    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    if args.eval_only:
        model_path = args.model_path or "outputs/grpo_quadratic_eq/final"
        run_command(
            [sys.executable, "scripts/evaluate_quadratic.py",
             "--model-path", model_path,
             "--num-tests", str(args.num_tests),
             "--verbose"],
            "Evaluate Model"
        )
        return

    # Step 1: Generate Data
    if not args.skip_datagen:
        cmd = [
            sys.executable, "scripts/generate_quadratic_data.py",
            "--mode", args.mode,
            "--num-samples", str(args.num_samples),
        ]
        if args.mode in ["teacher", "mixed"]:
            cmd.extend(["--model-name", args.teacher_model])
        run_command(cmd, "Generate Quadratic Equation Data")

    # Step 2: SFT Training
    if not args.skip_sft:
        run_command(
            [sys.executable, "scripts/train_sft_quadratic.py",
             "--model-name", args.model_name,
             "--epochs", str(args.sft_epochs),
             "--data-path", "data/raw/sft_quadratic_equations.jsonl"],
            "SFT Training"
        )

    # Step 3: GRPO Training
    if not args.skip_grpo:
        sft_path = "outputs/sft_quadratic_eq/final"
        run_command(
            [sys.executable, "scripts/train_grpo_quadratic.py",
             "--sft-path", sft_path,
             "--epochs", str(args.grpo_epochs),
             "--num-prompts", str(args.num_samples)],
            "GRPO Training"
        )

    # Step 4: Evaluate
    model_path = args.model_path or "outputs/grpo_quadratic_eq/final"
    if not Path(model_path).exists():
        model_path = "outputs/sft_quadratic_eq/final"

    run_command(
        [sys.executable, "scripts/evaluate_quadratic.py",
         "--model-path", model_path,
         "--num-tests", str(args.num_tests),
         "--verbose"],
        "Final Evaluation"
    )

    print("\n" + "╔══════════════════════════════════════════════════════════╗")
    print("║  Pipeline Complete!                                      ║")
    print(f"║  Final model: {model_path:<42}║")
    print("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
