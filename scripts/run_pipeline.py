"""Full pipeline: Generate Data -> CPT -> SFT -> Inference.

This script runs the complete fine-tuning pipeline end-to-end.

Usage:
    # Full pipeline with local data generation
    python scripts/run_pipeline.py

    # With Gemini API for data generation
    python scripts/run_pipeline.py --api gemini --api-key YOUR_KEY

    # Skip data generation (use existing data)
    python scripts/run_pipeline.py --skip-datagen

    # Skip CPT (SFT only)
    python scripts/run_pipeline.py --skip-cpt
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str):
    """Run a command and print its output."""
    print(f"\n{'=' * 60}")
    print(f"STEP: {description}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\nERROR: {description} failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n✓ {description} completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Run full fine-tuning pipeline")
    parser.add_argument("--api", choices=["gemini", "openai", "local"], default="local")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--skip-datagen", action="store_true")
    parser.add_argument("--skip-cpt", action="store_true")
    parser.add_argument("--num-cpt", type=int, default=150)
    parser.add_argument("--num-sft", type=int, default=150)
    parser.add_argument("--cpt-epochs", type=int, default=3)
    parser.add_argument("--sft-epochs", type=int, default=5)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("   QWEN3-0.6B FINE-TUNING PIPELINE")
    print("   Linear Equation Solving (Vietnamese)")
    print("=" * 60)
    print(f"\n  Model: {args.model_name}")
    print(f"  Data API: {args.api}")
    print(f"  CPT epochs: {args.cpt_epochs}")
    print(f"  SFT epochs: {args.sft_epochs}")
    print(f"  Skip datagen: {args.skip_datagen}")
    print(f"  Skip CPT: {args.skip_cpt}")

    # Step 1: Generate data
    if not args.skip_datagen:
        datagen_cmd = [
            sys.executable, "scripts/generate_data.py",
            "--api", args.api,
            "--num-cpt", str(args.num_cpt),
            "--num-sft", str(args.num_sft),
        ]
        if args.api_key:
            datagen_cmd.extend(["--api-key", args.api_key])
        run_command(datagen_cmd, "Data Generation")
    else:
        # Check data exists
        cpt_path = Path("data/raw/cpt_linear_equations.jsonl")
        sft_path = Path("data/raw/sft_linear_equations.jsonl")
        if not cpt_path.exists() or not sft_path.exists():
            print("ERROR: Data files not found. Run without --skip-datagen first.")
            sys.exit(1)
        print("\nUsing existing data files.")

    # Step 2: CPT
    cpt_output = "outputs/cpt_linear_eq"
    if not args.skip_cpt:
        cpt_cmd = [
            sys.executable, "scripts/train_cpt_unsloth.py",
            "--model-name", args.model_name,
            "--epochs", str(args.cpt_epochs),
            "--output-dir", cpt_output,
        ]
        run_command(cpt_cmd, "Continual Pre-Training (CPT)")

    # Step 3: SFT
    sft_output = "outputs/sft_linear_eq"
    sft_cmd = [
        sys.executable, "scripts/train_sft_unsloth.py",
        "--model-name", args.model_name,
        "--epochs", str(args.sft_epochs),
        "--output-dir", sft_output,
    ]
    if not args.skip_cpt and Path(f"{cpt_output}/final").exists():
        sft_cmd.extend(["--cpt-path", f"{cpt_output}/final"])
    run_command(sft_cmd, "Supervised Fine-Tuning (SFT)")

    # Step 4: Inference test
    inference_cmd = [
        sys.executable, "scripts/inference.py",
        "--model-path", f"{sft_output}/final",
    ]
    run_command(inference_cmd, "Inference Test")

    print("\n" + "=" * 60)
    print("   PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\n  CPT model: {cpt_output}/final")
    print(f"  SFT model: {sft_output}/final")
    print(f"\n  To test interactively:")
    print(f"    python scripts/inference.py --model-path {sft_output}/final --interactive")
    print()


if __name__ == "__main__":
    main()
