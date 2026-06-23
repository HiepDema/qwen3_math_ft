#!/bin/bash
# Generate sample data locally (no API needed)
# Usage: bash scripts/generate_sample_data.sh
# Or:    python scripts/generate_data.py --api local

cd "$(dirname "$0")/.."
python scripts/generate_data.py --api local --num-cpt 150 --num-sft 150
echo "Done! Data saved to data/raw/"
