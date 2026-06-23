"""
Exploratory Data Analysis for Math Datasets
============================================
Run with: jupyter notebook or python notebooks/01_eda.py
"""

# %% [markdown]
# # Dataset EDA: OpenWebMath + NuminaMath-CoT

# %%
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import load_from_disk
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Load Datasets

# %%
RAW_DIR = Path("data/raw")

print("Loading datasets...")
openwebmath = load_from_disk(str(RAW_DIR / "openwebmath"))
numinamath = load_from_disk(str(RAW_DIR / "numinamath_cot"))

print(f"OpenWebMath: {len(openwebmath)} samples")
print(f"NuminaMath-CoT: {len(numinamath)} samples")

# %% [markdown]
# ## 2. OpenWebMath Analysis (CPT data)

# %%
# Text length distribution
lengths = [len(text.split()) for text in openwebmath["text"][:10000]]
print(f"Word count stats:")
print(f"  Mean: {np.mean(lengths):.0f}")
print(f"  Median: {np.median(lengths):.0f}")
print(f"  P5: {np.percentile(lengths, 5):.0f}")
print(f"  P95: {np.percentile(lengths, 95):.0f}")

# %%
# Sample inspection
print("\n--- Sample 0 ---")
print(openwebmath[0]["text"][:500])
print("\n--- Sample 100 ---")
print(openwebmath[100]["text"][:500])

# %% [markdown]
# ## 3. NuminaMath-CoT Analysis (SFT data)

# %%
# Problem/solution length analysis
problem_lens = [len(s.split()) for s in numinamath["problem"][:10000]]
solution_lens = [len(s.split()) for s in numinamath["solution"][:10000]]

print(f"\nProblem lengths: mean={np.mean(problem_lens):.0f}, median={np.median(problem_lens):.0f}")
print(f"Solution lengths: mean={np.mean(solution_lens):.0f}, median={np.median(solution_lens):.0f}")
print(f"Solution/Problem ratio: {np.mean(solution_lens)/np.mean(problem_lens):.1f}x")

# %%
# Category distribution (if available)
if "source" in numinamath.column_names:
    sources = pd.Series(numinamath["source"]).value_counts()
    print("\nData sources:")
    print(sources.head(10))

# %%
# Sample SFT pair
print("\n--- Problem ---")
print(numinamath[0]["problem"])
print("\n--- Solution ---")
print(numinamath[0]["solution"][:1000])

# %% [markdown]
# ## 4. Quality Indicators

# %%
import re

def count_latex(text):
    return len(re.findall(r"\$[^$]+\$|\\\[.*?\\\]", text, re.DOTALL))

# LaTeX density in CPT data
latex_counts = [count_latex(text) for text in openwebmath["text"][:5000]]
print(f"\nLaTeX expressions per document:")
print(f"  Mean: {np.mean(latex_counts):.1f}")
print(f"  Zero LaTeX: {sum(1 for c in latex_counts if c == 0)} / {len(latex_counts)}")

# %%
# Check for boxed answers in SFT solutions
has_boxed = sum(1 for s in numinamath["solution"] if "\\boxed" in s)
print(f"\nSolutions with \\boxed answer: {has_boxed}/{len(numinamath)} ({100*has_boxed/len(numinamath):.1f}%)")

# %% [markdown]
# ## 5. Summary & Recommendations

# %%
print("""
=== EDA Summary ===

CPT (OpenWebMath):
- Large corpus of mathematical text
- Good LaTeX density (most documents contain formulas)
- Filter criteria: min quality score > 0.3, min length > 50 words

SFT (NuminaMath-CoT):
- Instruction-response pairs with chain-of-thought
- Solutions are typically 3-5x longer than problems
- Most solutions include \\boxed{} final answers
- Good for teaching step-by-step reasoning

Recommendations:
1. Filter CPT data by math content density
2. Ensure SFT solutions end with clear final answers
3. Consider sequence packing for efficiency
4. Watch for overly long sequences (truncate at 4096 for CPT, 2048 for SFT)
""")
