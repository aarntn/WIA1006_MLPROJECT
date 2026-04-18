"""
01_eda.py — Basic exploratory data analysis.

Run from project root:
    python scripts/01_eda.py

Outputs:
    - printed summaries to stdout
    - distribution plots to reports/figures/
    - correlation matrix to reports/figures/
"""
import sys
from pathlib import Path

# Make src/ importable when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter

from src.data import load_raw, get_feature_cols, MULTI_LABEL_COL, TARGET
from src.preprocessing import parse_interest_tags

# ---- setup ----
ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 110


def section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---- load ----
df = load_raw(extended=True)
num_cols, cat_cols = get_feature_cols(df)

section("SHAPE & DTYPES")
print(f"Shape: {df.shape}")
print(f"Numeric cols ({len(num_cols)}): {num_cols}")
print(f"Categorical cols ({len(cat_cols)}): {cat_cols}")

section("MISSING VALUES")
missing = df.isna().sum()
print(missing[missing > 0] if missing.sum() else "No missing values.")

section("TARGET DISTRIBUTION")
counts = df[TARGET].value_counts()
print(counts)
print(f"\nClass balance ratio (max/min): {counts.max() / counts.min():.3f}")
print("(values near 1.0 = perfectly balanced — a smoking gun for synthetic data)")

# ---- numeric summaries ----
section("NUMERIC FEATURE SUMMARY")
print(df[num_cols].describe().round(2))

# ---- categorical cardinality ----
section("CATEGORICAL CARDINALITY")
for c in cat_cols:
    print(f"  {c:25s} {df[c].nunique():4d} unique")

# ---- interest tags ----
section("INTEREST TAGS")
parsed = parse_interest_tags(df[MULTI_LABEL_COL])
tag_counts = Counter(t for row in parsed for t in row)
print(f"Total unique tags: {len(tag_counts)}")
print(f"Top 10: {tag_counts.most_common(10)}")
print(f"Bottom 5:  {tag_counts.most_common()[-5:]}")
print("(near-uniform tag frequencies = another synthetic-data fingerprint)")

# ==================== PLOTS ====================
section("GENERATING PLOTS — saving to reports/figures/")

# 1. Target distribution
fig, ax = plt.subplots(figsize=(11, 5))
counts.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
ax.set_title("match_outcome distribution")
ax.set_xlabel("count")
plt.tight_layout()
plt.savefig(FIGDIR / "01_target_distribution.png")
plt.close()
print("  [saved] 01_target_distribution.png")

# 2. Numeric distributions (grid of histograms)
n = len(num_cols)
fig, axes = plt.subplots(nrows=(n + 2) // 3, ncols=3, figsize=(15, 3 * ((n + 2) // 3)))
for ax, col in zip(axes.flat, num_cols):
    df[col].hist(bins=40, ax=ax, color="#4C72B0", edgecolor="white")
    ax.set_title(col)
for ax in axes.flat[n:]:
    ax.set_visible(False)
plt.tight_layout()
plt.savefig(FIGDIR / "02_numeric_distributions.png")
plt.close()
print("  [saved] 02_numeric_distributions.png")

# 3. Correlation heatmap on numeric features
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, cbar_kws={"shrink": 0.7})
ax.set_title("Numeric feature correlation matrix")
plt.tight_layout()
plt.savefig(FIGDIR / "03_correlation_heatmap.png")
plt.close()
print("  [saved] 03_correlation_heatmap.png")

# 4. Categorical distributions (sampled, top-6 most balanced)
cats_to_plot = cat_cols[:6]
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
for ax, col in zip(axes.flat, cats_to_plot):
    vc = df[col].value_counts()
    vc.plot(kind="bar", ax=ax, color="#55A868")
    ax.set_title(col)
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(FIGDIR / "04_categorical_distributions.png")
plt.close()
print("  [saved] 04_categorical_distributions.png")

# 5. Interest tag frequency
fig, ax = plt.subplots(figsize=(14, 6))
tag_df = pd.Series(dict(tag_counts)).sort_values(ascending=True)
tag_df.plot(kind="barh", ax=ax, color="#C44E52")
ax.set_title(f"Interest tag frequencies ({len(tag_counts)} unique tags)")
ax.set_xlabel("count")
plt.tight_layout()
plt.savefig(FIGDIR / "05_interest_tag_frequencies.png")
plt.close()
print("  [saved] 05_interest_tag_frequencies.png")

# 6. Target x numeric feature — do outcomes look different across features?
# Boxplots of key behavioral features stratified by match_outcome
key_features = ["likes_received", "mutual_matches", "message_sent_count", "bio_length"]
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
for ax, col in zip(axes.flat, key_features):
    sns.boxplot(data=df, x=TARGET, y=col, ax=ax, showfliers=False)
    ax.tick_params(axis="x", rotation=45)
    ax.set_title(f"{col} by {TARGET}")
plt.tight_layout()
plt.savefig(FIGDIR / "06_feature_by_target.png")
plt.close()
print("  [saved] 06_feature_by_target.png")
print("  ^ in a dataset with signal, these boxplots would look DIFFERENT per outcome.")
print("    if all 10 boxes look nearly identical, that's your visual evidence of no signal.")

print("\nDone. Next: run scripts/02_signal_test.py to quantify the independence.")
