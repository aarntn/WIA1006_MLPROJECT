"""
01b_eda_polished.py — Report-ready EDA plots.

Replaces the visual output of 01_eda.py with chart-by-chart improvements:
- every element labeled with exact numbers
- fewer, bigger, focused figures (6 total)
- report-document sizing (portrait, 150 dpi)
- meaningful chart types per data kind

Run from project root:
    python scripts/01b_eda_polished.py

Outputs to reports/figures/polished/
"""
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data import load_raw, get_feature_cols, MULTI_LABEL_COL, TARGET
from src.preprocessing import parse_interest_tags

# ---- setup ----
ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "reports" / "figures" / "polished"
FIGDIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "font.family": "DejaVu Sans",  # handles subscripts + unicode cleanly
})

# Color palette — consistent across all plots
COL_PRIMARY = "#2E5C8A"      # deep blue
COL_ACCENT = "#C44E52"       # red
COL_MUTED = "#8FA8C4"        # light blue
COL_HIGHLIGHT = "#E8A33D"    # orange
COL_GRID = "#E8E8E8"


def save(fig, name: str):
    """Save with tight bbox and report a confirmation."""
    path = FIGDIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [saved] polished/{name}.png")


def label_bars(ax, bars, values, fmt="{:.0f}", offset=3, fontsize=8, color="black"):
    """Annotate bars with values. Works for vertical bars."""
    for bar, val in zip(bars, values):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            fmt.format(val),
            ha="center", va="bottom", fontsize=fontsize, color=color,
        )


def label_hbars(ax, bars, values, fmt="{:.0f}", offset_frac=0.01, fontsize=8, color="black"):
    """Annotate horizontal bars with values."""
    xmax = ax.get_xlim()[1]
    offset = xmax * offset_frac
    for bar, val in zip(bars, values):
        w = bar.get_width()
        ax.text(
            w + offset,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(val),
            ha="left", va="center", fontsize=fontsize, color=color,
        )


# =====================================================================
# LOAD
# =====================================================================
df = load_raw(extended=True)
num_cols, cat_cols = get_feature_cols(df)

print("=" * 70)
print("Generating polished EDA plots")
print(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")
print("=" * 70)


# =====================================================================
# PLOT 1 — Target distribution (horizontal bars, count + %)
# =====================================================================
counts = df[TARGET].value_counts().sort_values()
pct = (counts / counts.sum() * 100)

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.barh(counts.index, counts.values, color=COL_PRIMARY, edgecolor="white")
ax.set_title(f"match_outcome distribution (N = {counts.sum():,})")
ax.set_xlabel("Count")
ax.set_xlim(0, counts.max() * 1.15)

# Label each bar with count and percentage
for bar, count, p in zip(bars, counts.values, pct.values):
    ax.text(
        bar.get_width() + counts.max() * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{count:,} ({p:.1f}%)",
        ha="left", va="center", fontsize=9,
    )

# Annotation highlighting the balance
balance_ratio = counts.max() / counts.min()
ax.text(
    0.98, 0.02,
    f"Class balance ratio: {balance_ratio:.3f}\n(1.000 = perfectly balanced)",
    transform=ax.transAxes, ha="right", va="bottom",
    fontsize=8.5, style="italic", color="#666",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="#DDD"),
)
ax.grid(axis="x", alpha=0.3)
ax.set_axisbelow(True)
save(fig, "01_target_distribution")


# =====================================================================
# PLOT 2 — Numeric distributions, grouped by kind, with mean/median
# =====================================================================
# Group features by semantic type
groups = {
    "Continuous": ["app_usage_time_min", "age", "weight_kg", "height_cm",
                   "bio_length", "likes_received"],
    "Discrete / Count": ["mutual_matches", "message_sent_count", "profile_pics_count",
                          "last_active_hour"],
    "Bounded ratio": ["swipe_right_ratio", "emoji_usage_rate"],
}

fig, axes = plt.subplots(4, 3, figsize=(11, 13))
axes = axes.flatten()

plot_idx = 0
for group_label, cols in groups.items():
    for col in cols:
        if col not in df.columns:
            continue
        ax = axes[plot_idx]
        mean = df[col].mean()
        median = df[col].median()
        std = df[col].std()

        # Histogram
        ax.hist(df[col], bins=30, color=COL_PRIMARY, edgecolor="white", alpha=0.85)

        # Mean and median lines with labeled values
        ax.axvline(mean, color=COL_ACCENT, linestyle="-", linewidth=1.8,
                   label=f"mean={mean:.2f}")
        ax.axvline(median, color=COL_HIGHLIGHT, linestyle="--", linewidth=1.8,
                   label=f"median={median:.2f}")
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)

        ax.set_title(f"{col}  [{group_label}]", fontsize=10)
        ax.set_ylabel("count", fontsize=8)
        ax.tick_params(labelsize=8)

        # Stats annotation
        ax.text(
            0.02, 0.97,
            f"n={len(df):,}\nσ={std:.2f}\nmin={df[col].min():.1f}\nmax={df[col].max():.1f}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#DDD", alpha=0.9),
        )
        plot_idx += 1

# Hide unused axes
for i in range(plot_idx, len(axes)):
    axes[i].set_visible(False)

fig.suptitle("Numeric feature distributions with exact central tendencies",
             fontsize=12, fontweight="bold", y=1.00)
fig.tight_layout()
save(fig, "02_numeric_distributions")


# =====================================================================
# PLOT 3 — Correlation heatmap, lower triangle, real correlations only
# =====================================================================
corr = df[num_cols].corr()

# Mask upper triangle + near-zero correlations
mask_upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
mask_noise = (np.abs(corr.values) < 0.05)
mask = mask_upper | mask_noise
# Keep diagonal visible (the 1.0 values)
np.fill_diagonal(mask, False)

fig, ax = plt.subplots(figsize=(10, 8.5))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    square=True, linewidths=0.5, linecolor="white",
    cbar_kws={"shrink": 0.6, "label": "Pearson r"},
    annot_kws={"fontsize": 9, "weight": "bold"},
    ax=ax,
)
ax.set_title("Numeric feature correlation matrix (|r| ≥ 0.05 shown)",
             fontsize=12, pad=15)
plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
plt.setp(ax.get_yticklabels(), rotation=0)

# Highlight the two meaningful correlations
# Find indices
cols_list = list(corr.columns)
notable = [("height_cm", "weight_kg"), ("likes_received", "mutual_matches")]
for a, b in notable:
    if a in cols_list and b in cols_list:
        i = cols_list.index(a)
        j = cols_list.index(b)
        # Only highlight if in lower triangle
        if i > j:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                        edgecolor=COL_HIGHLIGHT, lw=3))
        else:
            ax.add_patch(plt.Rectangle((i, j), 1, 1, fill=False,
                                        edgecolor=COL_HIGHLIGHT, lw=3))

# Footer note
ax.text(
    0.5, -0.30,
    "Orange box = meaningful correlation. Empty cells = |r| < 0.05 (negligible). "
    "All non-orange relationships are near-zero, consistent with synthetic data.",
    transform=ax.transAxes, ha="center", fontsize=8.5, style="italic", color="#555",
)
save(fig, "03_correlation_heatmap")


# =====================================================================
# PLOT 4 — Categorical balance overview + the one imbalanced feature
# =====================================================================
# Compute imbalance metric (max/min ratio) for each categorical
imbalance = {}
for col in cat_cols:
    vc = df[col].value_counts()
    imbalance[col] = vc.max() / vc.min()

imb_series = pd.Series(imbalance).sort_values()
most_imbalanced = imb_series.idxmax()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.3]})

# LEFT: imbalance ratio per feature
bars = ax1.barh(
    imb_series.index, imb_series.values,
    color=[COL_ACCENT if v > 2 else COL_MUTED for v in imb_series.values],
    edgecolor="white",
)
ax1.set_title("Class imbalance per categorical feature\n(ratio = max_count / min_count)")
ax1.set_xlabel("Imbalance ratio (1.0 = perfectly balanced)")
ax1.axvline(1.0, color="#999", linestyle=":", linewidth=1)
for bar, val in zip(bars, imb_series.values):
    ax1.text(
        val + imb_series.max() * 0.015, bar.get_y() + bar.get_height() / 2,
        f"{val:.2f}", ha="left", va="center", fontsize=9,
    )
ax1.set_xlim(0, imb_series.max() * 1.18)
ax1.grid(axis="x", alpha=0.3)
ax1.set_axisbelow(True)

# RIGHT: the imbalanced one, exact counts
vc = df[most_imbalanced].value_counts()
pct_vc = (vc / vc.sum() * 100)
bars = ax2.barh(vc.index, vc.values, color=COL_PRIMARY, edgecolor="white")
ax2.set_title(f"'{most_imbalanced}' — the only imbalanced feature")
ax2.set_xlabel("Count")
ax2.set_xlim(0, vc.max() * 1.18)
ax2.grid(axis="x", alpha=0.3)
ax2.set_axisbelow(True)
for bar, count, p in zip(bars, vc.values, pct_vc.values):
    ax2.text(
        bar.get_width() + vc.max() * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{count:,} ({p:.1f}%)",
        ha="left", va="center", fontsize=9,
    )

fig.suptitle("Categorical features: most are balanced, one is not",
             fontsize=13, fontweight="bold", y=1.00)
fig.tight_layout()
save(fig, "04_categorical_overview")


# =====================================================================
# PLOT 5 — Interest tags: top 15 only, clean and labeled
# =====================================================================
parsed = parse_interest_tags(df[MULTI_LABEL_COL])
tag_counts = Counter(t for row in parsed for t in row)

top_n = 15
top_tags = dict(tag_counts.most_common(top_n))
other_tags = dict(tag_counts.most_common()[top_n:])
other_min = min(other_tags.values())
other_max = max(other_tags.values())
other_count = len(other_tags)

fig, ax = plt.subplots(figsize=(10, 7))
tags = list(reversed(top_tags.keys()))
vals = list(reversed(top_tags.values()))
pct = [v / sum(tag_counts.values()) * 100 for v in vals]

bars = ax.barh(tags, vals, color=COL_PRIMARY, edgecolor="white")
ax.set_title(f"Top {top_n} interest tags (out of {len(tag_counts)} unique tags)")
ax.set_xlabel("Occurrences across all users")
ax.set_xlim(0, max(vals) * 1.15)

for bar, val, p in zip(bars, vals, pct):
    ax.text(
        bar.get_width() + max(vals) * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{val:,} ({p:.2f}%)",
        ha="left", va="center", fontsize=9,
    )

# Summary box
ax.text(
    0.98, 0.02,
    f"Remaining {other_count} tags: range {other_min:,}–{other_max:,} occurrences each.\n"
    f"All 49 tags ≈ uniform — synthetic-data fingerprint.",
    transform=ax.transAxes, ha="right", va="bottom",
    fontsize=9, style="italic", color="#555",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="#DDD"),
)
ax.grid(axis="x", alpha=0.3)
ax.set_axisbelow(True)
save(fig, "05_interest_tags_top15")


# =====================================================================
# PLOT 6 — Feature means by outcome (the "no signal" visual, with numbers)
# =====================================================================
# Pick 4 most interesting features
feats = ["likes_received", "mutual_matches", "message_sent_count", "bio_length"]

# Define a meaningful outcome order (most negative to most positive)
outcome_order = [
    "Blocked", "Catfished", "Ghosted", "Chat Ignored",
    "No Action", "One-sided Like",
    "Instant Match", "Mutual Match", "Date Happened", "Relationship Formed",
]
outcome_order = [o for o in outcome_order if o in df[TARGET].unique()]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()

for ax, col in zip(axes, feats):
    means = df.groupby(TARGET)[col].mean().reindex(outcome_order)
    stds = df.groupby(TARGET)[col].std().reindex(outcome_order)
    overall_mean = df[col].mean()

    bars = ax.bar(
        range(len(means)), means.values,
        yerr=stds.values, capsize=3,
        color=COL_PRIMARY, edgecolor="white",
        error_kw={"alpha": 0.5, "linewidth": 1},
    )
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(means.index, rotation=40, ha="right", fontsize=8.5)
    ax.set_ylabel(col)
    ax.set_title(f"Mean {col} per match_outcome (± std)")

    # Label each bar with the exact mean
    for bar, m in zip(bars, means.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + stds.values.max() * 0.05,
            f"{m:.1f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    # Overall mean reference line
    ax.axhline(overall_mean, color=COL_ACCENT, linestyle="--", linewidth=1.3,
               label=f"overall mean = {overall_mean:.1f}")
    ax.legend(loc="upper right", fontsize=8)

    # Show mean-range compactly
    spread = means.max() - means.min()
    ax.text(
        0.02, 0.97,
        f"Range of means: {spread:.2f}\n"
        f"Std of means: {means.std():.3f}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8, family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#DDD", alpha=0.9),
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

fig.suptitle(
    "If features had signal, means would differ across outcomes. They don't.",
    fontsize=13, fontweight="bold", y=1.00,
)
fig.tight_layout()
save(fig, "06_means_by_outcome")


# =====================================================================
# PLOT 7/8 — Rule-based engagement segments + emoji profile
# =====================================================================
# Create 4 disjoint, human-readable behavioral segments from existing labels
segment_conditions = [
    (
        (df["app_usage_time_label"].isin(["Addicted", "Extreme User"]))
        & (df["swipe_right_label"].isin(["Optimistic", "Swipe Maniac"]))
        & (df["relationship_intent"].eq("Serious Relationship")),
        "Heavy swipers, serious",
    ),
    (
        (df["app_usage_time_label"].isin(["Addicted", "Extreme User"]))
        & (df["swipe_right_label"].eq("Choosy"))
        & (df["relationship_intent"].isin(["Serious Relationship", "Exploring"])),
        "Heavy users, choosy",
    ),
    (
        (df["app_usage_time_label"].isin(["Moderate", "High"]))
        & (df["swipe_right_label"].eq("Balanced"))
        & (df["relationship_intent"].isin(["Casual Dating", "Exploring"])),
        "Balanced casuals",
    ),
    (
        (df["app_usage_time_label"].isin(["Barely", "Very Low", "Low"]))
        & (df["swipe_right_label"].isin(["Balanced", "Choosy"]))
        & (df["relationship_intent"].isin(["Friends Only", "Networking"])),
        "Low-activity social",
    ),
]

df["engagement_segment"] = "Other"
for condition, label in segment_conditions:
    df.loc[(df["engagement_segment"] == "Other") & condition, "engagement_segment"] = label

segment_order = [label for _, label in segment_conditions]
seg_df = df[df["engagement_segment"].isin(segment_order)].copy()

# Emoji usage profile buckets
emoji_bins = [-0.01, 0.33, 0.66, 1.00]
emoji_labels = ["Low emoji", "Medium emoji", "High emoji"]
seg_df["emoji_profile"] = pd.cut(
    seg_df["emoji_usage_rate"], bins=emoji_bins, labels=emoji_labels, include_lowest=True
)

segment_summary = (
    seg_df.groupby("engagement_segment")
    .agg(
        sample_size=("engagement_segment", "size"),
        median_likes_received=("likes_received", "median"),
        median_mutual_matches=("mutual_matches", "median"),
        median_message_sent_count=("message_sent_count", "median"),
        median_emoji_usage_rate=("emoji_usage_rate", "median"),
    )
    .reindex(segment_order)
)

emoji_profile_pct = (
    seg_df.groupby(["engagement_segment", "emoji_profile"], observed=False)
    .size()
    .unstack(fill_value=0)
    .reindex(segment_order)
)
emoji_profile_pct = emoji_profile_pct.div(emoji_profile_pct.sum(axis=1), axis=0) * 100

print("\n" + "=" * 70)
print("Rule-based engagement segments (4 disjoint segments)")
print("=" * 70)
print(segment_summary.round(2).to_string())
print("\nEmoji usage profile (% within segment):")
print(emoji_profile_pct.round(1).to_string())

# ---- grouped bar plot: medians + sample size ----
metrics_plot = segment_summary[
    ["median_likes_received", "median_mutual_matches", "median_message_sent_count"]
].reset_index()
metrics_long = metrics_plot.melt(
    id_vars="engagement_segment",
    var_name="metric",
    value_name="median_value",
)
metric_name_map = {
    "median_likes_received": "Likes received",
    "median_mutual_matches": "Mutual matches",
    "median_message_sent_count": "Messages sent",
}
metrics_long["metric"] = metrics_long["metric"].map(metric_name_map)

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [2.2, 1]}
)
sns.barplot(
    data=metrics_long,
    x="engagement_segment",
    y="median_value",
    hue="metric",
    palette=[COL_PRIMARY, COL_HIGHLIGHT, COL_ACCENT],
    ax=ax1,
)
ax1.set_title("Segment medians across engagement outcomes")
ax1.set_xlabel("")
ax1.set_ylabel("Median value")
ax1.tick_params(axis="x", rotation=15)
ax1.legend(title="")
ax1.grid(axis="y", alpha=0.3)

sample_sizes = segment_summary["sample_size"].reset_index()
sns.barplot(
    data=sample_sizes,
    x="engagement_segment",
    y="sample_size",
    color=COL_MUTED,
    ax=ax2,
)
ax2.set_title("Sample size by segment")
ax2.set_xlabel("Segment")
ax2.set_ylabel("Users")
ax2.tick_params(axis="x", rotation=15)
ax2.grid(axis="y", alpha=0.3)

for p in ax2.patches:
    ax2.annotate(
        f"{int(p.get_height()):,}",
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha="center",
        va="bottom",
        fontsize=8,
        xytext=(0, 3),
        textcoords="offset points",
    )

fig.tight_layout()
save(fig, "07_segment_grouped_bars")

# ---- boxplots + emoji stacked bars ----
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.5, 1]}
)
sns.boxplot(
    data=seg_df,
    x="engagement_segment",
    y="message_sent_count",
    order=segment_order,
    showfliers=False,
    color=COL_PRIMARY,
    ax=ax1,
)
ax1.set_title("Message volume distribution by segment")
ax1.set_xlabel("Segment")
ax1.set_ylabel("message_sent_count")
ax1.tick_params(axis="x", rotation=15)
ax1.grid(axis="y", alpha=0.3)

emoji_profile_pct.plot(
    kind="bar",
    stacked=True,
    color=[COL_MUTED, COL_HIGHLIGHT, COL_ACCENT],
    edgecolor="white",
    ax=ax2,
)
ax2.set_title("Emoji usage profile by segment")
ax2.set_xlabel("Segment")
ax2.set_ylabel("Percent within segment")
ax2.tick_params(axis="x", rotation=15)
ax2.legend(title="", loc="upper right")
ax2.set_ylim(0, 100)
ax2.grid(axis="y", alpha=0.3)

fig.tight_layout()
save(fig, "08_segment_boxplot_emoji_profile")

# ---- short textual insight block ----
top_messages = segment_summary["median_message_sent_count"].idxmax()
top_matches = segment_summary["median_mutual_matches"].idxmax()
top_likes = segment_summary["median_likes_received"].idxmax()
low_messages = segment_summary["median_message_sent_count"].idxmin()

print("\nSegment insights:")
print(f"- Highest median messages: {top_messages}")
print(f"- Highest median mutual matches: {top_matches}")
print(f"- Highest median likes received: {top_likes}")
print(f"- Lowest median messages (least engaged): {low_messages}")

print()
print("=" * 70)
print(f"Done. 8 polished plots saved to: {FIGDIR.relative_to(ROOT)}")
print("=" * 70)
