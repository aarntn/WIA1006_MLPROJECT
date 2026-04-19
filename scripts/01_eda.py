"""
01_eda.py — Exploratory data analysis (unified, report-ready output).

Run from project root:
    python scripts/01_eda.py

Outputs:
    - printed summaries to stdout
    - reports/eda_quality_audit.csv     (machine-readable QA table)
    - reports/eda_findings.md           (markdown summary for report)
    - reports/figures/01_target_distribution.png
    - reports/figures/02_numeric_distributions.png
    - reports/figures/03_correlation_heatmap.png
    - reports/figures/04_categorical_overview.png
    - reports/figures/05_interest_tags_top15.png
    - reports/figures/06_means_by_outcome.png
    - reports/figures/07_pca_structure.png
    - reports/figures/08_segment_grouped_bars.png
    - reports/figures/09_segment_boxplot_emoji_profile.png
"""
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Make src/ importable when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from src.data import MULTI_LABEL_COL, TARGET, get_feature_cols, load_raw
from src.preprocessing import parse_interest_tags

# setup
ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"
FIGDIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# plotting helpers (inlined so no src.plotting dependency)
COL_PRIMARY = "#4C72B0"
COL_ACCENT = "#DD8452"
COL_HIGHLIGHT = "#C44E52"
COL_MUTED = "#9AA1A9"


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 140
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9


def save_fig(fig, name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {name}.png")


setup_plot_style()


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

# DATA QUALITY AUDIT helpers
def iqr_outlier_count(series: pd.Series) -> int:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return 0
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((clean < lower) | (clean > upper)).sum())


def invalid_value_found(series: pd.Series, is_numeric: bool) -> bool:
    if is_numeric:
        numeric = pd.to_numeric(series, errors="coerce")
        return bool(np.isinf(numeric).sum())
    if pd.api.types.is_string_dtype(series) or series.dtype == "object":
        stripped = series.dropna().astype(str).str.strip()
        return bool((stripped == "").sum())
    return False


def min_max_or_top_categories(series: pd.Series, is_numeric: bool, top_n: int = 3) -> str:
    if is_numeric:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty:
            return "NA"
        return f"min={clean.min():.3f}, max={clean.max():.3f}"

    vc = series.dropna().astype(str).value_counts()
    if vc.empty:
        return "NA"
    top = vc.head(top_n)
    return ", ".join([f"{idx} ({cnt})" for idx, cnt in top.items()])


def build_data_quality_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        missing = int(s.isna().sum())
        missing_pct = 100 * missing / len(df) if len(df) else 0
        rows.append(
            {
                "feature name": col,
                "dtype": str(s.dtype),
                "missing count (%)": f"{missing} ({missing_pct:.2f}%)",
                "unique count": int(s.nunique(dropna=True)),
                "min/max or top categories": min_max_or_top_categories(s, is_num),
                "invalid values found (yes/no)": "yes" if invalid_value_found(s, is_num) else "no",
                "outlier count (IQR rule for numeric)": iqr_outlier_count(s) if is_num else "",
            }
        )
    return pd.DataFrame(rows)

# LOAD
df = load_raw(extended=True)
num_cols, cat_cols = get_feature_cols(df)

section("SHAPE & DTYPES")
print(f"Shape: {df.shape}")
print(f"Numeric cols ({len(num_cols)}): {num_cols}")
print(f"Categorical cols ({len(cat_cols)}): {cat_cols}")

section("MISSING VALUES")
missing = df.isna().sum()
print(missing[missing > 0] if missing.sum() else "No missing values.")

# DATA QUALITY AUDIT
section("DATA QUALITY AUDIT")
audit_df = build_data_quality_audit(df)
print(audit_df.to_string(index=False))

audit_csv_path = REPORTS_DIR / "eda_quality_audit.csv"
audit_df.to_csv(audit_csv_path, index=False)
print(f"\nSaved audit table to: {audit_csv_path.relative_to(ROOT)}")

invalid_count = int((audit_df["invalid values found (yes/no)"] == "yes").sum())

numeric_audit = audit_df[audit_df["outlier count (IQR rule for numeric)"] != ""].copy()
numeric_audit["outlier count (IQR rule for numeric)"] = pd.to_numeric(
    numeric_audit["outlier count (IQR rule for numeric)"], errors="coerce"
).fillna(0).astype(int)
top_outliers = numeric_audit.sort_values(
    "outlier count (IQR rule for numeric)", ascending=False
).head(5)

md_lines = [
    "# EDA Findings",
    "",
    "## Data Quality Audit",
    "",
    f"- Dataset rows: **{len(df):,}**",
    f"- Dataset columns: **{df.shape[1]}**",
    f"- Features with missing values: **{int((df.isna().sum() > 0).sum())}**",
    f"- Features with invalid values found: **{invalid_count}**",
    "",
    "### Top Numeric Features by Outlier Count (IQR Rule)",
    "",
]

if top_outliers.empty:
    md_lines.append("- No numeric features available for outlier analysis.")
else:
    for _, row in top_outliers.iterrows():
        md_lines.append(
            f"- **{row['feature name']}**: "
            f"{row['outlier count (IQR rule for numeric)']} outliers"
        )

findings_md_path = REPORTS_DIR / "eda_findings.md"
findings_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
print(f"Saved markdown summary to: {findings_md_path.relative_to(ROOT)}")

# BASIC SUMMARIES
section("TARGET DISTRIBUTION")
counts = df[TARGET].value_counts()
print(counts)
print(f"\nClass balance ratio (max/min): {counts.max() / counts.min():.3f}")
print("(values near 1.0 = perfectly balanced — a smoking gun for synthetic data)")

section("NUMERIC FEATURE SUMMARY")
print(df[num_cols].describe().round(2))

section("CATEGORICAL CARDINALITY")
for c in cat_cols:
    print(f"  {c:25s} {df[c].nunique():4d} unique")

section("INTEREST TAGS")
parsed = parse_interest_tags(df[MULTI_LABEL_COL])
tag_counts = Counter(t for row in parsed for t in row)
print(f"Total unique tags: {len(tag_counts)}")
print(f"Top 10: {tag_counts.most_common(10)}")
print(f"Bottom 5:  {tag_counts.most_common()[-5:]}")
print("(near-uniform tag frequencies = another synthetic-data fingerprint)")

# MULTIVARIATE STRUCTURE (PCA)
section("MULTIVARIATE STRUCTURE")
cat_base_cols = [c for c in cat_cols if c != MULTI_LABEL_COL]
cat_onehot = pd.get_dummies(df[cat_base_cols], drop_first=False, dtype=float)

tags = parse_interest_tags(df[MULTI_LABEL_COL])
mlb = MultiLabelBinarizer()
tag_onehot = pd.DataFrame(
    mlb.fit_transform(tags),
    columns=[f"{MULTI_LABEL_COL}__{tag}" for tag in mlb.classes_],
    index=df.index,
)

scaler = StandardScaler()
num_scaled = pd.DataFrame(
    scaler.fit_transform(df[num_cols]),
    columns=num_cols,
    index=df.index,
)

X_pca = pd.concat([num_scaled, cat_onehot, tag_onehot], axis=1)

pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(X_pca)

plot_df = pd.DataFrame(
    {"PC1": pcs[:, 0], "PC2": pcs[:, 1], TARGET: df[TARGET].values},
    index=df.index,
)

explained = pca.explained_variance_ratio_
explained_total = explained.sum()
centroids = plot_df.groupby(TARGET)[["PC1", "PC2"]].mean().to_numpy()

if len(centroids) > 1:
    centroid_dist = np.linalg.norm(
        centroids[:, None, :] - centroids[None, :, :], axis=2
    )
    avg_centroid_dist = centroid_dist[np.triu_indices_from(centroid_dist, k=1)].mean()
else:
    avg_centroid_dist = 0.0

within_dispersion = (
    plot_df.groupby(TARGET)[["PC1", "PC2"]].std().mean(axis=1).mean()
)
separation_index = avg_centroid_dist / (within_dispersion + 1e-8)

if separation_index < 1.0:
    interpretation = "Classes heavily overlap in PC space; no clear cluster boundaries by outcome."
elif separation_index < 1.5:
    interpretation = "Only mild separation appears in PC space; cluster boundaries by outcome are weak."
else:
    interpretation = "Outcomes show noticeable separation in PC space with partial cluster structure."

print(
    f"Explained variance — PC1: {explained[0]:.3f}, PC2: {explained[1]:.3f}, "
    f"total: {explained_total:.3f}"
)
print(f"Separation index: {separation_index:.3f}")
print(f"Interpretation: {interpretation}")

# PLOTS
section("GENERATING PLOTS — saving to reports/figures/")

# PLOT 01 — Target distribution (horizontal bars, count + %)
counts_sorted = df[TARGET].value_counts().sort_values()
pct = counts_sorted / counts_sorted.sum() * 100

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.barh(counts_sorted.index, counts_sorted.values, color=COL_PRIMARY, edgecolor="white")
ax.set_title(f"match_outcome distribution (N = {counts_sorted.sum():,})")
ax.set_xlabel("Count")
ax.set_xlim(0, counts_sorted.max() * 1.15)

for bar, count, p in zip(bars, counts_sorted.values, pct.values):
    ax.text(
        bar.get_width() + counts_sorted.max() * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{count:,} ({p:.1f}%)",
        ha="left",
        va="center",
        fontsize=9,
    )

balance_ratio = counts_sorted.max() / counts_sorted.min()
ax.text(
    0.98,
    0.02,
    f"Class balance ratio: {balance_ratio:.3f}\n(1.000 = perfectly balanced)",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=8.5,
    style="italic",
    color="#666",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="#DDD"),
)
ax.grid(axis="x", alpha=0.3)
ax.set_axisbelow(True)
save_fig(fig, "01_target_distribution", FIGDIR)

# PLOT 02 — Numeric distributions, grouped by kind, with mean/median
groups = {
    "Continuous": [
        "app_usage_time_min",
        "age",
        "weight_kg",
        "height_cm",
        "bio_length",
        "likes_received",
    ],
    "Discrete / Count": [
        "mutual_matches",
        "message_sent_count",
        "profile_pics_count",
        "last_active_hour",
    ],
    "Bounded ratio": [
        "swipe_right_ratio",
        "emoji_usage_rate",
    ],
}

fig, axes = plt.subplots(4, 3, figsize=(11, 13))
axes_flat = axes.flatten()
plot_idx = 0

for group_label, cols in groups.items():
    for col in cols:
        if col not in df.columns:
            continue

        ax = axes_flat[plot_idx]
        mean = df[col].mean()
        median = df[col].median()
        std = df[col].std()

        ax.hist(df[col], bins=30, color=COL_PRIMARY, edgecolor="white", alpha=0.85)
        ax.axvline(mean, color=COL_ACCENT, linestyle="-", linewidth=1.8, label=f"mean={mean:.2f}")
        ax.axvline(median, color=COL_HIGHLIGHT, linestyle="--", linewidth=1.8, label=f"median={median:.2f}")
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)

        ax.set_title(f"{col}  [{group_label}]", fontsize=10)
        ax.set_ylabel("count", fontsize=8)
        ax.tick_params(labelsize=8)

        ax.text(
            0.02,
            0.97,
            f"n={len(df):,}\nσ={std:.2f}\nmin={df[col].min():.1f}\nmax={df[col].max():.1f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#DDD", alpha=0.9),
        )
        plot_idx += 1

for i in range(plot_idx, len(axes_flat)):
    axes_flat[i].set_visible(False)

fig.suptitle("Numeric feature distributions with exact central tendencies", fontsize=12, fontweight="bold", y=1.00)
fig.tight_layout()
save_fig(fig, "02_numeric_distributions", FIGDIR)

# PLOT 03 — Correlation heatmap (lower triangle, |r| ≥ 0.05 shown)
corr = df[num_cols].corr()
mask_upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
mask_noise = np.abs(corr.values) < 0.05
mask = mask_upper | mask_noise
np.fill_diagonal(mask, False)

fig, ax = plt.subplots(figsize=(10, 8.5))
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"shrink": 0.6, "label": "Pearson r"},
    annot_kws={"fontsize": 9, "weight": "bold"},
    ax=ax,
)
ax.set_title("Numeric feature correlation matrix (|r| ≥ 0.05 shown)", fontsize=12, pad=15)
plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
plt.setp(ax.get_yticklabels(), rotation=0)

cols_list = list(corr.columns)
notable = [("height_cm", "weight_kg"), ("likes_received", "mutual_matches")]
for a, b in notable:
    if a in cols_list and b in cols_list:
        i = cols_list.index(a)
        j = cols_list.index(b)
        if i > j:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor=COL_HIGHLIGHT, lw=3))
        else:
            ax.add_patch(plt.Rectangle((i, j), 1, 1, fill=False, edgecolor=COL_HIGHLIGHT, lw=3))

ax.text(
    0.5,
    -0.30,
    "Orange box = meaningful correlation. Empty cells = |r| < 0.05 (negligible). "
    "All non-orange relationships are near-zero, consistent with synthetic data.",
    transform=ax.transAxes,
    ha="center",
    fontsize=8.5,
    style="italic",
    color="#555",
)
save_fig(fig, "03_correlation_heatmap", FIGDIR)

# PLOT 04 — Categorical balance overview + the imbalanced feature
imbalance = {}
for col in cat_cols:
    vc = df[col].value_counts()
    imbalance[col] = vc.max() / vc.min()

imb_series = pd.Series(imbalance).sort_values()
most_imbalanced = imb_series.idxmax()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.3]})

bars = ax1.barh(
    imb_series.index,
    imb_series.values,
    color=[COL_ACCENT if v > 2 else COL_MUTED for v in imb_series.values],
    edgecolor="white",
)
ax1.set_title("Class imbalance per categorical feature\n(ratio = max_count / min_count)")
ax1.set_xlabel("Imbalance ratio (1.0 = perfectly balanced)")
ax1.axvline(1.0, color="#999", linestyle=":", linewidth=1)

for bar, val in zip(bars, imb_series.values):
    ax1.text(
        val + imb_series.max() * 0.015,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.2f}",
        ha="left",
        va="center",
        fontsize=9,
    )

ax1.set_xlim(0, imb_series.max() * 1.18)
ax1.grid(axis="x", alpha=0.3)
ax1.set_axisbelow(True)

vc = df[most_imbalanced].value_counts()
pct_vc = vc / vc.sum() * 100
bars = ax2.barh(vc.index, vc.values, color=COL_PRIMARY, edgecolor="white")
ax2.set_title(f"'{most_imbalanced}' — the most imbalanced feature")
ax2.set_xlabel("Count")
ax2.set_xlim(0, vc.max() * 1.18)
ax2.grid(axis="x", alpha=0.3)
ax2.set_axisbelow(True)

for bar, count, p in zip(bars, vc.values, pct_vc.values):
    ax2.text(
        bar.get_width() + vc.max() * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{count:,} ({p:.1f}%)",
        ha="left",
        va="center",
        fontsize=9,
    )

fig.suptitle("Categorical features: most are balanced, some are not", fontsize=13, fontweight="bold", y=1.00)
fig.tight_layout()
save_fig(fig, "04_categorical_overview", FIGDIR)

# PLOT 05 — Interest tags: top 15 with exact counts + %
top_n = 15
top_tags = dict(tag_counts.most_common(top_n))
other_tags = dict(tag_counts.most_common()[top_n:])
other_min = min(other_tags.values())
other_max = max(other_tags.values())
other_count = len(other_tags)

fig, ax = plt.subplots(figsize=(10, 7))
tags_list = list(reversed(top_tags.keys()))
vals = list(reversed(top_tags.values()))
pct = [v / sum(tag_counts.values()) * 100 for v in vals]

bars = ax.barh(tags_list, vals, color=COL_PRIMARY, edgecolor="white")
ax.set_title(f"Top {top_n} interest tags (out of {len(tag_counts)} unique tags)")
ax.set_xlabel("Occurrences across all users")
ax.set_xlim(0, max(vals) * 1.15)

for bar, val, p in zip(bars, vals, pct):
    ax.text(
        bar.get_width() + max(vals) * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{val:,} ({p:.2f}%)",
        ha="left",
        va="center",
        fontsize=9,
    )

ax.text(
    0.98,
    0.02,
    f"Remaining {other_count} tags: range {other_min:,}–{other_max:,} occurrences each.\n"
    f"All {len(tag_counts)} tags ≈ uniform — synthetic-data fingerprint.",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=9,
    style="italic",
    color="#555",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="#DDD"),
)
ax.grid(axis="x", alpha=0.3)
ax.set_axisbelow(True)
save_fig(fig, "05_interest_tags_top15", FIGDIR)

# PLOT 06 — Means by outcome (the visual no-signal proof)
feats = ["likes_received", "mutual_matches", "message_sent_count", "bio_length"]

outcome_order_full = [
    "Blocked",
    "Catfished",
    "Ghosted",
    "Chat Ignored",
    "No Action",
    "One-sided Like",
    "Instant Match",
    "Mutual Match",
    "Date Happened",
    "Relationship Formed",
]
outcome_order = [o for o in outcome_order_full if o in df[TARGET].unique()]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes_flat = axes.flatten()

for ax, col in zip(axes_flat, feats):
    means = df.groupby(TARGET)[col].mean().reindex(outcome_order)
    stds = df.groupby(TARGET)[col].std().reindex(outcome_order)
    overall_mean = df[col].mean()

    bars = ax.bar(
        range(len(means)),
        means.values,
        yerr=stds.values,
        capsize=3,
        color=COL_PRIMARY,
        edgecolor="white",
        error_kw={"alpha": 0.5, "linewidth": 1},
    )
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(means.index, rotation=40, ha="right", fontsize=8.5)
    ax.set_ylabel(col)
    ax.set_title(f"Mean {col} per match_outcome (± std)")

    for bar, m in zip(bars, means.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + stds.values.max() * 0.05,
            f"{m:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.axhline(overall_mean, color=COL_ACCENT, linestyle="--", linewidth=1.3, label=f"overall mean = {overall_mean:.1f}")
    ax.legend(loc="upper right", fontsize=8)

    spread = means.max() - means.min()
    ax.text(
        0.02,
        0.97,
        f"Range of means: {spread:.2f}\nStd of means: {means.std():.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#DDD", alpha=0.9),
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

fig.suptitle("If features had signal, means would differ across outcomes. They don't.", fontsize=13, fontweight="bold", y=1.00)
fig.tight_layout()
save_fig(fig, "06_means_by_outcome", FIGDIR)

# PLOT 07 — PCA scatter (subsampled for readability), colored by outcome
plot_sample = plot_df.sample(n=min(5000, len(plot_df)), random_state=42)

fig, ax = plt.subplots(figsize=(11, 7))
sns.scatterplot(
    data=plot_sample,
    x="PC1",
    y="PC2",
    hue=TARGET,
    palette="tab10",
    alpha=0.55,
    s=28,
    linewidth=0,
    ax=ax,
)
ax.set_title(
    f"Multivariate structure via PCA (PC1 vs PC2)\n"
    f"explained variance: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}, total={explained_total:.1%}",
    pad=12,
)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(title=TARGET, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=8)

ax.text(
    0.02,
    0.02,
    f"Separation index: {separation_index:.3f}\n{interpretation}\n\n"
    f"(Showing {len(plot_sample):,} of {len(plot_df):,} rows for visibility.)",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=8,
    family="monospace",
    color="#555",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF4E6", edgecolor=COL_HIGHLIGHT),
)
fig.tight_layout()
save_fig(fig, "07_pca_structure", FIGDIR)

# PLOT 08–09 — Rule-based engagement segments (manual segmentation)
section("RULE-BASED ENGAGEMENT SEGMENTS (4 disjoint segments)")

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

emoji_bins = [-0.01, 0.33, 0.66, 1.00]
emoji_labels = ["Low emoji", "Medium emoji", "High emoji"]
seg_df["emoji_profile"] = pd.cut(
    seg_df["emoji_usage_rate"],
    bins=emoji_bins,
    labels=emoji_labels,
    include_lowest=True,
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

print(segment_summary.round(2).to_string())
print("\nEmoji usage profile (% within segment):")
print(emoji_profile_pct.round(1).to_string())

# Plot 08: grouped medians + sample size
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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [2.2, 1]})
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

for p in ax1.patches:
    h = p.get_height()
    if not np.isnan(h) and h > 0:
        ax1.annotate(
            f"{h:.1f}",
            (p.get_x() + p.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 2),
            textcoords="offset points",
        )

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
save_fig(fig, "08_segment_grouped_bars", FIGDIR)

# Plot 09: message volume + emoji profile
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.5, 1]})
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

for container in ax2.containers:
    labels = [f"{int(round(v))}" if v > 5 else "" for v in container.datavalues]
    ax2.bar_label(container, labels=labels, label_type="center", fontsize=7)

fig.tight_layout()
save_fig(fig, "09_segment_boxplot_emoji_profile", FIGDIR)

# Insight summary
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
print(f"Done. 9 plots saved to: {FIGDIR.relative_to(ROOT)}")
print("=" * 70)
print("\nNext: run scripts/02_signal_test.py to quantify the independence.")