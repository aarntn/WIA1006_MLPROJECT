"""
01_eda.py — Basic exploratory data analysis.

Run from project root:
    python scripts/01_eda.py

Outputs:
    - printed summaries to stdout
    - distribution plots to reports/figures/
    - correlation matrix to reports/figures/
    - data quality audit to reports/
"""
import sys
from pathlib import Path
import importlib.util
from collections import Counter

# Make src/ importable when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from src.data import load_raw, get_feature_cols, MULTI_LABEL_COL, TARGET
from src.preprocessing import parse_interest_tags

# ---- setup ----
ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"
FIGDIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


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

section("DATA QUALITY AUDIT")
audit_df = build_data_quality_audit(df)
print(audit_df.to_string(index=False))

audit_csv_path = REPORTS_DIR / "eda_quality_audit.csv"
audit_df.to_csv(audit_csv_path, index=False)
print(f"\nSaved audit table to: {audit_csv_path.relative_to(ROOT)}")

findings_md_path = REPORTS_DIR / "eda_findings.md"
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
            f"- **{row['feature name']}**: {row['outlier count (IQR rule for numeric)']} outliers"
        )

findings_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
print(f"Saved markdown summary to: {findings_md_path.relative_to(ROOT)}")

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

# ---- multivariate structure ----
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

X = pd.concat([num_scaled, cat_onehot, tag_onehot], axis=1)

pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(X)

plot_df = pd.DataFrame(
    {"PC1": pcs[:, 0], "PC2": pcs[:, 1], TARGET: df[TARGET].values},
    index=df.index,
)

explained = pca.explained_variance_ratio_
explained_total = explained.sum()
centroids = plot_df.groupby(TARGET)[["PC1", "PC2"]].mean().to_numpy()

if len(centroids) > 1:
    centroid_dist = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    avg_centroid_dist = centroid_dist[np.triu_indices_from(centroid_dist, k=1)].mean()
else:
    avg_centroid_dist = 0.0

within_dispersion = plot_df.groupby(TARGET)[["PC1", "PC2"]].std().mean(axis=1).mean()
separation_index = avg_centroid_dist / (within_dispersion + 1e-8)

if separation_index < 1.0:
    interpretation = "Classes heavily overlap in PC space; no clear cluster boundaries by outcome."
elif separation_index < 1.5:
    interpretation = "Only mild separation appears in PC space; cluster boundaries by outcome are weak."
else:
    interpretation = "Outcomes show noticeable separation in PC space with partial cluster structure."

print(f"Explained variance — PC1: {explained[0]:.3f}, PC2: {explained[1]:.3f}, total: {explained_total:.3f}")
print(f"Interpretation: {interpretation}")

# ---- plots ----
if importlib.util.find_spec("matplotlib") and importlib.util.find_spec("seaborn"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 110

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
    fig, axes = plt.subplots(
        nrows=(n + 2) // 3,
        ncols=3,
        figsize=(15, 3 * ((n + 2) // 3)),
    )
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
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        cbar_kws={"shrink": 0.7},
    )
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

    # 6. Target x numeric feature
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

    # 7. PCA multivariate structure
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue=TARGET,
        palette="tab10",
        alpha=0.55,
        s=28,
        linewidth=0,
        ax=ax,
    )
    ax.set_title("Multivariate structure via PCA (PC1 vs PC2)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title=TARGET, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig(FIGDIR / "07_multivariate_structure_pca.png")
    plt.close()
    print("  [saved] 07_multivariate_structure_pca.png")
else:
    section("GENERATING PLOTS — skipped")
    print("matplotlib/seaborn not available in this environment; skipping figure generation.")

print("\nDone. Next: run scripts/02_signal_test.py to quantify the independence.")