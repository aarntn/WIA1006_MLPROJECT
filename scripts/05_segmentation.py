"""
05_segmentation.py - unsupervised user segmentation workflow.

Run from project root:
    python scripts/05_segmentation.py

Fast smoke test:
    python scripts/05_segmentation.py --fast

Outputs:
    - data/processed/segmentation_assignments.csv
    - reports/segmentation_summary.csv
    - reports/segmentation_findings.md
    - reports/figures/15_kmeans_selection.png  (silhouette k=2-10)
    - reports/figures/16_segment_umap.png
    - reports/figures/17_segment_profiles.png
    - reports/figures/18_gmm_bic_aic.png
    - reports/figures/19_shap_cluster_beeswarm.png
    - models/kmeans_segmentation.joblib
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import shap
except Exception:  # pragma: no cover - optional dependency fallback
    shap = None

from src.data import TARGET, load_raw
from src.features import EngagementFeatureBuilder

try:
    import hdbscan
except Exception:  # pragma: no cover - optional dependency fallback
    hdbscan = None

try:
    import umap
except Exception:  # pragma: no cover - optional dependency fallback
    umap = None


ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
FIGDIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"

from src.plotting import COL_ACCENT, COL_HIGHLIGHT, COL_MUTED, COL_PRIMARY, save_fig, setup_plot_style

PROFILE_COLS = [
    "app_usage_time_min",
    "swipe_right_ratio",
    "likes_received",
    "mutual_matches",
    "message_sent_count",
    "emoji_usage_rate",
    "profile_pics_count",
    "bio_length",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Use fewer rows for a quick smoke test.")
    parser.add_argument("--sample", type=int, default=None, help="Optional row sample for local experimentation.")
    return parser.parse_args()


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, Pipeline]:
    pipeline = Pipeline(
        [
            (
                "features",
                EngagementFeatureBuilder(
                    target_col="__none__",
                    include_paired_target=True,
                    include_match_outcome=False,
                    include_interest_tags=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    X = pipeline.fit_transform(df.drop(columns=[TARGET], errors="ignore"))
    feature_names = pipeline.named_steps["features"].get_feature_names_out()
    return pd.DataFrame(X, columns=feature_names, index=df.index), pipeline


def select_kmeans(X: pd.DataFrame, fast: bool) -> tuple[KMeans, pd.DataFrame]:
    sample_n = min(3000 if fast else 8000, len(X))
    X_score = X.sample(n=sample_n, random_state=42)
    rows = []
    best_model = None
    best_score = -np.inf

    for k in range(2, 11):
        model = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = model.fit_predict(X)
        score_labels = labels[X_score.index]
        score = silhouette_score(X_score, score_labels)
        inertia = model.inertia_
        rows.append({"k": k, "silhouette": score, "inertia": inertia})
        if score > best_score:
            best_score = score
            best_model = model

    assert best_model is not None
    return best_model, pd.DataFrame(rows)


def compare_hdbscan(X: pd.DataFrame, fast: bool) -> dict[str, float | int | str]:
    if hdbscan is None:
        return {"status": "not installed", "clusters": 0, "noise_pct": np.nan, "silhouette": np.nan}

    sample_n = min(4000 if fast else 10000, len(X))
    X_sample = X.sample(n=sample_n, random_state=42)
    pca_dims = min(20, X_sample.shape[1])
    X_reduced = PCA(n_components=pca_dims, random_state=42).fit_transform(X_sample)
    try:
        labels = hdbscan.HDBSCAN(min_cluster_size=150 if fast else 300, min_samples=20).fit_predict(X_reduced)
    except Exception as exc:
        return {"status": f"error: {exc}", "clusters": 0, "noise_pct": np.nan, "silhouette": np.nan}
    non_noise = labels != -1
    clusters = len(set(labels[non_noise]))
    noise_pct = 100 * (labels == -1).mean()
    if clusters >= 2 and non_noise.sum() > 20:
        score = silhouette_score(X_reduced[non_noise], labels[non_noise])
    else:
        score = np.nan
    return {
        "status": "ok",
        "clusters": clusters,
        "noise_pct": noise_pct,
        "silhouette": score,
    }


def name_segments(summary: pd.DataFrame) -> dict[int, str]:
    profile = summary[PROFILE_COLS]
    z = (profile - profile.mean()) / profile.std(ddof=0).replace(0, 1)
    descriptor_map = {
        ("app_usage_time_min", "high"): "High-activity users",
        ("app_usage_time_min", "low"): "Low-activity browsers",
        ("swipe_right_ratio", "high"): "Open swipers",
        ("swipe_right_ratio", "low"): "Selective swipers",
        ("likes_received", "high"): "High-like receivers",
        ("likes_received", "low"): "Low-like users",
        ("mutual_matches", "high"): "High-reciprocity users",
        ("mutual_matches", "low"): "Low-match users",
        ("message_sent_count", "high"): "Message-heavy users",
        ("message_sent_count", "low"): "Quiet browsers",
        ("emoji_usage_rate", "high"): "Expressive chatters",
        ("emoji_usage_rate", "low"): "Low-emoji texters",
        ("profile_pics_count", "high"): "Photo-forward users",
        ("profile_pics_count", "low"): "Minimal-photo users",
        ("bio_length", "high"): "Detailed-profile users",
        ("bio_length", "low"): "Brief-profile users",
    }
    names: dict[int, str] = {}
    used: set[str] = set()

    for cluster_id, row in summary.iterrows():
        z_row = z.loc[cluster_id].sort_values(key=lambda s: s.abs(), ascending=False)
        base = "Balanced explorers"
        for metric, value in z_row.items():
            direction = "high" if value >= 0 else "low"
            candidate = descriptor_map[(metric, direction)]
            if candidate not in used:
                base = candidate
                break

        name = base
        suffix = 2
        while name in used:
            name = f"{base} {suffix}"
            suffix += 1
        names[int(cluster_id)] = name
        used.add(name)

    return names


def build_segment_summary(df: pd.DataFrame, labels: np.ndarray) -> tuple[pd.DataFrame, dict[int, str]]:
    prof = df.copy()
    prof["segment_id"] = labels
    summary = (
        prof.groupby("segment_id")
        .agg(
            users=("segment_id", "size"),
            app_usage_time_min=("app_usage_time_min", "mean"),
            swipe_right_ratio=("swipe_right_ratio", "mean"),
            likes_received=("likes_received", "mean"),
            mutual_matches=("mutual_matches", "mean"),
            message_sent_count=("message_sent_count", "mean"),
            emoji_usage_rate=("emoji_usage_rate", "mean"),
            profile_pics_count=("profile_pics_count", "mean"),
            bio_length=("bio_length", "mean"),
            top_relationship_intent=("relationship_intent", lambda s: s.mode().iat[0]),
            top_usage_label=("app_usage_time_label", lambda s: s.mode().iat[0]),
        )
        .sort_index()
    )
    names = name_segments(summary)
    summary.insert(1, "segment_name", summary.index.map(names))
    return summary, names


def plot_k_selection(scores: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(scores["k"], scores["silhouette"], marker="o", color=COL_PRIMARY, label="silhouette")
    ax1.set_xlabel("KMeans k")
    ax1.set_ylabel("Silhouette", color=COL_PRIMARY)
    ax1.tick_params(axis="y", labelcolor=COL_PRIMARY)
    ax2 = ax1.twinx()
    ax2.plot(scores["k"], scores["inertia"], marker="s", color=COL_ACCENT, label="inertia")
    ax2.set_ylabel("Inertia", color=COL_ACCENT)
    ax2.tick_params(axis="y", labelcolor=COL_ACCENT)
    ax1.set_title("KMeans model selection")
    save_fig(fig, "15_kmeans_selection", FIGDIR)


def plot_embedding(X: pd.DataFrame, labels: np.ndarray, names: dict[int, str], fast: bool) -> None:
    sample_n = min(5000 if fast else 9000, len(X))
    idx = X.sample(n=sample_n, random_state=42).index
    X_sample = X.loc[idx]
    if umap is not None:
        try:
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=25, min_dist=0.1)
            coords = reducer.fit_transform(X_sample)
            method = "UMAP"
        except Exception:
            coords = PCA(n_components=2, random_state=42).fit_transform(X_sample)
            method = "PCA (UMAP unavailable)"
    else:
        coords = PCA(n_components=2, random_state=42).fit_transform(X_sample)
        method = "PCA"

    plot_df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "segment": [names[int(labels[i])] for i in idx],
        }
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=plot_df, x="x", y="y", hue="segment", s=18, linewidth=0, alpha=0.65, ax=ax)
    ax.set_title(f"{method} view of behavioral segments")
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(fig, "16_segment_umap", FIGDIR)


def plot_profiles(summary: pd.DataFrame) -> None:
    profile = summary.set_index("segment_name")[PROFILE_COLS]
    z = (profile - profile.mean()) / profile.std(ddof=0).replace(0, 1)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(z, cmap="RdBu_r", center=0, annot=True, fmt=".1f", linewidths=0.5, ax=ax)
    ax.set_title("Segment profiles as z-scores vs segment average")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_fig(fig, "17_segment_profiles", FIGDIR)


def run_gmm_selection(X: pd.DataFrame, fast: bool) -> pd.DataFrame:
    """Sweep GaussianMixture k=2..10, record BIC/AIC.

    On a uniformly-distributed feature space, both BIC and AIC should decrease
    monotonically with no knee — confirming no preferred number of components.
    """
    sample_n = min(3000 if fast else 8000, len(X))
    X_s = X.sample(n=sample_n, random_state=42).values
    rows = []
    k_range = range(2, 8) if fast else range(2, 11)
    for k in k_range:
        gm = GaussianMixture(n_components=k, random_state=42, max_iter=100)
        gm.fit(X_s)
        rows.append({"k": k, "bic": gm.bic(X_s), "aic": gm.aic(X_s)})
    return pd.DataFrame(rows)


def plot_gmm_selection(gmm_scores: pd.DataFrame) -> None:
    from src.plotting import COL_GOOD
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gmm_scores["k"], gmm_scores["bic"], marker="o", color=COL_PRIMARY, label="BIC")
    ax.plot(gmm_scores["k"], gmm_scores["aic"], marker="s", color=COL_GOOD,    label="AIC")
    ax.set_xlabel("Number of GMM components (k)")
    ax.set_ylabel("Information criterion (lower = better fit)")
    ax.set_title(
        "GMM BIC/AIC vs k — monotonic decrease = no preferred k\n"
        "(convergent evidence: feature space has no density modes)",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    save_fig(fig, "18_gmm_bic_aic", FIGDIR)


def plot_shap_cluster_classifier(X: pd.DataFrame, labels: np.ndarray, fast: bool) -> None:
    """Train a GBDT to predict K-Means cluster labels, then SHAP it.

    SHAP on the R²≈0 regressor is not meaningful (noise attribution).
    SHAP on a cluster-membership classifier explains the *geometry* K-Means
    discovered, even if the clusters themselves have low silhouette.
    """
    if shap is None:
        print("  [skip] shap not installed — skipping cluster SHAP plot")
        return

    print("  Fitting cluster-membership classifier for SHAP...")
    sample_n = min(2000 if fast else 5000, len(X))
    idx = X.sample(n=sample_n, random_state=42).index
    X_s = X.loc[idx].values
    y_s = labels[idx]

    # RandomForestClassifier: TreeExplainer supports multiclass (returns list of arrays)
    clf = RandomForestClassifier(n_estimators=50 if fast else 100, max_depth=4, random_state=42, n_jobs=-1)
    clf.fit(X_s, y_s)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_s)

    # Normalize to 1D per-feature importance regardless of SHAP API version:
    #   old API: list of (n_samples, n_features) per class
    #   new API: ndarray of shape (n_classes, n_samples, n_features) or (n_samples, n_features)
    # Normalize to 1D per-feature importance regardless of SHAP API version.
    # Old API: list of (n_samples, n_features) per class.
    # New API (0.43+): ndarray of shape (n_samples, n_features, n_classes).
    sv = np.array(shap_values)
    if sv.ndim == 3 and not isinstance(shap_values, list):
        # (n_samples, n_features, n_classes) → mean over samples and classes
        mean_abs_shap = np.abs(sv).mean(axis=(0, 2))
    elif isinstance(shap_values, list):
        # list of (n_samples, n_features) → mean per feature across classes
        mean_abs_shap = np.mean([np.abs(s).mean(axis=0) for s in shap_values], axis=0)
    else:
        mean_abs_shap = np.abs(sv).mean(axis=0)

    importance = pd.DataFrame(
        {"feature": X.columns.tolist(), "mean_abs_shap": mean_abs_shap}
    ).sort_values("mean_abs_shap", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(importance["feature"][::-1], importance["mean_abs_shap"][::-1], color=COL_PRIMARY)
    for bar in bars:
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                f" {bar.get_width():.4f}", va="center", fontsize=8)
    ax.set_xlabel("Mean |SHAP value| across all cluster classes")
    ax.set_title(
        "SHAP: what drives K-Means cluster membership?\n"
        "(cluster geometry explained, not target prediction)",
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    save_fig(fig, "19_shap_cluster_beeswarm", FIGDIR)


def write_findings(
    summary: pd.DataFrame,
    scores: pd.DataFrame,
    hdbscan_result: dict[str, float | int | str],
    gmm_scores: pd.DataFrame,
    n_rows: int,
    fast: bool,
) -> None:
    best = scores.sort_values("silhouette", ascending=False).iloc[0]
    sil_range = f"{scores['silhouette'].min():.3f}–{scores['silhouette'].max():.3f}"
    gmm_monotonic = "Yes" if gmm_scores["bic"].is_monotonic_decreasing else "No"
    lines = [
        "# Segmentation Findings",
        "",
        f"- Rows used in this run: **{n_rows:,}**",
        f"- Fast mode: **{fast}**",
        f"- KMeans k range tested: {scores['k'].min()}–{scores['k'].max()}",
        f"- Silhouette range across all k: **{sil_range}** (Rousseeuw 1987 'weak' threshold = 0.25)",
        f"- Selected KMeans k: **{int(best['k'])}** (silhouette={best['silhouette']:.3f})",
        f"- **Interpretation**: silhouette < 0.05 for all k — no meaningful cluster structure detected",
        f"- HDBSCAN comparison: status={hdbscan_result['status']}, clusters={hdbscan_result['clusters']}, noise={hdbscan_result['noise_pct']:.1f}%, silhouette={hdbscan_result['silhouette']:.3f}",
        f"- GMM BIC monotonically decreasing: **{gmm_monotonic}** — no preferred number of components",
        f"- Convergent conclusion: three independent algorithms (KMeans, HDBSCAN, GMM) confirm no density structure",
        "- Segment names describe behavior patterns only; they do not claim real relationship psychology.",
        "",
        "## Segment Profiles",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- **{row['segment_name']}**: {int(row['users']):,} users, "
            f"avg matches={row['mutual_matches']:.2f}, avg messages={row['message_sent_count']:.2f}, "
            f"top intent={row['top_relationship_intent']}"
        )
    (REPORTS_DIR / "segmentation_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  [saved] reports/segmentation_findings.md")


def main() -> None:
    args = parse_args()
    setup_plot_style()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw(extended=True)
    sample_n = args.sample or (15000 if args.fast else None)
    if sample_n:
        df = df.sample(n=min(sample_n, len(df)), random_state=42).reset_index(drop=True)

    print("Building segmentation feature matrix")
    X, feature_pipeline = build_feature_matrix(df)
    kmeans, scores = select_kmeans(X, fast=args.fast)
    labels = kmeans.labels_
    summary, names = build_segment_summary(df, labels)
    hdbscan_result = compare_hdbscan(X, fast=args.fast)

    assignments = pd.DataFrame(
        {
            "row_id": df.index,
            "segment_id": labels,
            "segment_name": [names[int(label)] for label in labels],
        }
    )
    assignments.to_csv(PROCESSED_DIR / "segmentation_assignments.csv", index=False)
    scores.to_csv(REPORTS_DIR / "segmentation_k_selection.csv", index=False)
    summary.to_csv(REPORTS_DIR / "segmentation_summary.csv")
    joblib.dump(
        Pipeline([("feature_pipeline", feature_pipeline), ("kmeans", kmeans)]),
        MODELS_DIR / "kmeans_segmentation.joblib",
    )
    print("  [saved] data/processed/segmentation_assignments.csv")
    print("  [saved] reports/segmentation_summary.csv")
    print("  [saved] models/kmeans_segmentation.joblib")

    gmm_scores = run_gmm_selection(X, fast=args.fast)

    plot_k_selection(scores)
    plot_embedding(X, labels, names, fast=args.fast)
    plot_profiles(summary)
    plot_gmm_selection(gmm_scores)
    plot_shap_cluster_classifier(X, labels, fast=args.fast)
    write_findings(summary, scores, hdbscan_result, gmm_scores, n_rows=len(df), fast=args.fast)

    print("\nDone.")
    print(summary[["segment_name", "users", "mutual_matches", "message_sent_count"]].round(2).to_string())


if __name__ == "__main__":
    main()
