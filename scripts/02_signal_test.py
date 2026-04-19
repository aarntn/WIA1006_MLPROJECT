"""
02_signal_test.py — Quantify whether features predict match_outcome.

Run from project root:
    python scripts/02_signal_test.py

Outputs:
    - printed p-values and model accuracies
    - reports/signal_findings.md         (markdown summary for report)
    - reports/figures/10_signal_summary.png  (polished p-value + accuracy chart)
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import chi2_contingency, f_oneway
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.data import TARGET, get_feature_cols, load_raw
from src.preprocessing import label_encode, outcome_3class

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
COL_GOOD = "#55A868"


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

# Build markdown content as we go
lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    lines.append(msg)

# LOAD
df = load_raw(extended=True)
num_cols, cat_cols = get_feature_cols(df)

log("# Signal Findings — `match_outcome` prediction")
log("")
log(f"Dataset: {df.shape[0]} rows × {df.shape[1]} cols")
log(f"Target: `{TARGET}` ({df[TARGET].nunique()} classes, balanced)")
log(f"Random baseline: {1 / df[TARGET].nunique():.3f}")
log("")

# TEST 1 — Chi-square (categorical features vs target)
log("## 1. Chi-square tests (categorical features vs target)")
log("")
log("| Feature | chi² | dof | p-value | Significant? |")
log("|---|---|---|---|---|")
chi_results = []
for col in cat_cols:
    contingency = pd.crosstab(df[col], df[TARGET])
    chi2, p, dof, _ = chi2_contingency(contingency)
    sig = "✅" if p < 0.05 else "❌"
    log(f"| `{col}` | {chi2:.2f} | {dof} | {p:.4f} | {sig} |")
    chi_results.append((col, chi2, p))
log("")

# TEST 2 — ANOVA (numeric features across outcome groups)
log("## 2. ANOVA (numeric features across outcome groups)")
log("")
log("| Feature | F-stat | p-value | Significant? |")
log("|---|---|---|---|")
anova_results = []
classes = df[TARGET].unique()
for col in num_cols:
    groups = [df[df[TARGET] == c][col].values for c in classes]
    f, p = f_oneway(*groups)
    sig = "✅" if p < 0.05 else "❌"
    log(f"| `{col}` | {f:.3f} | {p:.4f} | {sig} |")
    anova_results.append((col, f, p))
log("")

# TEST 3 — Model sanity check (10-class)
log("## 3. Model sanity check — 10-class classification")
log("")
df_enc, _ = label_encode(df, cat_cols + [TARGET])
feature_cols = [c for c in df_enc.columns if c not in (TARGET, "interest_tags")]
X = df_enc[feature_cols]
y = df_enc[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

log(f"Random baseline: {1 / df[TARGET].nunique():.3f}")
log("")
log("| Model | Train acc | Test acc |")
log("|---|---|---|")

clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest (100)": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    "Gradient Boosting (50)": GradientBoostingClassifier(n_estimators=50, random_state=42),
}
clf_results = {}
for name, m in clf_models.items():
    m.fit(X_train, y_train)
    tr = m.score(X_train, y_train)
    te = m.score(X_test, y_test)
    clf_results[name] = {"train": tr, "test": te}
    log(f"| {name} | {tr:.3f} | {te:.3f} |")
log("")

# TEST 4 — 3-class reformulation
log("## 4. 3-class reformulation (Negative / Neutral / Positive)")
log("")
y3 = LabelEncoder().fit_transform(outcome_3class(df[TARGET]))
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(
    X, y3, test_size=0.2, random_state=42, stratify=y3
)
maj = np.bincount(y3).max() / len(y3)
log(f"Majority-class baseline: {maj:.3f}")
log("")
log("| Model | Train acc | Test acc |")
log("|---|---|---|")
clf_results_3 = {}
for name, m in clf_models.items():
    m_new = type(m)(**m.get_params())
    m_new.fit(X_train_3, y_train_3)
    tr = m_new.score(X_train_3, y_train_3)
    te = m_new.score(X_test_3, y_test_3)
    clf_results_3[name] = {"train": tr, "test": te}
    log(f"| {name} | {tr:.3f} | {te:.3f} |")
log("")

# Summary
n_sig_cat = sum(1 for _, _, p in chi_results if p < 0.05)
n_sig_num = sum(1 for _, _, p in anova_results if p < 0.05)
log("## Summary")
log("")
log(f"- **{n_sig_cat}/{len(chi_results)}** categorical features show significant association with target (p < 0.05)")
log(f"- **{n_sig_num}/{len(anova_results)}** numeric features show significant differences across outcomes")
log("- All models converge to ~random baseline")
log("- Random Forest train/test gap (train≈1.0 / test≈0.1) is the classic no-signal fingerprint")
log("")
log("**Conclusion:** No practically useful predictive signal for `match_outcome` was found.")
log("Any isolated univariate significance is weak and does not translate to generalizable predictive performance.")
log("Model-level evidence (chance-level test performance) is the final criterion for this conclusion.")
log("The dataset is synthetic and labels were assigned independently of features.")
log("")
log("Next step: see `scripts/03_alternative_targets.py` for targets where signal *does* exist.")

# ---- Save markdown ----
(REPORTS_DIR / "signal_findings.md").write_text("\n".join(lines), encoding="utf-8")
print("\n[saved] reports/signal_findings.md")

# PLOT 10 — Signal summary (polished, two-panel)
print("\nGenerating polished plot...")

all_results = (
    [(col, p, "categorical") for col, _, p in chi_results]
    + [(col, p, "numeric") for col, _, p in anova_results]
)
all_results.sort(key=lambda r: r[1])

fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.35)

# LEFT: feature independence tests
ax1 = fig.add_subplot(gs[0, 0])

feats = [r[0] for r in all_results]
pvals = [r[1] for r in all_results]
kinds = [r[2] for r in all_results]
neg_log_p = [-np.log10(p) if p > 0 else 10 for p in pvals]

colors = [COL_ACCENT if p < 0.05 else COL_MUTED for p in pvals]

feats_r = list(reversed(feats))
pvals_r = list(reversed(pvals))
colors_r = list(reversed(colors))
neg_log_p_r = list(reversed(neg_log_p))
kinds_r = list(reversed(kinds))

bars = ax1.barh(range(len(feats_r)), neg_log_p_r, color=colors_r, edgecolor="white")
ax1.set_yticks(range(len(feats_r)))
ax1.set_yticklabels(feats_r, fontsize=9)

threshold = -np.log10(0.05)
ax1.axvline(threshold, color="black", linestyle="--", linewidth=1.3, alpha=0.7)
xmax_data = max(max(neg_log_p_r), threshold * 1.3)
ax1.axvspan(threshold, xmax_data * 1.2, alpha=0.08, color=COL_ACCENT)

for bar, p, nlp, k in zip(bars, pvals_r, neg_log_p_r, kinds_r):
    plabel = f"p = {p:.1e}" if p < 0.001 else f"p = {p:.3f}"
    kmark = "[c]" if k == "categorical" else "[n]"
    ax1.text(
        nlp + xmax_data * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{plabel}  {kmark}",
        ha="left",
        va="center",
        fontsize=8.5,
        fontweight="bold" if p < 0.05 else "normal",
    )

ax1.set_xlim(0, xmax_data * 1.40)
ax1.set_xlabel("−log₁₀(p-value)    →    higher = more significant", fontsize=10)
ax1.set_title(
    "Feature → match_outcome independence tests\nsorted by p-value (most significant at top)",
    pad=12,
)

legend_elements = [
    Patch(facecolor=COL_ACCENT, label="p < 0.05  (significant)"),
    Patch(facecolor=COL_MUTED, label="p ≥ 0.05  (not significant)"),
]
ax1.legend(handles=legend_elements, loc="center right", framealpha=0.95)

ax1.text(
    0.98,
    0.02,
    f"{n_sig_cat + n_sig_num} of {len(pvals)} features significant\n"
    f"[c] = categorical / chi²\n[n] = numeric / ANOVA",
    transform=ax1.transAxes,
    ha="right",
    va="bottom",
    fontsize=8,
    family="monospace",
    color="#555",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="#DDD"),
)
ax1.grid(axis="x", alpha=0.3)
ax1.set_axisbelow(True)

# RIGHT: model accuracies
ax2 = fig.add_subplot(gs[0, 1])

sorted_models = sorted(clf_results.items(), key=lambda kv: kv[1]["test"], reverse=True)
mnames = [m[0].replace(" (", "\n(") for m in sorted_models]
train_acc = [m[1]["train"] for m in sorted_models]
test_acc = [m[1]["test"] for m in sorted_models]

x = np.arange(len(mnames))
w = 0.35
bars_train = ax2.bar(x - w / 2, train_acc, w, label="Train", color=COL_GOOD, edgecolor="white")
bars_test = ax2.bar(x + w / 2, test_acc, w, label="Test", color=COL_PRIMARY, edgecolor="white")

baseline = 1 / df[TARGET].nunique()
ax2.axhline(baseline, color=COL_ACCENT, linestyle="--", linewidth=1.5, label=f"Random baseline = {baseline:.2f}")

for bar, val in zip(bars_train, train_acc):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )
for bar, val in zip(bars_test, test_acc):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

ax2.set_xticks(x)
ax2.set_xticklabels(mnames, fontsize=9)
ax2.set_ylabel("Accuracy")
ax2.set_title("Model performance on 10-class match_outcome\nsorted by test accuracy", pad=12)
ax2.set_ylim(0, 1.15)
ax2.legend(loc="upper right", framealpha=0.95)
ax2.grid(axis="y", alpha=0.3)
ax2.set_axisbelow(True)

rf_train = clf_results["Random Forest (100)"]["train"]
rf_test = clf_results["Random Forest (100)"]["test"]
gap_rf = rf_train - rf_test
ax2.text(
    0.02,
    0.97,
    f"All test accuracies within\n"
    f"±{max(abs(t - baseline) for t in test_acc):.3f} of baseline.\n\n"
    f"Random Forest train–test gap:\n"
    f"{rf_train:.2f} − {rf_test:.2f} = {gap_rf:.2f}\n"
    f"(classic no-signal fingerprint)",
    transform=ax2.transAxes,
    ha="left",
    va="top",
    fontsize=8,
    family="monospace",
    color="#555",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF4E6", edgecolor=COL_HIGHLIGHT),
)

fig.suptitle(
    "Signal test summary: no feature predicts match_outcome above chance",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)
save_fig(fig, "10_signal_summary", FIGDIR)

print()
print("=" * 70)
print("Done. Run scripts/03_alternative_targets.py next for engagement regression.")
print("=" * 70)