"""
02b_signal_test_polished.py — Report-ready plots for the null result and engagement regression.

Same analysis as 02 and 03, but polished visualizations:
- every number labeled on every element
- sorted by meaningful order (not alphabetical)
- diagnostic annotations explain what matters
- report-document sizing (150 dpi)

Run from project root:
    python scripts/02b_signal_test_polished.py

Outputs to reports/figures/polished/:
    07_signal_summary.png           (feature p-values + model accuracy)
    08_regression_diagnostics.png   (actual vs predicted + residuals + metrics)
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.data import load_raw, get_feature_cols, TARGET
from src.preprocessing import label_encode

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
    "font.family": "DejaVu Sans",
})

COL_PRIMARY = "#2E5C8A"
COL_ACCENT = "#C44E52"
COL_MUTED = "#8FA8C4"
COL_HIGHLIGHT = "#E8A33D"
COL_GOOD = "#55A868"


def save(fig, name: str):
    path = FIGDIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [saved] polished/{name}.png")


# =====================================================================
# LOAD + RUN THE SIGNAL ANALYSIS (mirrors script 02)
# =====================================================================
df = load_raw(extended=True)
num_cols, cat_cols = get_feature_cols(df)

print("=" * 70)
print("Running analyses for polished plots")
print("=" * 70)

# Chi-square tests
print("\n→ Chi-square tests on categorical features...")
chi_results = []
for col in cat_cols:
    contingency = pd.crosstab(df[col], df[TARGET])
    chi2, p, dof, _ = chi2_contingency(contingency)
    chi_results.append((col, chi2, p, "categorical"))

# ANOVA tests
print("→ ANOVA on numeric features...")
anova_results = []
classes = df[TARGET].unique()
for col in num_cols:
    groups = [df[df[TARGET] == c][col].values for c in classes]
    f, p = f_oneway(*groups)
    anova_results.append((col, f, p, "numeric"))

# Classifier sanity check
print("→ Training classifiers on match_outcome...")
df_enc, _ = label_encode(df, cat_cols + [TARGET])
feature_cols = [c for c in df_enc.columns if c not in (TARGET, "interest_tags")]
X = df_enc[feature_cols]
y = df_enc[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf_models = {
    "Logistic\nRegression": LogisticRegression(max_iter=500),
    "Random\nForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    "Gradient\nBoosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
}
clf_results = {}
for name, m in clf_models.items():
    m.fit(X_train, y_train)
    clf_results[name] = {
        "train": m.score(X_train, y_train),
        "test": m.score(X_test, y_test),
    }
    print(f"   {name.replace(chr(10), ' ')}: train={clf_results[name]['train']:.3f}, "
          f"test={clf_results[name]['test']:.3f}")


# =====================================================================
# PLOT 07 — Signal summary (polished)
# =====================================================================
# Combine all feature tests, sort by p-value (smallest first = most "significant")
all_results = [(col, p, kind) for col, _, p, kind in chi_results + anova_results]
all_results.sort(key=lambda r: r[1])  # sort ascending by p-value

fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.35)

# -------- LEFT: feature independence tests --------
ax1 = fig.add_subplot(gs[0, 0])

feats = [r[0] for r in all_results]
pvals = [r[1] for r in all_results]
kinds = [r[2] for r in all_results]
neg_log_p = [-np.log10(p) if p > 0 else 10 for p in pvals]

# Color: red if significant, muted otherwise; hatched for numeric vs solid for categorical
colors = [COL_ACCENT if p < 0.05 else COL_MUTED for p in pvals]
# Reverse for horizontal display (most significant at top)
feats_r, pvals_r, colors_r, neg_log_p_r, kinds_r = (
    feats[::-1], pvals[::-1], colors[::-1], neg_log_p[::-1], kinds[::-1]
)

bars = ax1.barh(range(len(feats_r)), neg_log_p_r, color=colors_r, edgecolor="white")
ax1.set_yticks(range(len(feats_r)))
ax1.set_yticklabels(feats_r, fontsize=9)

# Significance threshold as filled zone
threshold = -np.log10(0.05)
ax1.axvline(threshold, color="black", linestyle="--", linewidth=1.3, alpha=0.7)
xmax_data = max(neg_log_p_r) if max(neg_log_p_r) > threshold else threshold * 1.3
ax1.axvspan(threshold, xmax_data * 1.2, alpha=0.08, color=COL_ACCENT,
            label="significant zone (p < 0.05)")

# Label every bar with its exact p-value + kind marker
for bar, p, nlp, k in zip(bars, pvals_r, neg_log_p_r, kinds_r):
    # Format p-value for readability
    if p < 0.001:
        plabel = f"p = {p:.1e}"
    elif p < 0.01:
        plabel = f"p = {p:.3f}"
    else:
        plabel = f"p = {p:.3f}"
    kmark = "[c]" if k == "categorical" else "[n]"
    ax1.text(
        nlp + xmax_data * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{plabel}  {kmark}",
        ha="left", va="center", fontsize=8.5,
        fontweight="bold" if p < 0.05 else "normal",
    )

ax1.set_xlim(0, xmax_data * 1.35)
ax1.set_xlabel("−log₁₀(p-value)    →    higher = more significant", fontsize=10)
ax1.set_title("Feature → match_outcome independence tests\nsorted by p-value (most significant at top)",
              pad=12)

# Legend — upper right, summary below
legend_elements = [
    Patch(facecolor=COL_ACCENT, label="p < 0.05  (significant)"),
    Patch(facecolor=COL_MUTED, label="p ≥ 0.05  (not significant)"),
]
ax1.legend(handles=legend_elements, loc="center right", framealpha=0.95)

# Annotation: count significant / total — bottom-right corner
n_sig = sum(1 for p in pvals if p < 0.05)
n_total = len(pvals)
ax1.text(
    0.98, 0.02,
    f"{n_sig} of {n_total} features significant\n"
    f"[c] = categorical / chi²\n[n] = numeric / ANOVA",
    transform=ax1.transAxes, ha="right", va="bottom",
    fontsize=8, family="monospace", color="#555",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="#DDD"),
)
ax1.grid(axis="x", alpha=0.3)
ax1.set_axisbelow(True)


# -------- RIGHT: model accuracies --------
ax2 = fig.add_subplot(gs[0, 1])

# Sort by test accuracy (descending)
sorted_models = sorted(clf_results.items(), key=lambda kv: kv[1]["test"], reverse=True)
mnames = [m[0] for m in sorted_models]
train_acc = [m[1]["train"] for m in sorted_models]
test_acc = [m[1]["test"] for m in sorted_models]

x = np.arange(len(mnames))
w = 0.35
bars_train = ax2.bar(x - w / 2, train_acc, w, label="Train", color=COL_GOOD, edgecolor="white")
bars_test = ax2.bar(x + w / 2, test_acc, w, label="Test", color=COL_PRIMARY, edgecolor="white")

# Baseline reference line
baseline = 1 / df[TARGET].nunique()
ax2.axhline(baseline, color=COL_ACCENT, linestyle="--", linewidth=1.5,
            label=f"Random baseline = {baseline:.2f}")

# Label every bar with its value
for bar, val in zip(bars_train, train_acc):
    ax2.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
        f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
    )
for bar, val in zip(bars_test, test_acc):
    ax2.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
        f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
    )

ax2.set_xticks(x)
ax2.set_xticklabels(mnames, fontsize=9)
ax2.set_ylabel("Accuracy")
ax2.set_title("Model performance on 10-class match_outcome\nsorted by test accuracy",
              pad=12)
ax2.set_ylim(0, 1.15)
ax2.legend(loc="upper right", framealpha=0.95)
ax2.grid(axis="y", alpha=0.3)
ax2.set_axisbelow(True)

# Annotation: the key insight
gap_rf = clf_results["Random\nForest"]["train"] - clf_results["Random\nForest"]["test"]
rf_train = clf_results["Random\nForest"]["train"]
rf_test = clf_results["Random\nForest"]["test"]
ax2.text(
    0.02, 0.97,
    f"All test accuracies within\n"
    f"±{max(abs(t - baseline) for t in test_acc):.3f} of baseline.\n\n"
    f"Random Forest train–test gap:\n"
    f"{rf_train:.2f} − {rf_test:.2f} = {gap_rf:.2f}\n"
    f"(classic no-signal fingerprint)",
    transform=ax2.transAxes, ha="left", va="top",
    fontsize=8, family="monospace", color="#555",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF4E6", edgecolor=COL_HIGHLIGHT),
)

fig.suptitle(
    "Signal test summary: no feature predicts match_outcome above chance",
    fontsize=13, fontweight="bold", y=1.02,
)
save(fig, "07_signal_summary")


# =====================================================================
# PLOT 08 — Regression diagnostics (polished)
# =====================================================================
print("\n→ Training regressors on mutual_matches...")

drop_cols = ["mutual_matches", "interest_tags", TARGET]
X_reg = df_enc.drop(columns=drop_cols)
y_reg = df["mutual_matches"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_rs = scaler.fit_transform(X_train_r)
X_test_rs = scaler.transform(X_test_r)

reg_models = {
    "Linear Regression": (LinearRegression(), True),
    "Ridge": (Ridge(alpha=1.0), True),
    "Random Forest": (RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42), False),
    "Gradient Boosting": (GradientBoostingRegressor(n_estimators=100, random_state=42), False),
}

reg_results = {}
for name, (m, scale) in reg_models.items():
    Xtr, Xte = (X_train_rs, X_test_rs) if scale else (X_train_r, X_test_r)
    m.fit(Xtr, y_train_r)
    preds = m.predict(Xte)
    reg_results[name] = {
        "r2": r2_score(y_test_r, preds),
        "mae": mean_absolute_error(y_test_r, preds),
        "rmse": np.sqrt(mean_squared_error(y_test_r, preds)),
        "preds": preds,
    }
    print(f"   {name}: R²={reg_results[name]['r2']:.3f}, "
          f"MAE={reg_results[name]['mae']:.2f}, RMSE={reg_results[name]['rmse']:.2f}")

# Best model
best_name = max(reg_results, key=lambda n: reg_results[n]["r2"])
best = reg_results[best_name]
preds = best["preds"]
residuals = y_test_r.values - preds

# Build figure with 3 panels: model comparison, actual-vs-predicted, residuals
fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1], wspace=0.35)

# -------- PANEL 1: model comparison (bar chart of R²) --------
ax1 = fig.add_subplot(gs[0, 0])
sorted_reg = sorted(reg_results.items(), key=lambda kv: kv[1]["r2"], reverse=True)
reg_names = [n for n, _ in sorted_reg]
r2_vals = [r["r2"] for _, r in sorted_reg]
mae_vals = [r["mae"] for _, r in sorted_reg]

colors_r2 = [COL_PRIMARY if n == best_name else COL_MUTED for n in reg_names]
bars = ax1.barh(range(len(reg_names)), r2_vals, color=colors_r2, edgecolor="white")
ax1.set_yticks(range(len(reg_names)))
ax1.set_yticklabels(reg_names, fontsize=9)
ax1.invert_yaxis()  # best on top

# Label R² + MAE on each bar
for bar, r2, mae in zip(bars, r2_vals, mae_vals):
    ax1.text(
        bar.get_width() + max(r2_vals) * 0.02,
        bar.get_y() + bar.get_height() / 2,
        f"R² = {r2:.3f}\nMAE = {mae:.2f}",
        ha="left", va="center", fontsize=8.5,
        fontweight="bold" if r2 == max(r2_vals) else "normal",
    )
ax1.set_xlim(0, max(r2_vals) * 1.6)
ax1.set_xlabel("R² (test set)", fontsize=10)
ax1.set_title(f"Model comparison\ntarget = mutual_matches", pad=10)
ax1.grid(axis="x", alpha=0.3)
ax1.set_axisbelow(True)

# Annotation
ax1.text(
    0.98, 0.02,
    f"Best: {best_name}\nWinner highlighted in blue",
    transform=ax1.transAxes, ha="right", va="bottom",
    fontsize=8, family="monospace", color="#555",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", edgecolor="#DDD"),
)

# -------- PANEL 2: actual vs predicted (hexbin) --------
ax2 = fig.add_subplot(gs[0, 1])

y_true = y_test_r.values
xy_min = min(y_true.min(), preds.min()) - 0.5
xy_max = max(y_true.max(), preds.max()) + 0.5

hb = ax2.hexbin(y_true, preds, gridsize=30, cmap="Blues", mincnt=1,
                extent=(xy_min, xy_max, xy_min, xy_max))
cb = fig.colorbar(hb, ax=ax2, shrink=0.85, pad=0.02)
cb.set_label("count", fontsize=9)

# Perfect prediction line
ax2.plot([xy_min, xy_max], [xy_min, xy_max], "--", color=COL_ACCENT,
         linewidth=1.8, label="perfect prediction")

# Mean prediction reference
pred_mean = preds.mean()
ax2.axhline(pred_mean, color=COL_HIGHLIGHT, linestyle=":", linewidth=1.5,
            label=f"mean prediction = {pred_mean:.1f}")

# Fitted trend (ordinary least squares through the predictions)
slope, intercept = np.polyfit(y_true, preds, 1)
xs = np.array([xy_min, xy_max])
ax2.plot(xs, slope * xs + intercept, color="black", linewidth=1.5, alpha=0.7,
         label=f"fitted: y = {slope:.2f}x + {intercept:.2f}")

ax2.set_xlim(xy_min, xy_max)
ax2.set_ylim(xy_min, xy_max)
ax2.set_xlabel("Actual mutual_matches", fontsize=10)
ax2.set_ylabel("Predicted mutual_matches", fontsize=10)
ax2.set_title(
    f"{best_name}: actual vs predicted\n"
    f"R² = {best['r2']:.3f}  |  MAE = {best['mae']:.2f}  |  RMSE = {best['rmse']:.2f}",
    pad=10, fontsize=11,
)
ax2.legend(loc="upper left", fontsize=8, framealpha=0.95)
ax2.set_aspect("equal")

# Diagnostic annotation
ax2.text(
    0.98, 0.02,
    f"fitted slope = {slope:.2f}\n"
    f"(perfect model = 1.00)\n\n"
    f"flatter slope = more\nregression to the mean",
    transform=ax2.transAxes, ha="right", va="bottom",
    fontsize=8, family="monospace", color="#555",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF4E6", edgecolor=COL_HIGHLIGHT),
)

# -------- PANEL 3: residuals distribution --------
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(residuals, bins=40, color=COL_PRIMARY, edgecolor="white", alpha=0.85)
ax3.axvline(0, color=COL_ACCENT, linestyle="--", linewidth=1.5, label="zero error")
ax3.axvline(residuals.mean(), color=COL_HIGHLIGHT, linestyle="-", linewidth=1.8,
            label=f"mean = {residuals.mean():+.2f}")

ax3.set_xlabel("Residual = actual − predicted", fontsize=10)
ax3.set_ylabel("count")
ax3.set_title("Residual distribution", pad=10)
ax3.legend(loc="upper right", fontsize=8, framealpha=0.95)

# Stats
ax3.text(
    0.02, 0.97,
    f"n = {len(residuals):,}\n"
    f"mean = {residuals.mean():+.2f}\n"
    f"std = {residuals.std():.2f}\n"
    f"min = {residuals.min():+.1f}\n"
    f"max = {residuals.max():+.1f}",
    transform=ax3.transAxes, ha="left", va="top",
    fontsize=8, family="monospace", color="#333",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#DDD"),
)
ax3.grid(axis="y", alpha=0.3)
ax3.set_axisbelow(True)

fig.suptitle(
    f"Engagement regression: partial but real signal on mutual_matches (best R² = {best['r2']:.3f})",
    fontsize=13, fontweight="bold", y=1.02,
)
save(fig, "08_regression_diagnostics")


print()
print("=" * 70)
print(f"Done. Polished plots saved to: {FIGDIR.relative_to(ROOT)}")
print("=" * 70)