"""
03_alternative_targets.py — Regression on engagement metrics with leakage testing.

Tests both with and without the paired engagement target (e.g. likes_received
when predicting mutual_matches), so the realistic-generalization estimate is
visible separately from the leakage-inflated one.

Run from project root:
    python scripts/03_alternative_targets.py

Outputs:
    - printed cross-validated R² and MAE per model, per setting
    - reports/figures/11_regression_diagnostics.png  (3-panel polished plot)
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, PoissonRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import TARGET, get_feature_cols, load_raw
from src.preprocessing import label_encode

# setup 
ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

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


def tail_metrics(y_true, y_pred, quantile: float = 0.90):
    """MAE and R² restricted to the high-value tail of the true target."""
    threshold = np.quantile(y_true, quantile)
    mask = y_true >= threshold
    if mask.sum() < 3:
        return np.nan, np.nan, threshold
    tail_mae = mean_absolute_error(y_true[mask], y_pred[mask])
    if np.unique(y_true[mask]).size < 2:
        tail_r2 = np.nan
    else:
        tail_r2 = r2_score(y_true[mask], y_pred[mask])
    return tail_mae, tail_r2, threshold


def build_models() -> dict:
    return {
        "Dummy (mean baseline)": DummyRegressor(strategy="mean"),
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
        "Linear (log1p target)": TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), LinearRegression()),
            func=np.log1p, inverse_func=np.expm1,
        ),
        "Ridge (log1p target)": TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            func=np.log1p, inverse_func=np.expm1,
        ),
        "Gradient Boosting (log1p target)": TransformedTargetRegressor(
            regressor=GradientBoostingRegressor(n_estimators=100, random_state=42),
            func=np.log1p, inverse_func=np.expm1,
        ),
        "Poisson Regressor": make_pipeline(
            StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=1000)
        ),
    }


def evaluate_setting(X, y, setting_name: str) -> dict:
    models = build_models()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n[{setting_name}]")
    print(
        f"{'Model':<36} {'R² (CV)':<10} {'MAE (CV)':<10} "
        f"{'Tail MAE@90%':<13} {'Tail R²@90%':<12}"
    )
    print("-" * 90)

    results = {}
    for name, estimator in models.items():
        preds = cross_val_predict(estimator, X, y, cv=cv, n_jobs=-1, method="predict")
        cv_r2 = r2_score(y, preds)
        cv_mae = mean_absolute_error(y, preds)
        tail_mae, tail_r2, tail_threshold = tail_metrics(
            y.to_numpy(), preds, quantile=0.90
        )
        results[name] = {
            "cv_r2": cv_r2,
            "cv_mae": cv_mae,
            "tail_mae": tail_mae,
            "tail_r2": tail_r2,
            "tail_threshold": tail_threshold,
            "cv_preds": preds,
        }
        print(
            f"{name:<36} {cv_r2:<10.3f} {cv_mae:<10.3f} "
            f"{tail_mae:<13.3f} {tail_r2:<12.3f}"
        )

    baseline = results["Dummy (mean baseline)"]
    print("\nAgainst baseline:")
    for name, metrics in results.items():
        if name == "Dummy (mean baseline)":
            continue
        mae_improvement = ((baseline["cv_mae"] - metrics["cv_mae"]) / baseline["cv_mae"]) * 100
        r2_lift = metrics["cv_r2"] - baseline["cv_r2"]
        print(
            f"  - {name:<32} MAE vs baseline: {mae_improvement:+.2f}% | "
            f"R² lift: {r2_lift:+.3f}"
        )

    best_name = max(results, key=lambda n: results[n]["cv_r2"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    holdout_preds = best_model.predict(X_test)
    results[best_name]["holdout_preds"] = holdout_preds
    results[best_name]["holdout_y_test"] = y_test

    return results


def evaluate_target(df, target_col: str, cat_cols) -> dict:
    section(f"Regression task: predict `{target_col}`")

    df_enc, _ = label_encode(df, cat_cols)
    y = df[target_col].astype(float)

    paired_target_map = {
        "mutual_matches": "likes_received",
        "likes_received": "mutual_matches",
    }
    paired_target_col = paired_target_map.get(target_col)

    base_drop_cols = [target_col, "interest_tags", TARGET]
    settings = [
        ("with paired target feature", False),
        ("without paired target feature", True),
    ]

    setting_results = {}
    for setting_name, drop_paired_target in settings:
        drop_cols = base_drop_cols.copy()
        if drop_paired_target and paired_target_col is not None:
            drop_cols.append(paired_target_col)
        X = df_enc.drop(columns=drop_cols)
        setting_results[setting_name] = evaluate_setting(X, y, setting_name)

    print("\nSummary comparison (best CV-R² by setting):")
    best_with_name = max(
        setting_results["with paired target feature"],
        key=lambda n: setting_results["with paired target feature"][n]["cv_r2"],
    )
    best_without_name = max(
        setting_results["without paired target feature"],
        key=lambda n: setting_results["without paired target feature"][n]["cv_r2"],
    )
    best_with = setting_results["with paired target feature"][best_with_name]
    best_without = setting_results["without paired target feature"][best_without_name]

    print(
        f"  with paired feature    -> {best_with_name}: "
        f"R²={best_with['cv_r2']:.3f}, MAE={best_with['cv_mae']:.3f}"
    )
    print(
        f"  without paired feature -> {best_without_name}: "
        f"R²={best_without['cv_r2']:.3f}, MAE={best_without['cv_mae']:.3f}"
    )
    print(
        "Performance is usually higher when the paired target feature is included. "
        "Treat the 'without paired target feature' setting as the safer estimate."
    )
    print(
        f"Best-model R² gap (with - without): "
        f"{best_with['cv_r2'] - best_without['cv_r2']:.3f}"
    )
    return setting_results

# LOAD + ANALYSIS
df = load_raw(extended=True)
num_cols, cat_cols = get_feature_cols(df)

section("Correlation matrix — key engagement metrics")
engagement_cols = [
    "app_usage_time_min",
    "swipe_right_ratio",
    "likes_received",
    "mutual_matches",
    "message_sent_count",
    "emoji_usage_rate",
    "profile_pics_count",
    "bio_length",
]
print(df[engagement_cols].corr().round(3))

section("Target candidates — how much variance do they have?")
for col in ["likes_received", "mutual_matches"]:
    print(
        f"{col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, "
        f"range=[{df[col].min()}, {df[col].max()}]"
    )

results_matches = evaluate_target(df, "mutual_matches", cat_cols)
results_likes = evaluate_target(df, "likes_received", cat_cols)

# PLOT 11 — Regression diagnostics (polished, 3-panel)
section("Plot: regression diagnostics for mutual_matches (best safe-CV model)")

matches_with = results_matches["with paired target feature"]
matches_without = results_matches["without paired target feature"]

best_name = max(matches_with, key=lambda n: matches_with[n]["cv_r2"])
best = matches_with[best_name]

y_test = best["holdout_y_test"]
preds = best["holdout_preds"]

r2_holdout = r2_score(y_test, preds)
mae_holdout = mean_absolute_error(y_test, preds)
rmse_holdout = float(np.sqrt(mean_squared_error(y_test, preds)))
residuals = y_test.values - preds

fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1], wspace=0.35)

# PANEL 1: model comparison
ax1 = fig.add_subplot(gs[0, 0])

model_names = [n for n in matches_with if n != "Dummy (mean baseline)"]
model_names.sort(key=lambda n: matches_with[n]["cv_r2"], reverse=True)

n_models = len(model_names)
bar_h = 0.4
y_pos = np.arange(n_models)

r2_with = [matches_with[n]["cv_r2"] for n in model_names]
r2_without = [matches_without[n]["cv_r2"] for n in model_names]

bars_with = ax1.barh(
    y_pos - bar_h / 2,
    r2_with,
    bar_h,
    color=COL_PRIMARY,
    edgecolor="white",
    label="with paired feature",
)
bars_without = ax1.barh(
    y_pos + bar_h / 2,
    r2_without,
    bar_h,
    color=COL_MUTED,
    edgecolor="white",
    label="without paired feature",
)

ax1.axvline(0, color="#999", linestyle=":", linewidth=1)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(
    [n.replace(" (log1p target)", "\n(log1p target)") for n in model_names],
    fontsize=8,
)
ax1.invert_yaxis()
ax1.set_xlabel("R² (5-fold CV)")
ax1.set_title("Model comparison — leakage isolated\ntarget = mutual_matches", pad=10)

for bar, val in zip(bars_with, r2_with):
    ax1.text(
        max(bar.get_width(), 0) + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{val:+.3f}",
        ha="left",
        va="center",
        fontsize=7.5,
        color=COL_PRIMARY,
        fontweight="bold",
    )
for bar, val in zip(bars_without, r2_without):
    ax1.text(
        max(bar.get_width(), 0) + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{val:+.3f}",
        ha="left",
        va="center",
        fontsize=7.5,
        color="#555",
    )

ax1.legend(loc="lower right", fontsize=8, framealpha=0.95)
ax1.grid(axis="x", alpha=0.3)
ax1.set_axisbelow(True)

gap = max(r2_with) - max(r2_without)
ax1.text(
    0.98,
    0.98,
    f"Best with paired:    {max(r2_with):+.3f}\n"
    f"Best without paired: {max(r2_without):+.3f}\n"
    f"Gap (= leakage):     {gap:+.3f}",
    transform=ax1.transAxes,
    ha="right",
    va="top",
    fontsize=8,
    family="monospace",
    color="#555",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF4E6", edgecolor=COL_HIGHLIGHT),
)

# PANEL 2: actual vs predicted
ax2 = fig.add_subplot(gs[0, 1])

y_true = y_test.values
xy_min = min(y_true.min(), preds.min()) - 0.5
xy_max = max(y_true.max(), preds.max()) + 0.5

hb = ax2.hexbin(
    y_true,
    preds,
    gridsize=30,
    cmap="Blues",
    mincnt=1,
    extent=(xy_min, xy_max, xy_min, xy_max),
)
cb = fig.colorbar(hb, ax=ax2, shrink=0.85, pad=0.02)
cb.set_label("count", fontsize=9)

ax2.plot(
    [xy_min, xy_max],
    [xy_min, xy_max],
    "--",
    color=COL_ACCENT,
    linewidth=1.8,
    label="perfect prediction",
)
pred_mean = preds.mean()
ax2.axhline(
    pred_mean,
    color=COL_HIGHLIGHT,
    linestyle=":",
    linewidth=1.5,
    label=f"mean prediction = {pred_mean:.1f}",
)
slope, intercept = np.polyfit(y_true, preds, 1)
xs = np.array([xy_min, xy_max])
ax2.plot(
    xs,
    slope * xs + intercept,
    color="black",
    linewidth=1.5,
    alpha=0.7,
    label=f"fitted: y = {slope:.2f}x + {intercept:.2f}",
)

ax2.set_xlim(xy_min, xy_max)
ax2.set_ylim(xy_min, xy_max)
ax2.set_xlabel("Actual mutual_matches", fontsize=10)
ax2.set_ylabel("Predicted mutual_matches", fontsize=10)
ax2.set_title(
    f"{best_name}: actual vs predicted (with paired feature)\n"
    f"R² = {r2_holdout:.3f}  |  MAE = {mae_holdout:.2f}  |  RMSE = {rmse_holdout:.2f}",
    pad=10,
    fontsize=11,
)
ax2.legend(loc="upper left", fontsize=8, framealpha=0.95)
ax2.set_aspect("equal")

ax2.text(
    0.98,
    0.02,
    f"fitted slope = {slope:.2f}\n"
    f"(perfect model = 1.00)\n\n"
    f"flatter slope = more\nregression to the mean",
    transform=ax2.transAxes,
    ha="right",
    va="bottom",
    fontsize=8,
    family="monospace",
    color="#555",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF4E6", edgecolor=COL_HIGHLIGHT),
)

# PANEL 3: residuals distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(residuals, bins=40, color=COL_PRIMARY, edgecolor="white", alpha=0.85)
ax3.axvline(0, color=COL_ACCENT, linestyle="--", linewidth=1.5, label="zero error")
ax3.axvline(
    residuals.mean(),
    color=COL_HIGHLIGHT,
    linestyle="-",
    linewidth=1.8,
    label=f"mean = {residuals.mean():+.2f}",
)

ax3.set_xlabel("Residual = actual − predicted", fontsize=10)
ax3.set_ylabel("count")
ax3.set_title("Residual distribution", pad=10)
ax3.legend(loc="upper right", fontsize=8, framealpha=0.95)

ax3.text(
    0.02,
    0.97,
    f"n = {len(residuals):,}\n"
    f"mean = {residuals.mean():+.2f}\n"
    f"std = {residuals.std():.2f}\n"
    f"min = {residuals.min():+.1f}\n"
    f"max = {residuals.max():+.1f}",
    transform=ax3.transAxes,
    ha="left",
    va="top",
    fontsize=8,
    family="monospace",
    color="#333",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#DDD"),
)
ax3.grid(axis="y", alpha=0.3)
ax3.set_axisbelow(True)

fig.suptitle(
    f"Engagement regression on mutual_matches "
    f"(best with-paired R² = {max(r2_with):.3f}, without-paired R² = {max(r2_without):.3f})",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)
save_fig(fig, "11_regression_diagnostics", FIGDIR)

# Final takeaway
section("Takeaway")
best_with_mm = max(r["cv_r2"] for r in results_matches["with paired target feature"].values())
best_without_mm = max(r["cv_r2"] for r in results_matches["without paired target feature"].values())
best_with_lr = max(r["cv_r2"] for r in results_likes["with paired target feature"].values())
best_without_lr = max(r["cv_r2"] for r in results_likes["without paired target feature"].values())

print(f"""
Best CV R² for mutual_matches (with paired feature):     {best_with_mm:+.3f}
Best CV R² for mutual_matches (without paired feature):  {best_without_mm:+.3f}
Best CV R² for likes_received (with paired feature):     {best_with_lr:+.3f}
Best CV R² for likes_received (without paired feature):  {best_without_lr:+.3f}

The with-vs-without paired-feature gap shows how much the models benefit from
using the adjacent engagement target directly. For a safer estimate of real
predictive value, prefer the "without paired target feature" results.

Against the dummy mean baseline, positive R² lift and lower MAE indicate the
models are learning some non-trivial structure. Log1p-target and Poisson
variants are included because these targets are non-negative and count-like,
which may improve both overall error and high-tail behavior.
""")