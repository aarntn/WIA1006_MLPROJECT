"""
03_alternative_targets.py — Regression on engagement metrics.

match_outcome has no signal (see script 02). But mutual_matches and likes_received
do carry some partial signal. This script explores that.

Run from project root:
    python scripts/03_alternative_targets.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, PoissonRegressor, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import get_feature_cols, load_raw
from src.preprocessing import label_encode

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def tail_metrics(y_true, y_pred, quantile=0.90):
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


def build_models():
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
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "Ridge (log1p target)": TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "Gradient Boosting (log1p target)": TransformedTargetRegressor(
            regressor=GradientBoostingRegressor(n_estimators=100, random_state=42),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "Poisson Regressor": make_pipeline(
            StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=1000)
        ),
    }


def evaluate_setting(X, y, setting_name):
    models = build_models()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print(
        f"\n[{setting_name}]"
    )
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

    baseline_names = [
        "Linear Regression",
        "Ridge",
        "Gradient Boosting",
        "Random Forest",
    ]
    advanced_names = [
        "Linear (log1p target)",
        "Ridge (log1p target)",
        "Gradient Boosting (log1p target)",
        "Poisson Regressor",
    ]

    best_baseline = min((results[n] for n in baseline_names), key=lambda r: r["cv_mae"])
    best_advanced = min((results[n] for n in advanced_names), key=lambda r: r["cv_mae"])
    best_baseline_r2 = max(results[n]["cv_r2"] for n in baseline_names)
    best_advanced_r2 = max(results[n]["cv_r2"] for n in advanced_names)

    print("\nInterpretation:")
    print(
        f"Top baseline MAE: {best_baseline['cv_mae']:.3f} | "
        f"Top transformed/count-aware MAE: {best_advanced['cv_mae']:.3f}"
    )
    print(
        f"Top baseline R²: {best_baseline_r2:.3f} | "
        f"Top transformed/count-aware R²: {best_advanced_r2:.3f}"
    )
    print(
        f"Top baseline tail MAE (90th pct): "
        f"{min(results[n]['tail_mae'] for n in baseline_names):.3f} | "
        f"Top transformed/count-aware tail MAE: "
        f"{min(results[n]['tail_mae'] for n in advanced_names):.3f}"
    )

    return results


# Load data

df = load_raw(extended=True)
num_cols, cat_cols = get_feature_cols(df)

# ---- Correlation check ----
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


# ---- Regression ----
def evaluate_target(df, target_col):
    section(f"Regression task: predict `{target_col}`")

    df_enc, _ = label_encode(df, cat_cols)
    y = df[target_col].astype(float)

    paired_target_map = {
        "mutual_matches": "likes_received",
        "likes_received": "mutual_matches",
    }
    paired_target_col = paired_target_map.get(target_col)

    base_drop_cols = [target_col, "interest_tags", "match_outcome"]
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

    print(
        "\nSummary comparison (best CV-R² by setting):"
    )
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
        "Treat the 'without paired target feature' setting as the safer estimate "
        "of realistic generalization."
    )
    print(
        f"Best-model R² gap (with - without): "
        f"{best_with['cv_r2'] - best_without['cv_r2']:.3f}"
    )

    return setting_results


# Run for both candidates
results_matches = evaluate_target(df, "mutual_matches")
results_likes = evaluate_target(df, "likes_received")

# ---- Visual: predicted vs actual (best safe model on mutual_matches) ----
section("Plot: predicted vs actual (best safe CV model, target=mutual_matches)")
matches_without_paired = results_matches["without paired target feature"]
best_name = max(matches_without_paired, key=lambda n: matches_without_paired[n]["cv_r2"])
best = matches_without_paired[best_name]

y_test = best["holdout_y_test"]
preds = best["holdout_preds"]

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, preds, alpha=0.15, s=12, color="#4C72B0")
lim = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
ax.plot(lim, lim, "--", color="red", label="perfect prediction")
ax.set_xlabel("Actual mutual_matches")
ax.set_ylabel("Predicted mutual_matches")
ax.set_title(f"{best_name} — CV R²={best['cv_r2']:.3f}")
ax.legend()
plt.tight_layout()
plt.savefig(FIGDIR / "08_regression_predicted_vs_actual.png")
plt.close()
print("[saved] reports/figures/08_regression_predicted_vs_actual.png")

# ---- Summary ----
section("Takeaway")
print(
    f"""
Best CV R² for mutual_matches (with paired feature): {max(r['cv_r2'] for r in results_matches['with paired target feature'].values()):.3f}
Best CV R² for mutual_matches (without paired feature): {max(r['cv_r2'] for r in results_matches['without paired target feature'].values()):.3f}
Best CV R² for likes_received (with paired feature): {max(r['cv_r2'] for r in results_likes['with paired target feature'].values()):.3f}
Best CV R² for likes_received (without paired feature): {max(r['cv_r2'] for r in results_likes['without paired target feature'].values()):.3f}

The with-vs-without paired-feature gap shows how much the models benefit from
using the adjacent engagement target directly. For a safer estimate of real
predictive value, prefer the "without paired target feature" results.

Against the dummy mean baseline, positive R² lift and lower MAE indicate the
models are learning some non-trivial structure. Log1p-target and Poisson
variants are included because these targets are non-negative and count-like,
which may improve both overall error and high-tail behavior.
"""
)
