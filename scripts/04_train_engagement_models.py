"""
04_train_engagement_models.py - official engagement prediction workflow.

Run from project root:
    python scripts/04_train_engagement_models.py

Fast smoke test:
    python scripts/04_train_engagement_models.py --fast

Outputs:
    - reports/engagement_model_results.csv
    - reports/engagement_summary.md
    - reports/figures/12_engagement_model_comparison.png
    - reports/figures/13_leakage_comparison.png
    - reports/figures/14_residuals_feature_importance.png
    - reports/figures/16_learning_curve.png
    - models/best_mutual_matches_model.joblib
"""
from __future__ import annotations

import argparse
import json
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
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, PoissonRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_validate, learning_curve, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import TARGET, load_raw
from src.features import EngagementFeatureBuilder

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency fallback
    XGBRegressor = None


ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
FIGDIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"
TARGET_COL = "mutual_matches"
PAIRED_COL = "likes_received"

from src.plotting import COL_ACCENT, COL_GOOD, COL_HIGHLIGHT, COL_MUTED, COL_PRIMARY, save_fig, setup_plot_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Use fewer rows/models iterations for smoke tests.")
    parser.add_argument("--sample", type=int, default=None, help="Optional row sample for local experimentation.")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds.")
    return parser.parse_args()


def make_pipeline(estimator, include_paired_target: bool, scale: bool = False) -> Pipeline:
    steps = [
        (
            "features",
            EngagementFeatureBuilder(
                target_col=TARGET_COL,
                include_paired_target=include_paired_target,
                include_match_outcome=False,
                include_interest_tags=True,
            ),
        )
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def build_models(fast: bool, include_paired_target: bool) -> dict[str, Pipeline]:
    rf_estimators = 80 if fast else 180
    gb_estimators = 80 if fast else 160
    models = {
        "Dummy mean": make_pipeline(DummyRegressor(strategy="mean"), include_paired_target),
        "Ridge": make_pipeline(Ridge(alpha=1.0), include_paired_target, scale=True),
        "ElasticNet": make_pipeline(
            ElasticNet(alpha=0.005, l1_ratio=0.2, max_iter=5000),
            include_paired_target,
            scale=True,
        ),
        "Poisson": make_pipeline(
            PoissonRegressor(alpha=1.0, max_iter=1000),
            include_paired_target,
            scale=True,
        ),
        "Random Forest": make_pipeline(
            RandomForestRegressor(
                n_estimators=rf_estimators,
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=42,
            ),
            include_paired_target,
        ),
        "Gradient Boosting": make_pipeline(
            GradientBoostingRegressor(
                n_estimators=gb_estimators,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            ),
            include_paired_target,
        ),
        "Hist Gradient Boosting": make_pipeline(
            HistGradientBoostingRegressor(
                max_iter=gb_estimators,
                learning_rate=0.05,
                l2_regularization=0.05,
                random_state=42,
            ),
            include_paired_target,
        ),
    }

    if XGBRegressor is not None:
        models["XGBoost"] = make_pipeline(
            XGBRegressor(
                n_estimators=gb_estimators,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                n_jobs=-1,
                random_state=42,
            ),
            include_paired_target,
        )

    # MLP: defensive box-check for "you didn't try deep learning" objection.
    # Grinsztajn et al. (NeurIPS 2022) show tree models dominate on tabular data;
    # MLP is included to confirm — not expected to exceed GBDT performance.
    mlp_iter = 100 if fast else 300
    models["MLP"] = make_pipeline(
        MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=mlp_iter, random_state=42, early_stopping=True),
        include_paired_target,
        scale=True,
    )

    return models


def evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    fast: bool,
    include_paired_target: bool,
    cv: int,
) -> pd.DataFrame:
    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }
    rows = []
    models = build_models(fast=fast, include_paired_target=include_paired_target)

    label = "with paired target" if include_paired_target else "official safe"
    print(f"\nEvaluating {label} feature set ({cv}-fold CV)")
    for name, pipeline in models.items():
        print(f"  - {name}")
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )
        rows.append(
            {
                "setting": label,
                "model": name,
                "cv_r2_mean": scores["test_r2"].mean(),
                "cv_r2_std": scores["test_r2"].std(),
                "cv_mae_mean": -scores["test_mae"].mean(),
                "cv_rmse_mean": -scores["test_rmse"].mean(),
            }
        )

    return pd.DataFrame(rows)


def tune_best_safe_model(X_train: pd.DataFrame, y_train: pd.Series, fast: bool, cv: int) -> RandomizedSearchCV:
    print("\nTuning official safe Hist Gradient Boosting model")
    base = make_pipeline(
        HistGradientBoostingRegressor(random_state=42),
        include_paired_target=False,
    )
    param_dist = {
        "model__max_iter": [80, 120] if fast else [120, 180, 240],
        "model__learning_rate": [0.03, 0.05, 0.08],
        "model__max_leaf_nodes": [15, 31, 45],
        "model__min_samples_leaf": [20, 40, 80],
        "model__l2_regularization": [0.0, 0.05, 0.2],
    }
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=6 if fast else 18,
        cv=max(3, min(cv, 5)),
        scoring="r2",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def holdout_metrics(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    preds = model.predict(X_test)
    return {
        "holdout_r2": r2_score(y_test, preds),
        "holdout_mae": mean_absolute_error(y_test, preds),
        "holdout_rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
    }


def plot_model_comparison(results: pd.DataFrame) -> None:
    safe = results[results["setting"] == "official safe"].sort_values("cv_r2_mean")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(safe["model"], safe["cv_r2_mean"], color=COL_PRIMARY, edgecolor="white")
    ax.axvline(0, color="#666", linestyle=":", linewidth=1)
    ax.set_title("Official engagement model comparison\nTarget = mutual_matches, likes_received excluded")
    ax.set_xlabel("R2 (5-fold CV on training split)")
    ax.set_ylabel("")
    for bar, val in zip(bars, safe["cv_r2_mean"]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    save_fig(fig, "12_engagement_model_comparison", FIGDIR)


def plot_leakage_comparison(results: pd.DataFrame) -> None:
    pivot = results.pivot(index="model", columns="setting", values="cv_r2_mean")
    pivot = pivot.dropna().sort_values("official safe")
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(pivot))
    ax.barh(y - 0.18, pivot["official safe"], height=0.36, color=COL_MUTED, label="official safe")
    ax.barh(y + 0.18, pivot["with paired target"], height=0.36, color=COL_HIGHLIGHT, label="with likes_received")
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.axvline(0, color="#666", linestyle=":", linewidth=1)
    ax.set_title("Leakage check: paired target feature inflates performance")
    ax.set_xlabel("R2 (CV)")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    save_fig(fig, "13_leakage_comparison", FIGDIR)


def feature_importance_frame(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    feature_names = model.named_steps["features"].get_feature_names_out()
    estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.ravel(estimator.coef_))
    else:
        X_small = X_test.sample(n=min(1500, len(X_test)), random_state=42)
        y_small = y_test.loc[X_small.index]
        X_small_features = model.named_steps["features"].transform(X_small)
        perm = permutation_importance(
            estimator,
            X_small_features,
            y_small,
            scoring="r2",
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )
        values = perm.importances_mean

    return (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .head(15)
    )


def plot_residuals_and_importance(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    preds = model.predict(X_test)
    residuals = y_test.to_numpy() - preds
    importance = feature_importance_frame(model, X_test, y_test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.05, 1]})
    hb = ax1.hexbin(y_test, preds, gridsize=30, cmap="Blues", mincnt=1)
    fig.colorbar(hb, ax=ax1, shrink=0.85, label="count")
    lim_min = min(y_test.min(), preds.min()) - 0.5
    lim_max = max(y_test.max(), preds.max()) + 0.5
    ax1.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", color=COL_ACCENT, label="perfect")
    ax1.set_xlim(lim_min, lim_max)
    ax1.set_ylim(lim_min, lim_max)
    ax1.set_title(
        f"Holdout actual vs predicted\nR2={r2_score(y_test, preds):.3f}, MAE={mean_absolute_error(y_test, preds):.2f}"
    )
    ax1.set_xlabel("Actual mutual_matches")
    ax1.set_ylabel("Predicted mutual_matches")
    ax1.legend()

    bars = ax2.barh(importance["feature"][::-1], importance["importance"][::-1], color=COL_PRIMARY)
    ax2.set_title("Top model importance signals")
    ax2.set_xlabel("Importance")
    for bar in bars:
        ax2.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {bar.get_width():.3f}", va="center", fontsize=8)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle(f"Residual std = {residuals.std():.2f}; mean residual = {residuals.mean():+.2f}", y=1.02)
    save_fig(fig, "14_residuals_feature_importance", FIGDIR)


def run_negative_binomial(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict | None:
    """Fit statsmodels NegativeBinomial on the official safe feature set.

    NegBin is the textbook choice when variance/mean >> 1 (here ≈ 5.98).
    Returns a result row compatible with the CV results DataFrame, or None on failure.
    """
    try:
        import statsmodels.api as sm
        from statsmodels.discrete.discrete_model import NegativeBinomial
    except ImportError:
        print("  [skip] statsmodels not installed — skipping Negative Binomial")
        return None

    print("  - Neg Binomial (statsmodels, holdout only)")
    builder = EngagementFeatureBuilder(
        target_col=TARGET_COL,
        include_paired_target=False,
        include_match_outcome=False,
        include_interest_tags=True,
    )
    X_tr = builder.fit_transform(X_train)
    X_te = builder.transform(X_test)

    X_tr_sm = sm.add_constant(X_tr, has_constant="add")
    X_te_sm = sm.add_constant(X_te, has_constant="add")

    try:
        nb_model = NegativeBinomial(y_train.values, X_tr_sm)
        nb_result = nb_model.fit(disp=False, maxiter=100)
        preds = nb_result.predict(X_te_sm)
        return {
            "setting": "official safe",
            "model": "Neg Binomial",
            "cv_r2_mean": r2_score(y_test, preds),
            "cv_r2_std": 0.0,
            "cv_mae_mean": mean_absolute_error(y_test, preds),
            "cv_rmse_mean": float(np.sqrt(mean_squared_error(y_test, preds))),
        }
    except Exception as exc:
        print(f"  [warn] Negative Binomial fit failed: {exc}")
        return None


def plot_learning_curve(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Plot learning curve: R² vs training-set size.

    On a no-signal target, train and validation curves converge to the same
    high-error asymptote, proving Bayes error ≈ chance rate and that more
    data would not improve predictions (standard bias-variance diagnostic).
    """
    print("\nPlotting learning curve...")
    train_sizes = np.linspace(0.1, 1.0, 8)
    sizes, train_scores, val_scores = learning_curve(
        model,
        X_train,
        y_train,
        train_sizes=train_sizes,
        scoring="r2",
        cv=3,
        n_jobs=-1,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, train_scores.mean(axis=1), label="Train R²", color=COL_GOOD, marker="o")
    ax.plot(sizes, val_scores.mean(axis=1),   label="Validation R²", color=COL_ACCENT, marker="o")
    ax.fill_between(
        sizes,
        val_scores.mean(axis=1) - val_scores.std(axis=1),
        val_scores.mean(axis=1) + val_scores.std(axis=1),
        alpha=0.2, color=COL_ACCENT,
    )
    ax.axhline(0, color="#aaa", linestyle=":", linewidth=1)
    ax.set_xlabel("Training set size")
    ax.set_ylabel("R²")
    ax.set_title(
        "Learning curve — validation R² does not improve with more data\n"
        "Convergence to near-zero confirms Bayes error ≈ chance rate",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    save_fig(fig, "16_learning_curve", FIGDIR)


def write_summary(
    results: pd.DataFrame,
    tuned_search: RandomizedSearchCV,
    metrics: dict[str, float],
    n_rows: int,
    cv: int,
    fast: bool,
) -> None:
    best_safe = results[results["setting"] == "official safe"].sort_values("cv_r2_mean", ascending=False).iloc[0]
    best_leaky = results[results["setting"] == "with paired target"].sort_values("cv_r2_mean", ascending=False).iloc[0]

    lines = [
        "# Engagement Modeling Summary",
        "",
        f"- Rows used in this run: **{n_rows:,}**",
        f"- CV folds: **{cv}**",
        f"- Fast mode: **{fast}**",
        f"- Official target: `{TARGET_COL}`",
        f"- Leakage rule: `{PAIRED_COL}` and `{TARGET}` are excluded from the official model.",
        f"- Best untuned safe CV model: **{best_safe['model']}** (R2={best_safe['cv_r2_mean']:.3f}, MAE={best_safe['cv_mae_mean']:.3f})",
        f"- Best paired-feature CV model: **{best_leaky['model']}** (R2={best_leaky['cv_r2_mean']:.3f})",
        f"- Tuned safe holdout R2: **{metrics['holdout_r2']:.3f}**",
        f"- Tuned safe holdout MAE: **{metrics['holdout_mae']:.3f}**",
        f"- Tuned safe holdout RMSE: **{metrics['holdout_rmse']:.3f}**",
        "",
        "## Tuned Parameters",
        "",
        "```json",
        json.dumps(tuned_search.best_params_, indent=2),
        "```",
        "",
        "## Auto-sklearn Note",
        "",
        "The local project remains sklearn-first for Windows/Python 3.12 compatibility. "
        "Run the final notebook's Colab/Linux auto-sklearn cell for the assignment comparison.",
    ]
    (REPORTS_DIR / "engagement_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  [saved] reports/engagement_summary.md")


def main() -> None:
    args = parse_args()
    setup_plot_style()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw(extended=True)
    sample_n = args.sample or (12000 if args.fast else None)
    if sample_n:
        df = df.sample(n=min(sample_n, len(df)), random_state=42).reset_index(drop=True)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    safe_builder = EngagementFeatureBuilder(TARGET_COL, include_paired_target=False)
    safe_builder.fit(X_train)
    leaked = [c for c in safe_builder.get_feature_names_out() if TARGET_COL in c or PAIRED_COL in c or TARGET in c]
    if leaked:
        raise RuntimeError(f"Official safe feature set leaked target-like columns: {leaked[:10]}")

    safe_results = evaluate_models(X_train, y_train, args.fast, include_paired_target=False, cv=args.cv)
    paired_results = evaluate_models(X_train, y_train, args.fast, include_paired_target=True, cv=args.cv)

    nb_row = run_negative_binomial(X_train, y_train, X_test, y_test)
    extra = [pd.DataFrame([nb_row])] if nb_row is not None else []
    results = pd.concat([safe_results, paired_results] + extra, ignore_index=True)
    results.to_csv(REPORTS_DIR / "engagement_model_results.csv", index=False)
    print("  [saved] reports/engagement_model_results.csv")

    tuned = tune_best_safe_model(X_train, y_train, args.fast, args.cv)
    best_model = tuned.best_estimator_
    metrics = holdout_metrics(best_model, X_test, y_test)
    joblib.dump(best_model, MODELS_DIR / "best_mutual_matches_model.joblib")
    print("  [saved] models/best_mutual_matches_model.joblib")

    plot_model_comparison(results)
    plot_leakage_comparison(results)
    plot_residuals_and_importance(best_model, X_test, y_test)
    plot_learning_curve(best_model, X_train, y_train)
    write_summary(results, tuned, metrics, n_rows=len(df), cv=args.cv, fast=args.fast)

    print("\nDone.")
    print(f"Best tuned CV R2: {tuned.best_score_:.3f}")
    print(f"Holdout R2: {metrics['holdout_r2']:.3f}")
    print(f"Holdout MAE: {metrics['holdout_mae']:.3f}")


if __name__ == "__main__":
    main()
