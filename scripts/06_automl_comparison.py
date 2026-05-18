"""
06_automl_comparison.py - run AutoML comparisons for the engagement model.

Windows-native run (FLAML + AutoGluon, no Linux required):
    python scripts/06_automl_comparison.py --backend flaml --sample 12000
    python scripts/06_automl_comparison.py --backend autogluon --sample 12000 --autogluon-time 600

Colab/Linux run (includes auto-sklearn):
    python scripts/06_automl_comparison.py --backend all

Outputs, when the corresponding backend runs successfully:
    - reports/automl_results.csv
    - reports/automl_leaderboard_autosklearn.csv
    - reports/automl_leaderboard_autogluon.csv
    - reports/automl_leaderboard_flaml.csv
    - reports/figures/20_automl_comparison.png
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import TARGET, load_raw
from src.features import EngagementFeatureBuilder
from src.plotting import COL_ACCENT, COL_GOOD, COL_HIGHLIGHT, COL_MUTED, COL_PRIMARY, save_fig, setup_plot_style


ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
FIGDIR = REPORTS_DIR / "figures"
AUTOML_MODEL_DIR = ROOT / "models" / "automl"
TARGET_COL = "mutual_matches"
PAIRED_COL = "likes_received"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["all", "autosklearn", "autogluon", "flaml"],
        default="all",
        help="AutoML backend to run. 'flaml' works on Windows without Linux/WSL.",
    )
    parser.add_argument("--sample", type=int, default=None, help="Optional row sample for faster smoke tests.")
    parser.add_argument("--autosklearn-time", type=int, default=3600, help="auto-sklearn time budget in seconds.")
    parser.add_argument("--autogluon-time", type=int, default=3600, help="AutoGluon time budget in seconds.")
    parser.add_argument("--flaml-time", type=int, default=120, help="FLAML time budget in seconds (default 120).")
    return parser.parse_args()


def prepare_data(sample: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = load_raw(extended=True)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)

    X_raw = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(float)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=0.2,
        random_state=42,
    )

    builder = EngagementFeatureBuilder(
        target_col=TARGET_COL,
        include_paired_target=False,
        include_match_outcome=False,
        include_interest_tags=True,
    )
    X_train = builder.fit_transform(X_train_raw)
    X_test = builder.transform(X_test_raw)

    leaked = [c for c in X_train.columns if TARGET_COL in c or PAIRED_COL in c or TARGET in c]
    if leaked:
        raise RuntimeError(f"Safe AutoML feature set leaked target-like columns: {leaked[:10]}")

    return X_train, X_test, y_train, y_test


def metric_row(
    model: str,
    backend: str,
    y_true: pd.Series,
    preds,
    elapsed_seconds: float,
    time_limit_seconds: int | None,
    status: str = "ok",
) -> dict[str, object]:
    return {
        "model": model,
        "backend": backend,
        "status": status,
        "r2": r2_score(y_true, preds),
        "mae": mean_absolute_error(y_true, preds),
        "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
        "elapsed_seconds": elapsed_seconds,
        "time_limit_seconds": time_limit_seconds,
    }


def baseline_rows(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> list[dict[str, object]]:
    from sklearn.ensemble import HistGradientBoostingRegressor  # not available in auto-sklearn's pinned sklearn

    rows: list[dict[str, object]] = []

    start = time.time()
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    rows.append(metric_row("Dummy mean", "manual baseline", y_test, dummy.predict(X_test), time.time() - start, None))

    start = time.time()
    manual = Pipeline(
        [
            (
                "model",
                HistGradientBoostingRegressor(
                    min_samples_leaf=40,
                    max_leaf_nodes=15,
                    max_iter=80,
                    learning_rate=0.03,
                    l2_regularization=0.2,
                    random_state=42,
                ),
            )
        ]
    )
    manual.fit(X_train, y_train)
    rows.append(
        metric_row(
            "Best manual tuned HistGB",
            "manual tuned",
            y_test,
            manual.predict(X_test),
            time.time() - start,
            None,
        )
    )

    return rows


def run_autosklearn(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    time_limit: int,
) -> dict[str, object]:
    try:
        from autosklearn.metrics import r2
        from autosklearn.regression import AutoSklearnRegressor
    except Exception as exc:
        raise RuntimeError(
            "auto-sklearn is unavailable. It is Linux-only in practice and cannot run "
            "on this Windows checkout without WSL/Docker/Colab."
        ) from exc

    start = time.time()
    automl = AutoSklearnRegressor(
        time_left_for_this_task=time_limit,
        per_run_time_limit=max(60, min(600, time_limit // 4)),
        metric=r2,
        seed=42,
        n_jobs=-1,
    )
    automl.fit(X_train, y_train)
    preds = automl.predict(X_test)

    leaderboard = automl.leaderboard()
    leaderboard.to_csv(REPORTS_DIR / "automl_leaderboard_autosklearn.csv", index=False)

    return metric_row("auto-sklearn", "auto-sklearn", y_test, preds, time.time() - start, time_limit)


def run_autogluon(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    time_limit: int,
) -> dict[str, object]:
    try:
        from autogluon.tabular import TabularDataset, TabularPredictor
    except Exception as exc:
        raise RuntimeError("AutoGluon is unavailable. Install with `pip install autogluon.tabular[all]`.") from exc

    start = time.time()
    train_data = TabularDataset(X_train.assign(**{TARGET_COL: y_train.to_numpy()}))
    test_data = TabularDataset(X_test.assign(**{TARGET_COL: y_test.to_numpy()}))

    model_path = AUTOML_MODEL_DIR / "autogluon"
    if model_path.exists():
        shutil.rmtree(model_path)

    predictor = TabularPredictor(
        label=TARGET_COL,
        eval_metric="r2",
        problem_type="regression",
        path=str(model_path),
        verbosity=2,
    ).fit(
        train_data,
        presets="best_quality",
        time_limit=time_limit,
    )
    preds = predictor.predict(X_test)

    leaderboard = predictor.leaderboard(test_data, silent=True)
    leaderboard.to_csv(REPORTS_DIR / "automl_leaderboard_autogluon.csv", index=False)

    return metric_row("AutoGluon best_quality", "AutoGluon", y_test, preds, time.time() - start, time_limit)


def run_flaml(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    time_limit: int,
) -> dict[str, object]:
    try:
        from flaml import AutoML
    except Exception as exc:
        raise RuntimeError("FLAML is unavailable. Install with `pip install flaml`.") from exc

    start = time.time()
    automl = AutoML()
    automl.fit(
        X_train,
        y_train,
        task="regression",
        time_budget=time_limit,
        metric="r2",
        seed=42,
        verbose=1,
    )
    preds = automl.predict(X_test)

    leaderboard_rows = []
    for estimator, cfg in (automl.best_config_per_estimator or {}).items():
        leaderboard_rows.append({"estimator": estimator, **cfg})
    pd.DataFrame(leaderboard_rows).to_csv(REPORTS_DIR / "automl_leaderboard_flaml.csv", index=False)

    return metric_row("FLAML AutoML", "FLAML", y_test, preds, time.time() - start, time_limit)


def plot_automl_comparison(results: pd.DataFrame) -> None:
    ok = results[results["status"] == "ok"].copy()
    ok = ok.sort_values("r2")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = [
        COL_MUTED if "manual" in backend
        else COL_PRIMARY if backend == "AutoGluon"
        else COL_ACCENT if backend == "FLAML"
        else COL_HIGHLIGHT
        for backend in ok["backend"]
    ]
    bars = ax.barh(ok["model"], ok["r2"], color=colors, edgecolor="white")
    ax.axvline(0, color="#666", linestyle=":", linewidth=1)
    ax.set_xlabel("Holdout R2")
    ax.set_title("AutoML comparison on safe mutual_matches feature set")
    for bar, val in zip(bars, ok["r2"]):
        x = val + 0.003 if val >= 0 else val - 0.003
        ha = "left" if val >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}", va="center", ha=ha, fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    save_fig(fig, "20_automl_comparison", FIGDIR)


def replace_section(text: str, heading: str, replacement: str) -> str:
    pattern = rf"(^## {re.escape(heading)}\n)(.*?)(?=^## |\Z)"
    if re.search(pattern, text, flags=re.M | re.S):
        return re.sub(pattern, replacement + "\n", text, flags=re.M | re.S)
    return text.rstrip() + "\n\n" + replacement + "\n"


def update_engagement_summary(results: pd.DataFrame) -> None:
    path = REPORTS_DIR / "engagement_summary.md"
    text = path.read_text(encoding="utf-8") if path.exists() else "# Engagement Modeling Summary\n"

    ok = results[results["status"] == "ok"].sort_values("r2", ascending=False)
    blocked = results[results["status"] != "ok"]
    autosklearn = results[results["model"] == "auto-sklearn"]
    autosklearn_ok = not autosklearn.empty and (autosklearn["status"] == "ok").any()
    flaml_ok = not results[results["model"] == "FLAML AutoML"].empty and (
        results[results["model"] == "FLAML AutoML"]["status"] == "ok"
    ).any()

    lines = [
        "## AutoML Comparison Under Platform Constraints",
        "",
    ]
    if autosklearn_ok and flaml_ok:
        lines.extend(
            [
                "The project was developed on Windows. FLAML is a Windows-native AutoML framework that does not "
                "depend on Unix-specific system resources and was executed locally. auto-sklearn was additionally "
                "executed in Google Colab/Linux. AutoGluon Tabular was also used as a second Windows-compatible benchmark.",
                "",
            ]
        )
    elif flaml_ok:
        lines.extend(
            [
                "The project was developed and executed on Windows. FLAML (Fast and Lightweight AutoML, by Microsoft) "
                "is a Windows-native AutoML framework that does not require Linux or WSL. AutoGluon Tabular is a second "
                "Windows-compatible AutoML benchmark. auto-sklearn is Linux-only and cannot run on Windows because it "
                "depends on Python's Unix-specific `resource` module; it is documented as a platform constraint without "
                "a placeholder score.",
                "",
            ]
        )
    elif autosklearn_ok:
        lines.extend(
            [
                "The project was developed on Windows, where auto-sklearn cannot run because it depends on "
                "Python's Unix-specific `resource` module. To satisfy the strict AutoML comparison fairly, "
                "auto-sklearn was executed in Google Colab/Linux, while AutoGluon remained the executable "
                "Windows-compatible AutoML benchmark.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "The project was developed and executed on Windows. The official auto-sklearn documentation states that "
                "auto-sklearn requires Linux and cannot run on Windows because it depends on Python's Unix-specific "
                "`resource` module. Therefore, AutoGluon is the executable local AutoML comparison, while auto-sklearn "
                "is reported as a Linux/Colab-only backend and is not assigned a placeholder score.",
                "",
            ]
        )
    lines.extend(
        [
            "| Model | Backend | Status | Holdout R2 | MAE | RMSE |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for _, row in results.iterrows():
        if row["status"] == "ok":
            lines.append(
                f"| {row['model']} | {row['backend']} | ok | {row['r2']:.3f} | {row['mae']:.3f} | {row['rmse']:.3f} |"
            )
        else:
            lines.append(f"| {row['model']} | {row['backend']} | {row['status']} | NA | NA | NA |")

    lines.append("")
    if not ok.empty:
        best = ok.iloc[0]
        lines.append(
            f"Best observed AutoML/manual comparison row: **{best['model']}** "
            f"({best['backend']}, holdout R2={best['r2']:.3f}). "
            "All comparisons use the safe feature set with `likes_received`, `match_outcome`, "
            "and the active target excluded."
        )
    if not blocked.empty:
        lines.append("")
        lines.append(
            "The blocked auto-sklearn row is included for methodological transparency only. It should not be "
            "described as confirming the result unless it is later run in Colab/Linux."
        )

    # Remove all legacy AutoML section variants, then append the canonical one.
    import re as _re
    _automl_pat = _re.compile(
        r"^## (?:AutoML Comparison.*?|AutoML Note|Auto-sklearn Note)\n.*?(?=^## |\Z)",
        _re.M | _re.S,
    )
    text = _automl_pat.sub("", text).rstrip() + "\n\n" + "\n".join(lines) + "\n"
    path.write_text(text.strip() + "\n", encoding="utf-8")


def merge_existing_results(results: list[dict[str, object]]) -> pd.DataFrame:
    """Preserve previously executed AutoML rows when running one backend at a time."""
    current = pd.DataFrame(results)
    path = REPORTS_DIR / "automl_results.csv"
    if not path.exists():
        return current

    existing = pd.read_csv(path)
    existing = existing[~existing["model"].isin(current["model"])]
    combined = pd.concat([current, existing], ignore_index=True)
    order = {
        "Dummy mean": 0,
        "Best manual tuned HistGB": 1,
        "FLAML AutoML": 2,
        "auto-sklearn": 3,
        "AutoGluon best_quality": 4,
    }
    combined["_order"] = combined["model"].map(order).fillna(99)
    return combined.sort_values(["_order", "model"]).drop(columns="_order").reset_index(drop=True)


def write_outputs(results: list[dict[str, object]]) -> pd.DataFrame:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    results_df = merge_existing_results(results)
    results_df.to_csv(REPORTS_DIR / "automl_results.csv", index=False)
    plot_automl_comparison(results_df)
    update_engagement_summary(results_df)
    print("  [saved] reports/automl_results.csv")
    print("  [saved] reports/figures/20_automl_comparison.png")
    print("  [updated] reports/engagement_summary.md")
    return results_df


def main() -> None:
    args = parse_args()
    setup_plot_style()
    X_train, X_test, y_train, y_test = prepare_data(args.sample)

    rows = baseline_rows(X_train, X_test, y_train, y_test)
    backends = ["autosklearn", "autogluon", "flaml"] if args.backend == "all" else [args.backend]

    _model_name = {
        "autosklearn": "auto-sklearn",
        "autogluon": "AutoGluon best_quality",
        "flaml": "FLAML AutoML",
    }
    _time_limit = {
        "autosklearn": args.autosklearn_time,
        "autogluon": args.autogluon_time,
        "flaml": args.flaml_time,
    }

    for backend in backends:
        try:
            if backend == "autosklearn":
                rows.append(run_autosklearn(X_train, X_test, y_train, y_test, args.autosklearn_time))
            elif backend == "autogluon":
                rows.append(run_autogluon(X_train, X_test, y_train, y_test, args.autogluon_time))
            elif backend == "flaml":
                rows.append(run_flaml(X_train, X_test, y_train, y_test, args.flaml_time))
        except Exception as exc:
            print(f"  [blocked] {backend}: {exc}")
            rows.append(
                {
                    "model": _model_name.get(backend, backend),
                    "backend": backend,
                    "status": f"blocked: {exc}",
                    "r2": np.nan,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "elapsed_seconds": 0.0,
                    "time_limit_seconds": _time_limit.get(backend, 0),
                }
            )

    results_df = write_outputs(rows)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
