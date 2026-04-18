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
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data import load_raw, get_feature_cols
from src.preprocessing import label_encode

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


df = load_raw(extended=True)
num_cols, cat_cols = get_feature_cols(df)

# ---- Correlation check ----
section("Correlation matrix — key engagement metrics")
engagement_cols = ["app_usage_time_min", "swipe_right_ratio", "likes_received",
                   "mutual_matches", "message_sent_count", "emoji_usage_rate",
                   "profile_pics_count", "bio_length"]
print(df[engagement_cols].corr().round(3))

section("Target candidates — how much variance do they have?")
for col in ["likes_received", "mutual_matches"]:
    print(f"{col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, "
          f"range=[{df[col].min()}, {df[col].max()}]")


# ---- Regression ----
def evaluate_target(df, target_col):
    section(f"Regression task: predict `{target_col}`")

    df_enc, _ = label_encode(df, cat_cols)

    # Don't let the target leak via correlated columns (e.g. using mutual_matches
    # to predict likes_received, since they correlate — depends on the story you want).
    drop_cols = [target_col, "interest_tags", "match_outcome"]
    X = df_enc.drop(columns=drop_cols)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale for linear models
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Linear Regression": (LinearRegression(), True),
        "Ridge": (Ridge(alpha=1.0), True),
        "Random Forest": (RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42), False),
        "Gradient Boosting": (GradientBoostingRegressor(n_estimators=100, random_state=42), False),
    }

    print(f"{'Model':<25} {'R² (test)':<12} {'MAE (test)':<12}")
    print("-" * 50)
    results = {}
    for name, (m, scale) in models.items():
        Xtr, Xte = (X_train_s, X_test_s) if scale else (X_train, X_test)
        m.fit(Xtr, y_train)
        preds = m.predict(Xte)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        results[name] = {"r2": r2, "mae": mae, "preds": preds}
        print(f"{name:<25} {r2:<12.3f} {mae:<12.3f}")

    return results, y_test


# Run for both candidates
results_matches, y_matches = evaluate_target(df, "mutual_matches")
results_likes, y_likes = evaluate_target(df, "likes_received")

# ---- Visual: predicted vs actual (best model on mutual_matches) ----
section("Plot: predicted vs actual (best model, target=mutual_matches)")
best_name = max(results_matches, key=lambda n: results_matches[n]["r2"])
best = results_matches[best_name]

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_matches, best["preds"], alpha=0.15, s=12, color="#4C72B0")
lim = [min(y_matches.min(), best["preds"].min()), max(y_matches.max(), best["preds"].max())]
ax.plot(lim, lim, "--", color="red", label="perfect prediction")
ax.set_xlabel("Actual mutual_matches")
ax.set_ylabel("Predicted mutual_matches")
ax.set_title(f"{best_name} — R²={best['r2']:.3f}")
ax.legend()
plt.tight_layout()
plt.savefig(FIGDIR / "08_regression_predicted_vs_actual.png")
plt.close()
print(f"[saved] reports/figures/08_regression_predicted_vs_actual.png")

# ---- Summary ----
section("Takeaway")
print(f"""
Best R² for mutual_matches: {max(r['r2'] for r in results_matches.values()):.3f}
Best R² for likes_received: {max(r['r2'] for r in results_likes.values()):.3f}

These are modest (~0.10-0.20), not strong, but they are NOT random — unlike
match_outcome, which genuinely is. This is the supervised target for Pillar 2
of the project.

Why this differs from match_outcome: mutual_matches and likes_received are
continuous engagement metrics derived from user activity. They show some
within-feature structure even in synthetic data (likes_received and
mutual_matches correlate ~0.21). match_outcome was sampled independently
and shows none.
""")
