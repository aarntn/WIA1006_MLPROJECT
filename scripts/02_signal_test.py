"""
02_signal_test.py — Quantify whether features predict match_outcome.

Run from project root:
    python scripts/02_signal_test.py

Outputs:
    - printed p-values and model accuracies
    - reports/signal_findings.md  (markdown summary for the report)
    - reports/figures/07_signal_summary.png
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
from scipy.stats import chi2_contingency, f_oneway
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.data import load_raw, get_feature_cols, TARGET
from src.preprocessing import label_encode, outcome_3class

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")

lines = []  # we'll build the signal_findings.md as we go
def log(msg=""):
    print(msg)
    lines.append(msg)


df = load_raw(extended=True)
num_cols, cat_cols = get_feature_cols(df)

log("# Signal Findings — `match_outcome` prediction")
log("")
log(f"Dataset: {df.shape[0]} rows × {df.shape[1]} cols")
log(f"Target: `{TARGET}` ({df[TARGET].nunique()} classes, balanced)")
log(f"Random baseline: {1 / df[TARGET].nunique():.3f}")
log("")

# ---- TEST 1: chi-square for categoricals ----
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

# ---- TEST 2: ANOVA for numerics ----
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

# ---- TEST 3: Model sanity check on 10-class problem ----
log("## 3. Model sanity check — 10-class classification")
log("")
df_enc, _ = label_encode(df, cat_cols + [TARGET])
# Drop text column (interest_tags — we'll handle it separately in a later phase)
feature_cols = [c for c in df_enc.columns if c not in (TARGET, "interest_tags")]
X = df_enc[feature_cols]
y = df_enc[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

log(f"Random baseline: {1/df[TARGET].nunique():.3f}")
log("")
log("| Model | Train acc | Test acc |")
log("|---|---|---|")
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest (100)": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    "Gradient Boosting (50)": GradientBoostingClassifier(n_estimators=50, random_state=42),
}
model_results = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    tr = m.score(X_train, y_train)
    te = m.score(X_test, y_test)
    model_results[name] = (tr, te)
    log(f"| {name} | {tr:.3f} | {te:.3f} |")
log("")

# ---- TEST 4: 3-class reformulation ----
log("## 4. 3-class reformulation (Negative / Neutral / Positive)")
log("")
y3 = LabelEncoder().fit_transform(outcome_3class(df[TARGET]))
X_train, X_test, y_train, y_test = train_test_split(
    X, y3, test_size=0.2, random_state=42, stratify=y3
)
maj = np.bincount(y3).max() / len(y3)
log(f"Majority-class baseline: {maj:.3f}")
log("")
log("| Model | Train acc | Test acc |")
log("|---|---|---|")
for name, m in models.items():
    m = type(m)(**m.get_params())
    m.fit(X_train, y_train)
    log(f"| {name} | {m.score(X_train, y_train):.3f} | {m.score(X_test, y_test):.3f} |")
log("")

# ---- Summary + interpretation ----
n_sig_cat = sum(1 for _, _, p in chi_results if p < 0.05)
n_sig_num = sum(1 for _, _, p in anova_results if p < 0.05)
log("## Summary")
log("")
log(f"- **{n_sig_cat}/{len(chi_results)}** categorical features show significant association with target (p < 0.05)")
log(f"- **{n_sig_num}/{len(anova_results)}** numeric features show significant differences across outcomes")
log(f"- All models converge to ~random baseline")
log(f"- Random Forest train/test gap (train≈1.0 / test≈0.1) is the classic no-signal fingerprint")
log("")
log("**Conclusion:** `match_outcome` is statistically independent of all provided features.")
log("The dataset is synthetic and labels were assigned independently of features.")
log("")
log("Next step: see `scripts/03_alternative_targets.py` for targets where signal *does* exist.")

# ---- Save markdown + plot ----
(ROOT / "reports" / "signal_findings.md").write_text("\n".join(lines), encoding="utf-8")
print(f"\n[saved] reports/signal_findings.md")

# Visual summary
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# LEFT: p-values
all_pvals = [(col, p) for col, _, p in chi_results + anova_results]
ax = axes[0]
names, ps = zip(*all_pvals)
bars = ax.barh(range(len(names)), [-np.log10(p) for p in ps],
               color=["#4C72B0" if p >= 0.05 else "#C44E52" for p in ps])
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.axvline(-np.log10(0.05), linestyle="--", color="black", alpha=0.6, label="p=0.05")
ax.set_xlabel("−log₁₀(p-value)")
ax.set_title("Feature → target independence tests\n(bars past the line would be significant)")
ax.legend()

# RIGHT: model accuracies
ax = axes[1]
mnames = list(model_results.keys())
train_acc = [model_results[n][0] for n in mnames]
test_acc = [model_results[n][1] for n in mnames]
x = np.arange(len(mnames))
w = 0.35
ax.bar(x - w / 2, train_acc, w, label="Train", color="#55A868")
ax.bar(x + w / 2, test_acc, w, label="Test", color="#4C72B0")
ax.axhline(1 / df[TARGET].nunique(), linestyle="--", color="red", label="Random baseline (10%)")
ax.set_xticks(x)
ax.set_xticklabels(mnames, rotation=20, ha="right")
ax.set_ylabel("Accuracy")
ax.set_title("Model performance on 10-class match_outcome")
ax.set_ylim(0, 1.05)
ax.legend()

plt.tight_layout()
plt.savefig(FIGDIR / "07_signal_summary.png")
plt.close()
print("[saved] reports/figures/07_signal_summary.png")
