# Swipe Atlas — Remaining Work & Submission Checklist

**Deadline**: Week 13, 8 June 2026, Monday 12.00pm (SPECTRUM)
**Total marks**: 14% of final grade

Everything in the codebase (EDA, signal testing, 7 regression models, segmentation,
Streamlit app skeleton) is already done. What remains is submitting it correctly,
polishing what exists, and adding a few things the rubric explicitly asks for.

---

## Marks Breakdown (know what to focus on)

| Rubric Item | Marks | Status |
|-------------|-------|--------|
| Relevance and significance of problem | 1% | ⚠️ Done in code, needs to be written in report |
| Data collection and preprocessing | 2% | ⚠️ Done in code, needs documentation in report |
| Model selection and performance | 3% | ⚠️ 7 models done; auto-sklearn missing; report missing |
| Presentation quality (slides + video) | 2% | ❌ Not started |
| Creativity and innovation | 1% | ⚠️ Good angle, needs to be explicitly argued |
| Teamwork and collaboration | 1% | ⚠️ Demonstrate during Q&A |
| Report structure | 2% | ❌ Not written |
| Writing quality and integrity | 2% | ❌ Not written |
| **Total** | **14%** | |

---

## Master Checklist (ordered by priority)

### Code quality fixes (do these first — some are blocking)
- [ ] **Q1** — Add `if __name__ == "__main__": main()` guard to `scripts/02_signal_test.py` (currently runs at import time)
- [x] **Q2** — Delete `scripts/plotting.py` — dead duplicate of `src/plotting.py`; all scripts already import from `src.plotting`
- [ ] **Q3** — Delete or mark unused `engineer_ratio_features()` in `src/preprocessing.py` — duplicate of `add_engineered_features()` in `src/features.py`
- [ ] **Q4** — Add `.fillna("Neutral")` to `outcome_3class()` in `src/preprocessing.py:68` — `.map()` silently returns NaN for unmapped labels
- [ ] **Q5** — Add a feature selection step (rubric point 4): use the permutation importance already computed in `feature_importance_frame()` to drop the bottom-N features and train a "feature-selected" variant — this satisfies the rubric's explicit ask for feature selection/extraction

### Deliverables (required to submit at all)
- [ ] **D1** — Google Colab notebook with full pipeline
- [ ] **D2** — Group project report (PDF)
- [ ] **D3** — Presentation slides
- [ ] **D4** — 5-minute video (link in slides)

### Technical (affects 6% marks)
- [ ] **T1** — Auto-sklearn comparison in Colab
- [ ] **T2** — Full-dataset run on 50k rows (remove --fast)
- [ ] **T3** — SHAP explainability plot in modeling script + dashboard

### Dashboard polish (affects "application" marks under creativity)
- [ ] **A1** — Project intro banner (context for cold readers)
- [ ] **A2** — Display key figures with `st.image()`
- [ ] **A3** — Prediction disclaimer (R² ≈ 0, illustrative only)
- [ ] **A4** — Segment prediction from sidebar input
- [ ] **A5** — Better layout with `st.columns()`

### Bonus (portfolio quality + creativity mark)
- [ ] **B1** — Learning curves
- [ ] **B2** — Segment-level regression
- [ ] **B3** — 3-class confusion matrix figure

---

## What Is Already Done (do not redo)

| Component | Script / Output | Done |
|-----------|----------------|------|
| EDA (9 figures) | `scripts/01_eda.py` → `reports/eda_findings.md` | ✅ |
| Signal test | `scripts/02_signal_test.py` → `reports/signal_findings.md` | ✅ |
| 7 regression models + leakage check | `scripts/04_train_engagement_models.py` | ✅ |
| K-Means segmentation (k=7, UMAP) | `scripts/05_segmentation.py` | ✅ |
| Streamlit dashboard skeleton | `app/streamlit_app.py` | ✅ |
| Feature engineering pipeline | `src/features.py` | ✅ |
| Hyperparameter tuning (RandomizedSearchCV) | inside script 04 | ✅ |

---

## D1 — Google Colab Notebook (Required Deliverable)

**Why**: The assignment submission explicitly requires a "Google Colab/Kaggle Notebook link".
The grader will click this link — it must run end-to-end in Colab without the local repo.

**What to do**: Build one single Colab notebook that runs the full pipeline.
Upload the dataset to Colab Files or mount from Google Drive.

**Structure of the Colab notebook** (one cell per section):

```
Cell 1:  !pip install xgboost hdbscan umap-learn shap streamlit
Cell 2:  Upload / mount dataset (dating_app_behavior_dataset_extended1.csv)
Cell 3:  EDA — load data, show shape, class balance, correlation heatmap
Cell 4:  Signal test — chi-square, ANOVA, classifier accuracy table
Cell 5:  Feature engineering — show what features are built
Cell 6:  Regression models — train 7 models, show CV R2/MAE/RMSE table
Cell 7:  Leakage comparison — with vs without likes_received
Cell 8:  Hyperparameter tuning — RandomizedSearchCV on HistGradientBoosting
Cell 9:  Auto-sklearn — install + run (see T1 below for code)
Cell 10: Model comparison table (manual models vs auto-sklearn)
Cell 11: SHAP waterfall plot for best model
Cell 12: K-Means segmentation — k selection, UMAP plot, segment profiles
Cell 13: Conclusions
```

All the code for Cells 3–8 and 11–12 already exists in the local scripts — copy-paste
from `scripts/` into the notebook. Cells 9–10 are new (see T1 below).

**Share the notebook**: Set sharing to "Anyone with the link can view", then paste the
link into your presentation slides.

---

## D2 — Group Project Report (Required Deliverable, 4% marks)

**Format**: PDF. The rubric has a specific required structure — follow it exactly.

### Required sections (from rubric):

**1. Problem and Objective**
- State the original problem (predict match outcomes from dating app behaviour)
- Explain why match_outcome was found to be unlearnable (random baseline, no signal)
- State the pivot: mutual_matches regression + user segmentation
- Why it matters: unreliable models mislead users about dating success

**2. Methodology and Model Explanation**
- Dataset: 50,000 records, 25 features (extended), Kaggle source
- Preprocessing steps: encoding, scaling, multi-hot for interest tags, engineered features
  (bio_effort, night_user, emoji_heavy, engagement ratios)
- Feature engineering: describe what each engineered feature captures
- Leakage policy: explain why likes_received is excluded from official regression
- Models used (name each + why it was chosen):
  - Dummy mean (baseline)
  - Ridge / ElasticNet (linear, regularized)
  - Poisson Regressor (appropriate for count targets)
  - Random Forest (non-linear ensemble)
  - Gradient Boosting / Hist Gradient Boosting (boosted ensemble)
  - XGBoost (optimized gradient boosting)
  - Auto-sklearn (AutoML comparison, see T1)
- Segmentation: K-Means k=7 with silhouette selection, HDBSCAN comparison, UMAP visualization

**3. Results and Visualization**
- Table of all model CV R2/MAE/RMSE (copy from `reports/engagement_model_results.csv`)
- Leakage comparison chart (Figure 13)
- Signal test summary (Figure 10 — p-values + classifier accuracies)
- Segment profile heatmap (Figure 17)
- UMAP projection (Figure 16)
- SHAP waterfall for one prediction (once T3 is done)
- Auto-sklearn comparison table (once T1 is done)

**4. Insights and Interpretation**
- match_outcome has no practical predictive signal (chi-square: 1/11 features p < 0.05,
  ANOVA: 0/12 features, all classifiers ≈ 10% = random baseline)
- mutual_matches also shows near-zero signal (R² ≈ −0.003 without leakage)
- Including likes_received inflates R² to 0.13 — demonstrates leakage awareness
- 7 user segments exist but are loosely separated (silhouette = 0.017); synthetic data
  fingerprints (uniform tag frequencies, perfect class balance) explain the low signal
- Auto-sklearn confirms the finding: AutoML cannot extract signal that doesn't exist

**5. Conclusion**
- The data is synthetic and deliberately noisy — forcing classification would produce
  misleading results
- The project demonstrates that honest ML (reporting near-zero signal) is more valuable
  than a fake high-accuracy model
- Practical takeaway: dating-app success depends on factors not captured in profile/swipe data
- Limitations: synthetic data; real-world data with reply time, conversation length,
  and re-engagement signals would likely produce genuinely learnable models

---

## D3 — Presentation Slides (Required Deliverable, 2% marks)

**Format**: 5-minute video means roughly 8–12 slides. Keep each slide focused.

**Suggested slide structure**:
1. Title slide — Swipe Atlas, group members, date
2. Problem statement — what we tried to predict, why it matters
3. The twist — match_outcome has no signal (show Figure 10 directly)
4. Pivot — what we did instead (regression + segmentation)
5. Data and preprocessing — dataset overview, engineered features
6. Regression results — model comparison table + leakage check (Figures 12 & 13)
7. Auto-sklearn comparison — one slide showing our models vs AutoML
8. Segmentation — UMAP + segment profile heatmap (Figures 16 & 17)
9. SHAP explainability — one waterfall chart, interpret it
10. Dashboard demo — screenshot or screen recording of the Streamlit app
11. Conclusions and limitations
12. Q&A / Thank you

**Tips for creativity mark**: Emphasize the intellectual honesty angle —
most groups will chase a high-accuracy model; you are the group that diagnosed
why that would be wrong. That IS the innovative contribution.

---

## D4 — 5-Minute Video (Required Deliverable)

- Record a screen share + voiceover going through the slides
- Each group member should speak for at least one slide (shows teamwork)
- Upload to YouTube (unlisted) or Google Drive, paste link in the slides
- Keep it to 5 minutes — practice once before recording

---

## T1 — Auto-sklearn Comparison

**Why**: Step 7 of the ML flow in the rubric explicitly asks "How does your model compare
to auto-sklearn?" This is a required comparison, not optional.

**Runs in Google Colab only** (Linux, not Windows).

**Step 1**: In your Colab notebook, add a cell:
```python
!pip install auto-sklearn
```

**Step 2**: Run this block:
```python
import pandas as pd
from autosklearn.regression import AutoSklearnRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("dating_app_behavior_dataset_extended1.csv")

# Same safe feature set as our official model
drop_cols = ["match_outcome", "mutual_matches", "likes_received", "interest_tags"]
X = pd.get_dummies(df.drop(columns=drop_cols), drop_first=True)
y = df["mutual_matches"].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

automl = AutoSklearnRegressor(time_left_for_this_task=300, per_run_time_limit=60, seed=42)
automl.fit(X_train, y_train)

preds = automl.predict(X_test)
print(f"Auto-sklearn  R2={r2_score(y_test, preds):.3f}  MAE={mean_absolute_error(y_test, preds):.3f}")
print(automl.leaderboard())
```

**Step 3**: Add a comparison table in the Colab notebook and in the report:

| Model | CV R2 | Holdout R2 | MAE |
|-------|-------|-----------|-----|
| Dummy mean (baseline) | ≈ 0.000 | ≈ −0.003 | ≈ 7.91 |
| Best manual (Hist GB tuned) | ≈ 0.000 | ≈ −0.003 | ≈ 7.91 |
| Auto-sklearn | *(fill from output)* | *(fill)* | *(fill)* |

**Expected result**: Auto-sklearn will also produce R² near zero, confirming that
the low signal is a data property, not a modeling failure.

**Step 4**: Write 2–3 sentences interpreting the result for the report.

---

## T2 — Full-Dataset Run

All scripts ran in `--fast` mode using 3 000–15 000 rows. The final submission should
use all 50 000 rows. Run from project root with venv active:

```bash
python scripts/01_eda.py
python scripts/02_signal_test.py
python scripts/04_train_engagement_models.py --cv 5
python scripts/05_segmentation.py
```

This regenerates all 17 figures, all report `.md` files, and both model `.joblib` files.
Expected runtime: 10–20 minutes. Do this last, after all code changes are final.

---

## T3 — SHAP Explainability

`shap` is already in `requirements.txt` but never used. Showing SHAP demonstrates
understanding of model interpretability — directly relevant to the "insights and
interpretation" section of the rubric.

**In `scripts/04_train_engagement_models.py`**, add a new function after `plot_residuals_and_importance`:

```python
def plot_shap_summary(model: Pipeline, X_test: pd.DataFrame) -> None:
    import shap
    X_features = model.named_steps["features"].transform(X_test.sample(n=500, random_state=42))
    feature_names = model.named_steps["features"].get_feature_names_out()
    explainer = shap.Explainer(model.named_steps["model"], X_features)
    shap_values = explainer(X_features)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    ax.set_title("SHAP feature impact on mutual_matches prediction")
    save_fig(fig, "15_shap_beeswarm", FIGDIR)
```

Call it in `main()` after `plot_residuals_and_importance(...)`.

**In `app/streamlit_app.py`**, add a SHAP waterfall to the prediction tab so
the user can see which features drove their specific prediction score up or down.

---

## A1 — Dashboard: Project Intro Banner

Markers open the app cold. Add context above the tabs in `main()` inside `app/streamlit_app.py`:

```python
st.markdown("""
### About this project
**Swipe Atlas** analyses dating-app behaviour data to explore whether user actions
(swiping, messaging, profile setup) can predict engagement or reveal meaningful user types.

**Key finding**: `match_outcome` has no statistically detectable signal in this dataset —
every classifier we tested converged to the 10% random baseline. Rather than report a
fake result, we pivoted to predicting `mutual_matches` via regression and segmenting
users into 7 behavioural clusters using K-Means.
""")
```

---

## A2 — Dashboard: Display Key Figures

The Evidence tab only shows a table. Markers need to see the visualizations.
In `evidence_tab()` in `app/streamlit_app.py`, add after the model results table:

```python
FIGDIR = ROOT / "reports" / "figures"

st.subheader("Signal Test Results")
st.image(str(FIGDIR / "10_signal_summary.png"),
         caption="All classifiers converge to 10% random baseline for match_outcome")

st.subheader("Leakage Check")
st.image(str(FIGDIR / "13_leakage_comparison.png"),
         caption="Including likes_received inflates R² from 0 to 0.13 — this is target leakage")

st.subheader("Feature Importance")
st.image(str(FIGDIR / "14_residuals_feature_importance.png"),
         caption="No single feature dominates; residuals show no learnable structure")
```

In the Segments tab, after the segment bar chart:
```python
st.image(str(FIGDIR / "16_segment_umap.png"), caption="UMAP 2D projection of 7 segments")
st.image(str(FIGDIR / "17_segment_profiles.png"), caption="Segment z-score profiles")
```

---

## A3 — Dashboard: Prediction Disclaimer

In `prediction_tab()`, after `st.metric(...)`:

```python
st.warning(
    "This model has R² ≈ 0 on held-out data. Predictions are illustrative only "
    "and should not be used to draw conclusions about real dating behaviour. "
    "The low signal is a property of this synthetic dataset — see the Evidence tab."
)
```

---

## A4 — Dashboard: Segment Prediction from Sidebar

Show which segment the user belongs to based on their sidebar inputs.
In `prediction_tab()`, after the disclaimer:

```python
@st.cache_resource
def load_seg_model():
    path = ROOT / "models" / "kmeans_segmentation.joblib"
    return joblib.load(path) if path.exists() else None

seg_model = load_seg_model()
summary = load_csv(SEGMENT_SUMMARY_PATH)
if seg_model is not None and summary is not None:
    try:
        seg_id = int(seg_model.predict(sample)[0])
        match = summary[summary["segment_id"] == seg_id]
        if not match.empty:
            seg_name = match.iloc[0]["segment_name"]
            st.info(f"Based on your profile, you resemble the **{seg_name}** segment.")
    except Exception:
        pass
```

---

## A5 — Dashboard: Better Layout

The current layout is single-column and looks like a draft. Split the prediction tab
into two columns so it doesn't feel empty:

```python
def prediction_tab(df: pd.DataFrame) -> None:
    model = load_model()
    sample = sidebar_input(df)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Prediction")
        if model is None:
            st.warning("Run `python scripts/04_train_engagement_models.py` first.")
        else:
            pred = float(model.predict(sample)[0])
            st.metric("Predicted mutual matches", f"{pred:.1f}")
            st.warning("R² ≈ 0 — illustrative only.")

    with col_right:
        st.subheader("Your Profile")
        st.dataframe(
            sample.drop(columns=[TARGET, "mutual_matches", "likes_received"], errors="ignore"),
            use_container_width=True
        )
```

---

## B1 — Learning Curves (Bonus)

Adds a figure showing that the model's near-zero R² does not improve with more data,
proving low signal is a data property and not fixable with a larger sample.

Add to `scripts/04_train_engagement_models.py`:

```python
from sklearn.model_selection import learning_curve

def plot_learning_curve(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="r2", cv=3, n_jobs=-1
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, train_scores.mean(axis=1), label="Train R²", color=COL_GOOD)
    ax.plot(sizes, val_scores.mean(axis=1), label="Validation R²", color=COL_ACCENT)
    ax.fill_between(sizes,
                    val_scores.mean(axis=1) - val_scores.std(axis=1),
                    val_scores.mean(axis=1) + val_scores.std(axis=1),
                    alpha=0.2, color=COL_ACCENT)
    ax.axhline(0, color="#aaa", linestyle=":")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("R²")
    ax.set_title("Learning curve — R² does not improve with more data")
    ax.legend()
    save_fig(fig, "16_learning_curve", FIGDIR)
```

Call it in `main()` after tuning: `plot_learning_curve(best_model, X_train, y_train)`

---

## B2 — Segment-Level Regression (Bonus)

After segmentation, fit a separate regression model per cluster. If a segment shows
higher R² than the global ≈0, it reveals sub-population signal missed by the global model.
This demonstrates creative ML thinking and is strong report material.

Add at the bottom of `scripts/05_segmentation.py` or as `scripts/06_segment_regression.py`:

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

results = []
TARGET_COL = "mutual_matches"
for seg_id in sorted(df["segment_id"].unique()):
    subset = df[df["segment_id"] == seg_id]
    if len(subset) < 100:
        continue
    X_seg = subset.drop(columns=[TARGET_COL, "match_outcome", "likes_received", "segment_id"])
    y_seg = subset[TARGET_COL].astype(float)
    X_enc = pd.get_dummies(X_seg.drop(columns=["interest_tags"], errors="ignore"))
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(HistGradientBoostingRegressor(random_state=42),
                             X_enc, y_seg, cv=3, scoring="r2")
    results.append({"segment_id": seg_id, "cv_r2": scores.mean(), "n": len(subset)})
print(pd.DataFrame(results))
```

---

## B3 — 3-Class Confusion Matrix (Bonus)

`scripts/02_signal_test.py` already collapses match_outcome into Negative/Neutral/Positive
and trains classifiers, but never plots a confusion matrix. Adding this figure strengthens
the no-signal argument visually.

Add to `scripts/02_signal_test.py`:

```python
from sklearn.metrics import ConfusionMatrixDisplay

# after fitting the 3-class logistic regression:
cm_disp = ConfusionMatrixDisplay.from_estimator(
    clf_3class, X_test_3class, y_test_3class,
    display_labels=["Negative", "Neutral", "Positive"],
    cmap="Blues"
)
fig = cm_disp.figure_
fig.suptitle("3-class match_outcome — model predicts majority class only")
save_fig(fig, "11_confusion_matrix_3class", FIGDIR)
```

---

## How to Run Everything

```bash
# Activate venv first
.\.venv\Scripts\activate

# In order:
python scripts/01_eda.py
python scripts/02_signal_test.py
python scripts/04_train_engagement_models.py --cv 5
python scripts/05_segmentation.py

# Launch dashboard
streamlit run app/streamlit_app.py
```

Or via the master notebook:
```bash
jupyter notebook notebooks/Swipe_Atlas_Final_Workflow.ipynb
```

---

## Key Files to Know

| File | Purpose |
|------|---------|
| `src/data.py` | Column definitions, `load_raw()` |
| `src/features.py` | `EngagementFeatureBuilder` — the sklearn transformer used in all models |
| `src/plotting.py` | Shared colors, `save_fig()` — **change colors/style here to affect all 17 figures** |
| `reports/engagement_summary.md` | Last-run model metrics (R², MAE, RMSE) |
| `reports/segmentation_findings.md` | Segment names and user counts |
| `reports/signal_findings.md` | Statistical evidence that match_outcome has no signal |
| `reports/figures/` | All 17 publication-ready figures |

---

## Critical Design Decisions (do not break these)

1. **`likes_received` is excluded from regression** — including it inflates R² from 0 to 0.13
   (target leakage). Figure 13 demonstrates this. Do not add it back to the official model.

2. **`match_outcome` is never used as a feature** — it is the no-signal target we moved away
   from. Using it as a feature in a mutual_matches model would also be leakage.

3. **Segmentation intentionally includes `likes_received`** — in unsupervised tasks there is
   no target variable, so there is no leakage. This is correct and intentional.

4. **R² ≈ 0 is the finding, not a bug** — the honest conclusion of this project is that
   mutual_matches has near-zero predictable signal. Forcing a high R² by using leaky features
   would be academically dishonest and would contradict the entire project argument.
