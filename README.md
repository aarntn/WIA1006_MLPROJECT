# Swipe Atlas
### From Noisy Match Outcomes to Engagement-Based Dating App Insights

WIA1006 / WID3006 Machine Learning — Group Assignment, Sem 2 2025/2026
Universiti Malaya, FCSIT

---

## What This Project Does

Most groups on this dataset will try to predict `match_outcome` (Ghosted, Matched, etc.).
We tested that first and found it has **no predictive signal** — every classifier converges
to the 10-class random baseline of 10% accuracy. Building a model on a no-signal target
produces misleading results.

Swipe Atlas reframes the problem honestly:

1. **EDA and data quality audit** — distributions, correlations, class balance, PCA structure, and synthetic-data fingerprints that explain why the data is hard to model.
2. **No-signal proof** — chi-square, ANOVA, and three classifiers all confirm `match_outcome` is not learnable from these features.
3. **Leakage-aware engagement regression** — predicts `mutual_matches` using 7 supervised models with an explicit leakage check (excluding `likes_received`).
4. **Hyperparameter tuning** — RandomizedSearchCV on the best regression model.
5. **User segmentation** — K-Means (k=7) groups users into behavioral archetypes, visualized with UMAP.
6. **Auto-sklearn comparison** — AutoML baseline run in Google Colab for required assignment comparison.
7. **Interactive Streamlit dashboard** — engagement prediction, segment explorer, and model evidence.

---

## Key Findings

| Finding | Evidence |
|---------|----------|
| `match_outcome` has no predictive signal | 1/11 categorical features p < 0.05 (chi-square); 0/12 numeric (ANOVA); all classifiers ≈ 10% test accuracy |
| Random Forest severely overfits | Train accuracy = 1.00, test accuracy = 0.10 |
| `mutual_matches` also has near-zero signal | Best safe model R² ≈ −0.003 (held-out) |
| `likes_received` causes leakage | Including it inflates R² from ≈0 to 0.13 |
| 7 behavioral segments identified | K-Means k=7, silhouette = 0.017 |
| Data is synthetic | Perfect class balance, near-uniform tag frequencies, zero missing values |

---

## Project Structure

```
WIA1006_MLPROJECT/
├── app/
│   └── streamlit_app.py              # Interactive dashboard (Prediction, Segments, Evidence)
├── data/
│   ├── raw/
│   │   ├── dating_app_behavior_dataset.csv              # 50,000 rows × 19 cols
│   │   └── dating_app_behavior_dataset_extended1.csv    # 50,000 rows × 25 cols (used)
│   └── processed/
│       └── segmentation_assignments.csv
├── models/
│   ├── best_mutual_matches_model.joblib
│   └── kmeans_segmentation.joblib
├── notebooks/
│   └── Swipe_Atlas_Final_Workflow.ipynb
├── reports/
│   ├── eda_findings.md
│   ├── signal_findings.md
│   ├── engagement_summary.md
│   ├── segmentation_findings.md
│   ├── engagement_model_results.csv
│   ├── segmentation_summary.csv
│   └── figures/                      # 17 publication-ready plots (Figures 01–17)
├── scripts/
│   ├── 01_eda.py                     # EDA → 9 figures + audit report
│   ├── 02_signal_test.py             # Signal test → Figure 10 + findings
│   ├── 03_alternative_targets.py     # Supplementary regression exploration
│   ├── 04_train_engagement_models.py # 7 models + leakage check + tuning → Figures 12–14
│   └── 05_segmentation.py            # K-Means + HDBSCAN + UMAP → Figures 15–17
├── src/
│   ├── data.py                       # Column definitions, load_raw()
│   ├── features.py                   # EngagementFeatureBuilder (sklearn transformer)
│   ├── plotting.py                   # Shared colors, save_fig(), setup_plot_style()
│   └── preprocessing.py              # Tag parsing, encoding utilities
├── NEXT_STEPS.md                     # Remaining tasks + submission checklist
└── requirements.txt
```

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

---

## Run the Full Pipeline

```bash
python scripts/01_eda.py
python scripts/02_signal_test.py
python scripts/04_train_engagement_models.py --cv 5
python scripts/05_segmentation.py
streamlit run app/streamlit_app.py
```

Fast smoke tests (reduced rows, for development):

```bash
python scripts/04_train_engagement_models.py --fast
python scripts/05_segmentation.py --fast
```

Or run everything via the master notebook:

```bash
jupyter notebook notebooks/Swipe_Atlas_Final_Workflow.ipynb
```

---

## Modeling Design

**Official regression target**: `mutual_matches`

**Leakage rules**:
- `match_outcome` excluded from all feature sets (no-signal classification target).
- `likes_received` excluded from the official regression model — paired engagement outcome that inflates R² from ≈0 to 0.13 when included (see Figure 13).
- A separate paired-feature model is trained to demonstrate the leakage effect explicitly.

**Models compared** (80/20 train-test split, 5-fold CV):

| Model | Type |
|-------|------|
| Dummy mean | Baseline |
| Ridge | Linear, L2 regularized |
| ElasticNet | Linear, L1+L2 |
| Poisson Regressor | Count-appropriate linear |
| Random Forest | Non-linear ensemble |
| Gradient Boosting | Boosted ensemble |
| HistGradientBoosting | Best tuned safe model |
| XGBoost | Optimized boosting (when available) |
| Auto-sklearn | AutoML baseline (Colab only) |

**Evaluation**: R², MAE, RMSE

**Tuning**: RandomizedSearchCV on HistGradientBoosting
(max_iter, learning_rate, max_leaf_nodes, min_samples_leaf, l2_regularization)

---

## Segmentation Design

- Features: all engagement features including `likes_received` (no leakage concern in unsupervised tasks)
- Preprocessing: StandardScaler
- K selection: silhouette score for k = 3–7; **k = 7 selected** (silhouette = 0.017)
- HDBSCAN: tested as alternative; produced 100% noise — rejected
- Visualization: UMAP 2D projection (falls back to PCA if umap-learn unavailable)
- Naming: automatic, based on z-score deviations per cluster

**7 segments**: Open swipers, Selective swipers, Low-emoji texters, Photo-forward users, Minimal-photo users, Low-like users, High-activity users

---

## Auto-sklearn

Cannot run on Windows / Python 3.12. Run via the Colab cell in the master notebook:

```python
!pip install auto-sklearn
from autosklearn.regression import AutoSklearnRegressor
# ... see NEXT_STEPS.md Task T1 for full code
```

Expected: R² near zero, confirming the low-signal finding is a data property not fixable by AutoML.

---

## Key Outputs

| File | Content |
|------|---------|
| `reports/eda_findings.md` | Data quality audit |
| `reports/signal_findings.md` | Statistical no-signal evidence |
| `reports/engagement_summary.md` | Best model metrics + tuned parameters |
| `reports/segmentation_findings.md` | Segment names, sizes, profiles |
| `reports/engagement_model_results.csv` | All model CV R²/MAE/RMSE |
| `reports/figures/` | 17 figures (EDA, signal, models, segments) |
| `models/best_mutual_matches_model.joblib` | Tuned sklearn pipeline |
| `models/kmeans_segmentation.joblib` | Segmentation pipeline |

All outputs are reproducible by re-running the scripts from the raw data.

---

## Data

Source: [Kaggle — Dating App Behavior Dataset](https://www.kaggle.com/datasets/keyushnisar/dating-app-behavior-dataset)

| File | Rows | Cols | Notes |
|------|------|------|-------|
| `dating_app_behavior_dataset.csv` | 50,000 | 19 | Base dataset |
| `dating_app_behavior_dataset_extended1.csv` | 50,000 | 25 | + age, height, weight, zodiac, body type, relationship intent |

The dataset is synthetic with perfect class balance and no missing values.
These properties (uniform tag frequencies, balanced targets) are synthetic fingerprints
that directly explain the near-zero signal across all models.

---

## Submission

**Deadline**: Week 13, 8 June 2026, Monday 12.00pm (SPECTRUM)

See `NEXT_STEPS.md` for the full remaining task checklist with instructions.
