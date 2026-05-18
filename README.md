# Swipe Atlas
### From Noisy Match Outcomes to Engagement-Based Dating App Insights

WIA1006 / WID3006 Machine Learning — Group Assignment, Sem 2 2025/2026
Universiti Malaya, FCSIT

**Deadline**: 8 June 2026 (Monday 12:00pm, SPECTRUM)

---

## What This Project Does

Most groups on this dataset will try to predict `match_outcome` (Ghosted, Matched, etc.).
We tested that first and found it has **no predictive signal**: every classifier converges
to the 10-class random baseline of 10% accuracy. Building a model on a no-signal target
produces misleading results.

Swipe Atlas reframes the problem honestly:

1. **EDA and data quality audit** — distributions, correlations, class balance, PCA structure, and synthetic-data fingerprints that explain why the data is hard to model.
2. **No-signal proof** — Bonferroni/BH-FDR-corrected chi-square + ANOVA across 23 feature-target pairs, three classifiers all at chance, RF max-depth ablation, and learning curves all confirm `match_outcome` is not learnable.
3. **Leakage-aware engagement regression** — predicts `mutual_matches` using 9 supervised models (including MLP and Negative Binomial) with an explicit leakage check excluding `likes_received`.
4. **Hyperparameter tuning** — RandomizedSearchCV on the best regression model.
5. **User segmentation** — K-Means with silhouette-vs-k sweep (k=2..10), GMM BIC/AIC, HDBSCAN, and UMAP visualization. All algorithms confirm no density structure (silhouette < 0.05 for all k).
6. **AutoML comparison** — FLAML and AutoGluon run locally on Windows; auto-sklearn documented as Linux/Colab-only (see `notebooks/AUTOSKLEARN_COLAB_CELLS.md`).
7. **Interactive Streamlit dashboard** — engagement prediction, segment explorer, and model evidence.

---

## Key Findings (50,000-row full run)

| Finding | Evidence |
|---------|----------|
| `match_outcome` has no predictive signal | 0/23 feature-target tests survive Bonferroni/BH-FDR correction; all classifiers at 10% = random baseline |
| Random Forest severely overfits | Train accuracy = 1.00, test accuracy = 0.10; flat across all max_depth values |
| `mutual_matches` also has near-zero signal | Tuned model holdout R² = −0.001 (safe feature set) |
| `likes_received` causes leakage | Including it inflates R² from −0.001 to 0.127 |
| AutoML confirms the finding | FLAML R² = −0.006, AutoGluon R² = 0.000 — both at baseline |
| No cluster structure detected | K-Means k=3 selected (best silhouette = 0.019, well below 0.25 "weak" threshold); GMM BIC monotone; HDBSCAN all-noise |
| Data is synthetic | Perfect class balance, near-uniform tag frequencies (~2% each across 49 tags), zero missing values |

---

## Project Structure

```text
WIA1006_MLPROJECT/
├── app/
│   └── streamlit_app.py              # Interactive dashboard
├── data/
│   ├── raw/
│   │   ├── dating_app_behavior_dataset.csv              # 50,000 rows × 19 cols
│   │   └── dating_app_behavior_dataset_extended1.csv    # 50,000 rows × 25 cols
│   └── processed/
│       └── segmentation_assignments.csv
├── docs/                             # Reference documents (not script outputs)
│   ├── WIA1006_WID3006_Group_Assignment_2526.pdf
│   ├── ML_GROUP2_DOCUMENTATIONS.pdf
│   ├── NEXT_STEPS.md
│   ├── SESSION_LOG_2026-05-18.md
│   ├── CRITIQUE_CHANGES.md
│   ├── project_proposal.md
│   └── professor_proposal_message.md
├── models/
│   ├── best_mutual_matches_model.joblib
│   └── kmeans_segmentation.joblib
├── notebooks/
│   ├── AUTOSKLEARN_COLAB_CELLS.md    # Run auto-sklearn 2.0 in Google Colab
│   ├── AutoSklearn_Colab_Run.ipynb
│   └── Swipe_Atlas_Final_Workflow.ipynb
├── reports/
│   ├── eda_findings.md
│   ├── signal_findings.md
│   ├── engagement_summary.md         # Best model metrics + AutoML comparison
│   ├── final_report_draft.md
│   ├── segmentation_findings.md
│   ├── automl_results.csv
│   ├── automl_leaderboard_autogluon.csv
│   ├── automl_leaderboard_flaml.csv
│   ├── engagement_model_results.csv
│   ├── segmentation_summary.csv
│   ├── segmentation_k_selection.csv
│   ├── eda_quality_audit.csv
│   └── figures/                      # 20 numbered publication-ready plots
│       ├── 01_target_distribution.png
│       ├── 02_numeric_distributions.png
│       ├── 03_correlation_heatmap.png
│       ├── 04_categorical_overview.png
│       ├── 05_interest_tags_top15.png
│       ├── 06_means_by_outcome.png
│       ├── 07_pca_structure.png
│       ├── 08_segment_grouped_bars.png
│       ├── 09_segment_boxplot_emoji_profile.png
│       ├── 10_signal_summary.png
│       ├── 11_confusion_matrix_3class.png
│       ├── 11_rf_depth_ablation.png
│       ├── 12_engagement_model_comparison.png
│       ├── 13_leakage_comparison.png
│       ├── 14_residuals_feature_importance.png
│       ├── 14b_learning_curve.png
│       ├── 14c_shap_beeswarm.png
│       ├── 15_kmeans_selection.png
│       ├── 16_segment_umap.png
│       ├── 17_segment_profiles.png
│       ├── 18_gmm_bic_aic.png
│       ├── 19_shap_cluster_beeswarm.png
│       └── 20_automl_comparison.png
├── scripts/
│   ├── 01_eda.py
│   ├── 02_signal_test.py
│   ├── 04_train_engagement_models.py
│   ├── 05_segmentation.py
│   └── 06_automl_comparison.py
├── src/
│   ├── data.py
│   ├── features.py
│   ├── plotting.py
│   └── preprocessing.py
├── requirements.txt
└── README.md
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
python scripts/06_automl_comparison.py --backend flaml --sample 12000 --flaml-time 120
python scripts/06_automl_comparison.py --backend autogluon --sample 12000 --autogluon-time 600
streamlit run app/streamlit_app.py
```

Fast smoke tests (reduced rows, for development):

```bash
python scripts/04_train_engagement_models.py --fast
python scripts/05_segmentation.py --fast
```

---

## Modeling Design

**Official regression target**: `mutual_matches`

**Leakage rules**:
- `match_outcome` excluded from all feature sets (no-signal classification target).
- `likes_received` excluded from the official regression model — it is a paired engagement outcome that inflates R² from −0.001 to 0.127 when included (see Figure 13).
- A separate paired-feature model is trained explicitly to demonstrate the leakage effect.

**Models compared** (50,000 rows, 80/20 train-test split, 5-fold CV):

| Model | Type | CV R² |
|-------|------|-------|
| Dummy mean | Baseline | −0.000 |
| Ridge | Linear, L2 regularized | −0.004 |
| ElasticNet | Linear, L1+L2 | −0.003 |
| Poisson Regressor | Count-appropriate linear | −0.003 |
| Negative Binomial | Count, overdispersion-aware | −0.005 |
| Random Forest | Non-linear ensemble | −0.010 |
| Gradient Boosting | Boosted ensemble | −0.002 |
| HistGradientBoosting | Best tuned model | −0.000 |
| XGBoost | Optimized boosting | −0.002 |
| MLP | Neural network (defensive check) | −0.028 |
| FLAML AutoML | Windows-native AutoML | −0.006 |
| AutoGluon Tabular | Windows-native AutoML | 0.000 |
| auto-sklearn 2.0 | Linux/Colab only — see `notebooks/AUTOSKLEARN_COLAB_CELLS.md` | — |

**Tuning**: RandomizedSearchCV on HistGradientBoosting
(max_iter, learning_rate, max_leaf_nodes, min_samples_leaf, l2_regularization)

**Tuned holdout result**: R² = −0.001, MAE = 7.892, RMSE = 9.088

---

## Segmentation Design

- Features: all engagement features including `likes_received` (no leakage concern in unsupervised tasks)
- Preprocessing: StandardScaler
- K selection: silhouette score sweep k=2..10; **k=3 selected** (silhouette = 0.019 — below Rousseeuw 1987 "weak" threshold of 0.25)
- GMM BIC/AIC: monotone decrease, no preferred k — convergent evidence of no structure
- HDBSCAN: all points classified as noise — convergent evidence of no structure
- Visualization: UMAP 2D projection
- SHAP: trained cluster-membership classifier to explain what features define each segment geometrically

**3 segments (50k run)**: High-like receivers (14,851 users), Low-emoji texters (8,276), Photo-forward users (26,873)

---

## AutoML Under Platform Constraints

This project was developed on Windows. `auto-sklearn` requires Linux (depends on Python's Unix-specific `resource` module) and cannot install on Windows. The Windows-native AutoML comparisons are:

- **FLAML** (Microsoft) — `pip install flaml`, no Linux required. R² = −0.006.
- **AutoGluon Tabular** — `pip install autogluon.tabular`. R² = 0.000.

To run **auto-sklearn 2.0** (Feurer et al., JMLR 2022) — open `notebooks/AUTOSKLEARN_COLAB_CELLS.md` and run the cells in Google Colab (Linux backend, works from any browser).

All three executable AutoML systems land at R² ≈ 0, matching the manual tuned model. This confirms the low signal is a dataset property, not a modelling failure.

---

## Key Output Files

| File | Content |
|------|---------|
| `reports/eda_findings.md` | Data quality audit and EDA summary |
| `reports/signal_findings.md` | Statistical no-signal evidence with Bonferroni/BH-FDR |
| `reports/engagement_summary.md` | Best model metrics, tuned parameters, AutoML comparison |
| `reports/final_report_draft.md` | Report-ready methodology sections |
| `reports/engagement_model_results.csv` | All model CV R²/MAE/RMSE |
| `reports/automl_results.csv` | AutoML comparison table |
| `reports/segmentation_findings.md` | Segment names, sizes, profiles |
| `reports/figures/` | 22 numbered publication-ready plots |
| `models/best_mutual_matches_model.joblib` | Tuned sklearn pipeline |
| `models/kmeans_segmentation.joblib` | Segmentation pipeline |
| `docs/NEXT_STEPS.md` | Submission checklist and remaining tasks |

---

## Data

Source: [Kaggle — Dating App Behavior Dataset](https://www.kaggle.com/datasets/keyushnisar/dating-app-behavior-dataset)

| File | Rows | Cols | Notes |
|------|------|------|-------|
| `dating_app_behavior_dataset.csv` | 50,000 | 19 | Base dataset |
| `dating_app_behavior_dataset_extended1.csv` | 50,000 | 25 | Extended with age, height, weight, zodiac, body type, relationship intent |

The dataset is synthetic: perfect class balance, near-uniform interest-tag frequencies (~2% each across 49 tags), and zero missing values. These properties are the standard signature of programmatically generated tabular data and directly explain the near-zero predictive signal across all models and AutoML systems.
