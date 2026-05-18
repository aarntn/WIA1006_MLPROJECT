# Swipe Atlas — Project Progress Report
**Date**: 19 May 2026  
**Branch**: `aaron-staging`  
**Deadline**: 8 June 2026, 12:00pm (SPECTRUM)

---

## 1. Overall Status

| Area | Status |
|------|--------|
| EDA & data quality audit | ✅ Complete |
| No-signal proof (match_outcome) | ✅ Complete |
| Engagement regression (10 models) | ✅ Complete |
| Hyperparameter tuning | ✅ Complete |
| Feature selection (permutation importance) | ✅ Complete |
| SHAP explainability | ✅ Complete |
| User segmentation (K-Means, GMM, HDBSCAN, UMAP) | ✅ Complete |
| AutoML comparison (FLAML, AutoGluon, auto-sklearn) | ✅ Complete |
| Streamlit dashboard | ✅ Complete (skeleton) |
| Colab notebook (D1 deliverable) | ✅ Running end-to-end |
| Group report PDF (D2) | ❌ Not started |
| Presentation slides (D3) | ❌ Not started |
| 5-minute video (D4) | ❌ Not started |

---

## 2. Key Results

### 2.1 No-Signal Proof — `match_outcome`

- **Chi-square tests**: 0 out of 23 feature-target pairs survive Bonferroni/BH-FDR correction
- **ANOVA tests**: 0 out of 12 numeric features show significant difference across outcome classes
- **Classifier performance** (3 models, 10-class):
  - Logistic Regression: 10% (random baseline)
  - Random Forest: 10% test accuracy (train = 100% — severe overfitting)
  - Gradient Boosting: 10%
- **RF depth ablation**: test accuracy flat at 10% for all max_depth values (1–20+)
- **Conclusion**: `match_outcome` is not learnable from this feature set

### 2.2 Engagement Regression — `mutual_matches`

**Safe feature set results (5-fold CV, 12,000-row fast run):**

| Model | CV R² | CV MAE |
|-------|------:|------:|
| Dummy mean (baseline) | −0.000 | 7.936 |
| Ridge | −0.018 | 7.984 |
| ElasticNet | −0.017 | 7.983 |
| Poisson Regressor | −0.015 | 7.976 |
| Negative Binomial | −0.012 | 7.935 |
| Random Forest | −0.022 | 7.993 |
| Gradient Boosting | −0.004 | 7.948 |
| Hist Gradient Boosting | −0.012 | 7.974 |
| XGBoost | −0.004 | 7.950 |
| MLP | −0.050 | 8.074 |
| HistGB top-15 features (permutation-selected) | −0.041 | 8.054 |

**With `likes_received` included (leakage demonstration):**

| Model | CV R² |
|-------|------:|
| Gradient Boosting | 0.129 |
| XGBoost | 0.128 |
| Hist Gradient Boosting | 0.118 |
| Random Forest | 0.117 |

**Leakage finding**: Including `likes_received` inflates R² from ≈−0.001 to ≈0.127. This feature is a paired engagement outcome (not an input) and is excluded from the official model.

### 2.3 Hyperparameter Tuning

- **Method**: RandomizedSearchCV (50 iterations, 5-fold CV) on HistGradientBoostingRegressor
- **Tuned parameters**:
  - `max_iter`: 80
  - `learning_rate`: 0.03
  - `max_leaf_nodes`: 15
  - `min_samples_leaf`: 40
  - `l2_regularization`: 0.2
- **Tuned holdout result**: R² = −0.001, MAE = 7.909, RMSE = 9.108

### 2.4 Feature Selection

- **Method**: Permutation importance on tuned model's test set (n_repeats=5)
- **Selected**: Top 15 features by mean importance
- **Result**: HistGB top-15 CV R² = −0.041 (worse than full set — confirms no features carry signal)

### 2.5 AutoML Comparison

| Model | Backend | Holdout R² | MAE | RMSE | Time |
|-------|---------|----------:|----:|-----:|-----:|
| Dummy mean | Manual baseline | −0.000 | 7.897 | 9.097 | — |
| Best manual tuned HistGB | Manual tuned | −0.003 | 7.909 | 9.108 | — |
| FLAML AutoML | FLAML (Windows) | −0.006 | 7.916 | 9.123 | 120s |
| auto-sklearn 2.0 | Colab/Linux | −0.000 | 7.892 | 9.098 | 3600s |
| AutoGluon best_quality | AutoGluon (Windows) | +0.000 | 7.898 | 9.096 | 626s |

**Conclusion**: All three independent AutoML frameworks confirm R²≈0. The low signal is a dataset property, not a modelling failure.

### 2.6 User Segmentation

- **Algorithm**: K-Means with StandardScaler
- **K selection**: Silhouette sweep k=2..10; **k=3 selected** (silhouette = 0.019)
- **Rousseeuw threshold**: 0.25 ("weak structure") — our score is well below this
- **Convergent evidence of no structure**:
  - GMM BIC/AIC: monotone decrease — no preferred k
  - HDBSCAN: all 50,000 points classified as noise
  - UMAP projection: diffuse, overlapping — no tight clusters

**3 segments (50,000-row full run):**

| Segment | Users | Defining feature |
|---------|------:|-----------------|
| High-like receivers | 14,851 | Low swipe-right ratio (0.27); high likes received |
| Low-emoji texters | 8,276 | High swipe-right ratio (0.80); low emoji usage |
| Photo-forward users | 26,873 | Average swipe-right ratio (0.54); most users |

---

## 3. Figures Generated

| Figure | Script | Description |
|--------|--------|-------------|
| 01_target_distribution.png | 01_eda.py | match_outcome class balance — ratio = 1.000 |
| 02_numeric_distributions.png | 01_eda.py | Histograms for 12 numeric features |
| 03_correlation_heatmap.png | 01_eda.py | Pearson correlation matrix |
| 04_categorical_overview.png | 01_eda.py | Class imbalance across categorical features |
| 05_interest_tags_top15.png | 01_eda.py | Near-uniform tag frequencies (synthetic fingerprint) |
| 06_means_by_outcome.png | 01_eda.py | Feature means per outcome — flat lines |
| 07_pca_structure.png | 01_eda.py | PCA scatter — outcomes fully overlap |
| 08_segment_grouped_bars.png | 01_eda.py | Rule-based EDA segment engagement |
| 09_segment_boxplot_emoji_profile.png | 01_eda.py | Message volume + emoji by EDA segment |
| 10_signal_summary.png | 02_signal_test.py | p-values + classifier accuracies (all at 10%) |
| 11_confusion_matrix_3class.png | 02_signal_test.py | 3-class confusion matrix — predicts majority only |
| 11_rf_depth_ablation.png | 02_signal_test.py | RF test accuracy flat at 10% across all depths |
| 12_engagement_model_comparison.png | 04_train.py | All 10 models ranked by CV R² |
| 13_leakage_comparison.png | 04_train.py | Safe vs leaky R² side-by-side |
| 14_residuals_feature_importance.png | 04_train.py | Actual vs predicted + feature importances |
| 14b_learning_curve.png | 04_train.py | Train/val R² vs training size — flat |
| 14c_shap_beeswarm.png | 04_train.py | SHAP beeswarm — all values near zero |
| 15_kmeans_selection.png | 05_segmentation.py | Silhouette sweep k=2..10 |
| 16_segment_umap.png | 05_segmentation.py | UMAP 2D projection — diffuse clusters |
| 17_segment_profiles.png | 05_segmentation.py | Z-score heatmap per segment |
| 18_gmm_bic_aic.png | 05_segmentation.py | GMM BIC/AIC — monotone, no preferred k |
| 19_shap_cluster_beeswarm.png | 05_segmentation.py | SHAP on cluster-membership classifier |
| 20_automl_comparison.png | 06_automl.py | Manual vs FLAML vs AutoGluon vs auto-sklearn |

---

## 4. Codebase Summary

| File | Purpose |
|------|---------|
| `scripts/01_eda.py` | EDA: 9 figures, data quality audit, PCA |
| `scripts/02_signal_test.py` | Signal test: chi-square, ANOVA, 3 classifiers, Bonferroni |
| `scripts/04_train_engagement_models.py` | 10 regression models, tuning, SHAP, feature selection, learning curve |
| `scripts/05_segmentation.py` | K-Means, GMM, HDBSCAN, UMAP, SHAP cluster |
| `scripts/06_automl_comparison.py` | FLAML, AutoGluon, auto-sklearn backends |
| `src/data.py` | Column definitions, `load_raw()` |
| `src/features.py` | `EngagementFeatureBuilder` sklearn transformer |
| `src/preprocessing.py` | Tag parsing, label encoding, 3-class outcome |
| `src/plotting.py` | Shared palette, `save_fig()` |
| `app/streamlit_app.py` | Dashboard: Prediction, Segments, Evidence tabs |
| `notebooks/Swipe_Atlas_Final_Workflow.ipynb` | D1 Colab deliverable — full pipeline with inline figures |

---

## 5. Remaining Deliverables

### D1 — Colab Notebook Link
- Notebook runs end-to-end in Colab ✅
- **Action needed**: Save to Google Drive → share link → paste into slides

### D2 — Group Report PDF (4% marks)
Required sections:
1. Problem and Objective
2. Methodology and Model Explanation
3. Results and Visualization
4. Insights and Interpretation
5. Conclusion

### D3 — Presentation Slides (2% marks)
~10–12 slides for a 5-minute presentation. Key slides:
- Problem statement + the twist (no signal)
- Data and preprocessing
- Regression results + leakage check
- AutoML comparison
- Segmentation (UMAP + profiles)
- SHAP explainability
- Dashboard demo
- Conclusions

### D4 — 5-Minute Video
- Screen share + voiceover through slides
- Each group member speaks at least one slide
- Upload to YouTube (unlisted) or Google Drive
- Paste link in slides

---

## 6. Design Decisions (Do Not Change)

1. **`likes_received` excluded from regression** — leakage; inflates R² from −0.001 to 0.127
2. **`match_outcome` never used as a feature** — no-signal target; using it as a feature would be leakage
3. **Segmentation includes `likes_received`** — no leakage in unsupervised tasks (no target variable)
4. **R²≈0 is the finding, not a bug** — honest reporting; forcing high R² via leaky features is academically dishonest
5. **auto-sklearn runs in Colab only** — depends on Unix `resource` module; cannot install on Windows
