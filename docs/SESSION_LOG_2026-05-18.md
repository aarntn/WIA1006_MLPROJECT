# Session Log — 2026-05-18

## What was done in this session

---

### 1. FLAML added as Windows-native AutoML backend

**Problem:** auto-sklearn is Linux-only. AutoGluon alone covered the AutoML comparison, but having
only one AutoML framework with one blocked row looked thin.

**Fix:** Added Microsoft's FLAML (`pip install flaml`) as a second Windows-native AutoML comparison.

**Files changed:**
- `scripts/06_automl_comparison.py` — added `run_flaml()` function, `--flaml-time` argument,
  `flaml` added to `--backend` choices, color/order map updated, `update_engagement_summary()`
  updated to detect FLAML results
- `requirements.txt` — added `flaml>=2.1` and `lightgbm>=4.0` (LightGBM is required by FLAML
  as its default estimator)

**FLAML result (12k sample, 120s budget):** R² = −0.006, MAE = 7.916, RMSE = 9.123

**AutoML comparison table after this session:**

| Model | Backend | Holdout R² | MAE | RMSE |
|---|---|---:|---:|---:|
| Dummy mean | manual baseline | −0.000 | 7.897 | 9.097 |
| Best manual tuned HistGB | manual tuned | −0.003 | 7.909 | 9.108 |
| FLAML AutoML | FLAML | −0.006 | 7.916 | 9.123 |
| auto-sklearn 2.0 | Linux/Colab only | — | — | — |
| AutoGluon best_quality | AutoGluon | 0.000 | 7.898 | 9.096 |

---

### 2. auto-sklearn 2.0 Colab notebook updated

**Problem:** The rubric explicitly asks for auto-sklearn. The existing Colab cells referenced
`auto-sklearn=0.15.0` without explaining this IS auto-sklearn 2.0 (the JMLR 2022 paper).

**Fix:** Updated `notebooks/AUTOSKLEARN_COLAB_CELLS.md` to:
- Explicitly state that `auto-sklearn 0.15.0` = auto-sklearn 2.0 per Feurer et al. JMLR 23(261):1–61, 2022
- Explain why it cannot install on Windows (the `pyrfr` C extension requires POSIX headers)
- Updated Cell 2 description to reference "auto-sklearn 2.0 Environment"
- Added full BibTeX citation for the paper
- Updated the "Report Sentence After It Runs" to mention FLAML and AutoGluon alongside auto-sklearn

**Action required (manual):** Open Google Colab in your browser, paste/run the cells from
`notebooks/AUTOSKLEARN_COLAB_CELLS.md`. Colab runs Linux so auto-sklearn will install and run.

---

### 3. Full 50,000-row pipeline run

**Problem:** All scripts had been running in `--fast` / 12k-sample mode. Final submission
numbers should use the complete dataset.

**Scripts run (in order):**

| Script | Mode | Key outputs |
|---|---|---|
| `scripts/01_eda.py` | full 50k | 9 figures in `reports/figures/` |
| `scripts/02_signal_test.py` | full 50k | `reports/signal_findings.md`, 3 figures |
| `scripts/04_train_engagement_models.py --cv 5` | full 50k | `reports/engagement_model_results.csv`, 4 figures, `models/best_mutual_matches_model.joblib` |
| `scripts/05_segmentation.py` | full 50k | `reports/segmentation_findings.md`, 5 figures, `models/kmeans_segmentation.joblib` |

**50k model results (official safe feature set, 5-fold CV):**

| Model | CV R² | CV MAE |
|---|---:|---:|
| Dummy mean | −0.000 | 7.908 |
| Ridge | −0.004 | 7.918 |
| ElasticNet | −0.003 | 7.917 |
| Poisson | −0.003 | 7.917 |
| Random Forest | −0.010 | 7.930 |
| Gradient Boosting | −0.002 | 7.915 |
| Hist Gradient Boosting | −0.000 | 7.910 |
| XGBoost | −0.002 | 7.915 |
| MLP | −0.028 | 7.978 |
| Neg Binomial (holdout) | −0.005 | 7.906 |

**Tuned model (holdout):** R² = **−0.001**, MAE = **7.892**, RMSE = **9.088**

**With leakage (includes `likes_received`):** XGBoost CV R² = 0.127 — confirms the leakage ceiling.

**statsmodels** was not installed; installed during this session (`pip install statsmodels`).
Negative Binomial now runs correctly in script 04.

---

### 4. Bug fixed: duplicate AutoML section in engagement_summary.md

**Problem:** `update_engagement_summary()` in script 06 called `replace_section()` twice with
slightly different heading strings. When neither matched an existing heading, both calls appended,
creating a duplicate section.

**Fix:** Replaced the two `replace_section` calls with a single regex that strips all AutoML
section variants (`AutoML Comparison Under Platform Constraints`, `AutoML Comparison`,
`AutoML Note`, `Auto-sklearn Note`) before appending the canonical section once.

---

## Final state of key numbers

| Metric | Value | Source |
|---|---|---|
| Dataset rows | 50,000 | full run |
| CV folds | 5 | |
| Tuned holdout R² (safe) | −0.001 | script 04, 50k |
| Tuned holdout MAE (safe) | 7.892 | script 04, 50k |
| Leakage upper bound R² | 0.127 | script 04, 50k |
| AutoGluon R² | 0.000 | script 06, 12k sample |
| FLAML R² | −0.006 | script 06, 12k sample |
| Segmentation k | 3 (selected by silhouette) | script 05, 50k |
| Silhouette score | < 0.05 for all k tested | script 05, 50k |

## What still needs to be done before submission

| Task | Priority |
|---|---|
| Run auto-sklearn 2.0 in Google Colab (`notebooks/AUTOSKLEARN_COLAB_CELLS.md`) | High |
| Columbia Speed Dating real-data control (optional but high-value per review PDF) | Medium |
| Write final report PDF (D2) from `reports/final_report_draft.md` | Required |
| Build Google Colab notebook (D1) — full pipeline in one notebook | Required |
| Presentation slides (D3) | Required |
| 5-minute video (D4) | Required |
