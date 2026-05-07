# Swipe Atlas

WIA1006 / WID3006 Machine Learning - Group Assignment, Sem 2 2025/2026  
Universiti Malaya, FCSIT

Swipe Atlas is an engagement prediction and behavioral segmentation project on a synthetic dating-app dataset. The final story is intentionally honest: `match_outcome` is not practically predictable from this dataset, so the scored ML core focuses on leakage-aware `mutual_matches` regression and user segmentation.

Professor-ready proposal: `reports/project_proposal.md`

## Team

- Aaron - [role]
- [teammate] - [role]

## What This Project Does

1. **EDA and data quality audit** - distributions, missing values, class balance, correlations, PCA structure, and synthetic-data fingerprints.
2. **No-signal target check** - statistical tests and classifiers show that `match_outcome` stays near random baseline.
3. **Engagement prediction** - supervised regression predicts `mutual_matches` using a leakage-safe feature set.
4. **User segmentation** - unsupervised clustering groups users into behavioral archetypes.
5. **Interactive dashboard** - Streamlit app for prediction, segment exploration, and model evidence.

## Project Structure

```text
swipe-atlas/
|-- app/
|   `-- streamlit_app.py
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|-- notebooks/
|   `-- Swipe_Atlas_Final_Workflow.ipynb
|-- reports/
|   |-- project_proposal.md
|   `-- figures/
|-- scripts/
|   |-- 01_eda.py
|   |-- 02_signal_test.py
|   |-- 03_alternative_targets.py
|   |-- 04_train_engagement_models.py
|   `-- 05_segmentation.py
|-- src/
|   |-- data.py
|   |-- features.py
|   `-- preprocessing.py
`-- requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run The Full Workflow

From the project root:

```bash
python scripts/01_eda.py
python scripts/02_signal_test.py
python scripts/04_train_engagement_models.py
python scripts/05_segmentation.py
streamlit run app/streamlit_app.py
```

Fast local smoke tests:

```bash
python scripts/04_train_engagement_models.py --fast
python scripts/05_segmentation.py --fast
```

## Modeling Design

Official target: `mutual_matches`

Leakage-safe rules:

- Exclude `match_outcome` from engagement models.
- Exclude `likes_received` from the official `mutual_matches` model because it is a paired engagement outcome.
- Keep a separate paired-feature comparison to show how leakage inflates performance.

Models compared:

- Dummy mean baseline
- Ridge
- ElasticNet
- Poisson Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- HistGradientBoosting Regressor
- XGBoost Regressor, when available

Auto-sklearn:

- Use the final notebook's Colab/Linux-only cell for the required auto-sklearn comparison.
- The local Windows/Python 3.12 repo remains sklearn-first for compatibility.

## Key Outputs

- `reports/eda_findings.md`
- `reports/project_proposal.md`
- `reports/signal_findings.md`
- `reports/engagement_model_results.csv`
- `reports/engagement_summary.md`
- `reports/segmentation_summary.csv`
- `reports/segmentation_findings.md`
- `models/best_mutual_matches_model.joblib`
- `models/kmeans_segmentation.joblib`

Generated models, processed data, and figures are ignored by git but reproducible from the scripts.

## Data

Source: [Kaggle - Dating App Behavior Dataset](https://www.kaggle.com/datasets/keyushnisar/dating-app-behavior-dataset)

- `dating_app_behavior_dataset.csv` - original 50,000 rows x 19 columns
- `dating_app_behavior_dataset_extended1.csv` - extended 50,000 rows x 25 columns

The dataset is synthetic, balanced across `match_outcome`, and has no missing values.
