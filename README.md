# Swipe Atlas

WIA1006 / WID3006 Machine Learning — Group Assignment, Sem 2 2025/2026
Universiti Malaya, FCSIT

A behavioral segmentation + engagement analysis project on a synthetic dating-app dataset. Not a match-outcome predictor — the data doesn't support that. See `reports/signal_findings.md` after running the scripts.

## Team

- Aaron — [role]
- [teammate] — [role]

## What this project does

Three pillars:

1. **Segmentation** — unsupervised clustering of users into dater archetypes
2. **Engagement prediction** — supervised regression on `mutual_matches` / `likes_received`
3. **Null-result documentation** — explicit, honest reporting that `match_outcome` is not predictable from the features in this dataset

Plus a Streamlit app (`app/`) exposing the above interactively.

## Project structure

```
swipe-atlas/
├── data/
│   ├── raw/                    # original CSVs (checked in, small enough)
│   └── processed/              # engineered features, cleaned data (gitignored)
├── notebooks/                  # for final submission .ipynb
├── scripts/
│   ├── 01_eda.py              # distributions, missing values, summaries
│   ├── 02_signal_test.py      # chi-square / ANOVA / model sanity checks
│   └── 03_alternative_targets.py  # engagement regression, where signal exists
├── src/                       # reusable modules imported by scripts
│   ├── data.py                # load / cache dataset
│   └── preprocessing.py       # encoders, feature engineering
├── app/                       # Streamlit app (built later)
├── reports/
│   ├── figures/               # saved plots
│   └── signal_findings.md     # generated summary of signal tests
├── models/                    # trained model artifacts (.pkl)
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Clone and enter
git clone <repo-url>
cd swipe-atlas

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate     # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Running

From the project root, in order:

```bash
python scripts/01_eda.py              # basic EDA, saves plots to reports/figures/
python scripts/02_signal_test.py      # statistical + model-based signal tests
python scripts/03_alternative_targets.py   # regression on mutual_matches
```

Each script is self-contained and prints a summary to stdout. Plots and artifacts go to `reports/figures/`.

## Data

Source: [Kaggle — Dating App Behavior Dataset](https://www.kaggle.com/datasets/keyushnisar/dating-app-behavior-dataset)

- `dating_app_behavior_dataset.csv` — original, 50,000 rows × 19 cols
- `dating_app_behavior_dataset_extended1.csv` — same rows, +6 cols (age, height, weight, zodiac, body type, relationship intent)

Synthetic data, balanced across classes, no missing values.

## Key findings (once you've run the scripts)

- No practically useful predictive signal for `match_outcome` was found (see `02_signal_test.py`)
- Any isolated univariate significance is weak and does not translate to generalizable predictive performance
- Model-level evidence (chance-level test performance) is the final criterion for the conclusion
- All classifiers converge to the 10% random baseline on the 10-class problem
- `likes_received` correlates 0.21 with `mutual_matches` — only non-trivial pairwise signal in the data
- Regression on `mutual_matches` from behavioral features yields R² ≈ 0.13 — modest but real

## Timeline

- **Weeks 1–2** — EDA, preprocessing, initial clustering
- **Weeks 3–4** — Finalize clustering, train supervised models, tune
- **Week 5** — Auto-sklearn comparison, report phases 1–3
- **Week 6** — Streamlit app
- **Week 7** — Slides, video, polish
- **Week 8** — Buffer + submit (deadline: June 8, 12:00pm)
