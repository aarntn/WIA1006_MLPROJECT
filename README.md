WIA1006 / WID3006 Machine Learning — Group Assignment, Sem 2 2025/2026
Universiti Malaya, FCSIT

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

