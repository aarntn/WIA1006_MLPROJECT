# Auto-sklearn Google Colab Cells

Use these cells in a fresh Google Colab notebook to run the strict auto-sklearn comparison.
This keeps the Windows project honest: AutoGluon runs locally on Windows, while auto-sklearn
runs in Colab/Linux because it is not supported on native Windows.

## Cell 1 - Clone The Project

```python
%%bash
set -e
cd /content
if [ ! -d WIA1006_MLPROJECT ]; then
  git clone -b aaron-staging https://github.com/aarntn/WIA1006_MLPROJECT.git
fi
cd WIA1006_MLPROJECT
git checkout aaron-staging
git pull
```

```python
%cd /content/WIA1006_MLPROJECT
```

## Cell 2 - Create The Auto-sklearn Environment

This installs Miniforge and creates a separate Python 3.9 environment. Do not use plain
`pip install auto-sklearn` on Colab unless this conda route fails.

```python
%%bash
set -e

if [ ! -x /content/miniforge/bin/conda ]; then
  wget -q -O /content/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /content/miniforge.sh -b -p /content/miniforge
fi

/content/miniforge/bin/conda config --set channel_priority strict

if ! /content/miniforge/bin/conda env list | grep -q "^autosklearn39 "; then
  /content/miniforge/bin/conda create -y -n autosklearn39 -c conda-forge \
    python=3.9 auto-sklearn=0.15.0 pandas numpy matplotlib seaborn joblib
fi
```

## Cell 3 - Check The Install

```python
!/content/miniforge/bin/conda run -n autosklearn39 python -c "import sys, sklearn, autosklearn; print(sys.version); print('sklearn', sklearn.__version__); print('autosklearn', autosklearn.__version__)"
```

## Cell 4 - Run Auto-sklearn

The sample size matches the existing AutoGluon run, so the comparison is fair.

```python
!/content/miniforge/bin/conda run -n autosklearn39 python scripts/06_automl_comparison.py --backend autosklearn --sample 12000 --autosklearn-time 3600
```

## Cell 5 - Inspect Results

```python
import pandas as pd

results = pd.read_csv("reports/automl_results.csv")
display(results)

leaderboard = pd.read_csv("reports/automl_leaderboard_autosklearn.csv")
display(leaderboard.head(15))
```

## Cell 6 - Download Outputs

```python
from google.colab import files

for path in [
    "reports/automl_results.csv",
    "reports/automl_leaderboard_autosklearn.csv",
    "reports/engagement_summary.md",
    "reports/figures/20_automl_comparison.png",
]:
    files.download(path)
```

## Report Sentence After It Runs

Only use this after the auto-sklearn row has real numbers:

> auto-sklearn was executed in Google Colab/Linux because it cannot run on native Windows.
> Its holdout R2 was [fill], compared with AutoGluon's 0.000 and the tuned manual model's
> -0.003, showing that AutoML also could not extract meaningful predictive signal from the
> safe feature set.
