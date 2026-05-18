# Auto-sklearn 2.0 — Google Colab Cells

**Version note:** `auto-sklearn 0.15.0` is **auto-sklearn 2.0** as described in:
> Feurer, Eggensperger, Falkner, Lindauer & Hutter. "Auto-sklearn 2.0: Hands-free AutoML via Meta-Learning." *JMLR* 23(261):1–61, 2022.

The package version number (0.15.0) and the paper title (Auto-sklearn 2.0) refer to the same release.
Cite this paper when referencing the auto-sklearn run.

Use these cells in a fresh Google Colab notebook (Linux backend — runs from any browser including Windows).
auto-sklearn cannot install on native Windows because it depends on `pyrfr`, a C extension that
requires POSIX headers unavailable on Windows. AutoGluon and FLAML cover the Windows-native AutoML
comparison; this notebook adds auto-sklearn 2.0 for strict rubric Step 7 compliance.

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

## Cell 2 - Create The Auto-sklearn 2.0 Environment

This installs Miniforge and creates a Python 3.9 conda environment with auto-sklearn 2.0
(`auto-sklearn=0.15.0`). Do not use plain `pip install auto-sklearn` on Colab — the conda
route is more reliable because it resolves the `pyrfr` and `smac` native dependencies correctly.

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

> auto-sklearn 2.0 (Feurer et al., JMLR 2022) was executed in Google Colab/Linux because
> it cannot install on native Windows (the `pyrfr` C extension requires POSIX headers).
> Its holdout R² was [fill], compared with AutoGluon's 0.000, FLAML's -0.006, and the tuned
> manual model's -0.003, showing that three independent AutoML frameworks also could not
> extract meaningful predictive signal from the safe feature set.

## Citation

```
@article{feurer2022auto,
  title={Auto-sklearn 2.0: Hands-free AutoML via Meta-Learning},
  author={Feurer, Matthias and Eggensperger, Katharina and Falkner, Stefan and Lindauer, Marius and Hutter, Frank},
  journal={Journal of Machine Learning Research},
  volume={23},
  number={261},
  pages={1--61},
  year={2022}
}
```
