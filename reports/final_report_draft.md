# Final Report Draft

## Methodology: Falsification Criteria

Before changing the project direction, we defined falsification criteria for the original `match_outcome` prediction task. We would treat `match_outcome` as practically unlearnable if: (1) corrected chi-square/ANOVA tests showed no robust feature-target associations after multiple-testing correction, (2) supervised classifiers performed at or near the random baseline on held-out data, and (3) simpler reformulations such as the 3-class outcome did not produce meaningful improvement over the majority-class baseline. These criteria were met: after Bonferroni and Benjamini-Hochberg FDR correction, **0/23** feature-target tests remained statistically significant; 10-class classifier test accuracies stayed near the **0.100** random baseline (**0.096-0.104**); and the 3-class reformulation stayed near the **0.401** majority-class baseline (**0.393-0.405**). Therefore, the project pivoted from forcing match-outcome classification to a more honest analysis of engagement regression and user segmentation.

## Methodology: AutoML Comparison Under Platform Constraints

This project was developed on Windows. `auto-sklearn` cannot run on Windows because it depends on Python's Unix-specific `resource` module. To keep the AutoML comparison fully executable without requiring Linux, WSL, or Docker, we used two Windows-native AutoML frameworks as benchmarks:

1. **FLAML** (Fast and Lightweight AutoML, by Microsoft) — installs via `pip` on Windows, no system-level dependencies. It searched over LightGBM, XGBoost, Random Forest, and Extra Trees models within a 120-second budget.
2. **AutoGluon Tabular** (`best_quality` preset) — a second Windows-compatible AutoML framework, run with a 600-second budget.

Both used the same safe feature policy as the official supervised model: `mutual_matches` was the target, while `likes_received`, `match_outcome`, and the active target were excluded from the feature set.

| Model | Backend | Holdout R2 | MAE | RMSE |
|---|---|---:|---:|---:|
| Dummy mean | manual baseline | -0.000 | 7.897 | 9.097 |
| Best manual tuned HistGB | manual tuned | -0.003 | 7.909 | 9.108 |
| FLAML AutoML | FLAML | -0.006 | 7.916 | 9.123 |
| AutoGluon best\_quality | AutoGluon | 0.000 | 7.898 | 9.096 |
| auto-sklearn | Linux/Colab only | — | — | — |

All three executable models (manual HistGB, FLAML, and AutoGluon) produce near-zero R2 on the holdout set, supporting the conclusion that the low predictive performance is a property of the dataset and feature set rather than a failure of manual model selection. `auto-sklearn` is documented as a platform constraint and is not assigned a placeholder result.
