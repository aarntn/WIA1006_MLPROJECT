# EDA Findings

## Final EDA decision table

| Finding | Evidence | Decision |
|---|---|---|
| `match_outcome` is not suitable as the **primary supervised target**. | EDA and signal-check outputs describe `match_outcome` as effectively class-balanced and not associated with stable feature signal; project notes explicitly state no practically useful predictive signal for this label. | Keep `match_outcome` for descriptive analysis/visualization only, but **reject it as the main supervised learning target**. |
| `mutual_matches` and `likes_received` should be retained for supervised tasks. | Numeric EDA highlights a meaningful positive association between `likes_received` and `mutual_matches` (about `r ≈ 0.21`), while project findings state this is the only non-trivial pairwise structure and supports engagement-oriented prediction framing. | **Retain both variables** as engagement outcomes (primary/secondary regression targets) in the EDA-to-modeling handoff. |
| Preprocessing must preserve signal while preventing leakage/noise. | Feature taxonomy separates numeric, categorical, and multi-label text-like tags; preprocessing utilities already define: multi-hot encoding for `interest_tags`, label encoding for categoricals, and explicit target/column handling (`TARGET = match_outcome`). Correlation + distribution EDA also indicate mixed scales across numeric features. | Required preprocessing (EDA-driven): **encode categoricals** (including multi-hot for `interest_tags`), **scale numeric columns** before distance/regularized models, and **drop leakage-prone fields** (active target column and non-model-ready raw multi-label text when not encoded). |

> Scope note: This table is EDA-only and intentionally excludes model tuning or hyperparameter choices.
