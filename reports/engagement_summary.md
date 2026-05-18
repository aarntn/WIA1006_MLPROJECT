# Engagement Modeling Summary

- Rows used in this run: **12,000**
- CV folds: **5**
- Fast mode: **True**
- Official target: `mutual_matches`
- Leakage rule: `likes_received` and `match_outcome` are excluded from the official model.
- Best untuned safe CV model: **Dummy mean** (R2=-0.000, MAE=7.936)
- Best paired-feature CV model: **Gradient Boosting** (R2=0.129)
- Tuned safe holdout R2: **-0.003**
- Tuned safe holdout MAE: **7.909**
- Tuned safe holdout RMSE: **9.108**

## Tuned Parameters

```json
{
  "model__min_samples_leaf": 40,
  "model__max_leaf_nodes": 15,
  "model__max_iter": 80,
  "model__learning_rate": 0.03,
  "model__l2_regularization": 0.2
}
```


## AutoML Comparison Under Platform Constraints

The project was developed and executed on Windows. The official auto-sklearn documentation states that auto-sklearn requires Linux and cannot run on Windows because it depends on Python's Unix-specific `resource` module. Therefore, AutoGluon is the executable local AutoML comparison, while auto-sklearn is reported as a Linux/Colab-only backend and is not assigned a placeholder score.

| Model | Backend | Status | Holdout R2 | MAE | RMSE |
|---|---|---|---:|---:|---:|
| Dummy mean | manual baseline | ok | -0.000 | 7.897 | 9.097 |
| Best manual tuned HistGB | manual tuned | ok | -0.003 | 7.909 | 9.108 |
| auto-sklearn | autosklearn | blocked: auto-sklearn is unavailable. It is Linux-only in practice and cannot run on this Windows checkout without WSL/Docker/Colab. | NA | NA | NA |
| AutoGluon best_quality | AutoGluon | ok | 0.000 | 7.898 | 9.096 |

Best observed AutoML/manual comparison row: **AutoGluon best_quality** (AutoGluon, holdout R2=0.000). All comparisons use the safe feature set with `likes_received`, `match_outcome`, and the active target excluded.

The blocked auto-sklearn row is included for methodological transparency only. It should not be described as confirming the result unless it is later run in Colab/Linux.
