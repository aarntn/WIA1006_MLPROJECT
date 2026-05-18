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

## Auto-sklearn Note

The local project remains sklearn-first for Windows/Python 3.12 compatibility. Run the final notebook's Colab/Linux auto-sklearn cell for the assignment comparison.
