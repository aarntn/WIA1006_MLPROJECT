# Signal Findings — `match_outcome` Prediction

**Dataset:** 50,000 rows × 25 columns  
**Target:** `match_outcome` (10 classes, balanced)  
**Random baseline:** 0.100

## 1. Chi-square Tests (Categorical Features vs Target)

| Feature | chi² | dof | p-value | Significant? |
|---|---:|---:|---:|:---:|
| `gender` | 69.85 | 45 | 0.0102 | ✅ |
| `sexual_orientation` | 80.44 | 63 | 0.0684 | ❌ |
| `location_type` | 29.30 | 45 | 0.9662 | ❌ |
| `income_bracket` | 52.20 | 54 | 0.5442 | ❌ |
| `education_level` | 71.50 | 72 | 0.4943 | ❌ |
| `app_usage_time_label` | 30.96 | 54 | 0.9950 | ❌ |
| `swipe_right_label` | 23.01 | 27 | 0.6842 | ❌ |
| `swipe_time_of_day` | 41.22 | 45 | 0.6330 | ❌ |
| `zodiac_sign` | 107.98 | 99 | 0.2525 | ❌ |
| `body_type` | 41.87 | 45 | 0.6055 | ❌ |
| `relationship_intent` | 40.56 | 45 | 0.6604 | ❌ |

## 2. ANOVA (Numeric Features Across Outcome Groups)

| Feature | F-stat | p-value | Significant? |
|---|---:|---:|:---:|
| `app_usage_time_min` | 0.655 | 0.7506 | ❌ |
| `swipe_right_ratio` | 1.446 | 0.1621 | ❌ |
| `likes_received` | 0.777 | 0.6374 | ❌ |
| `mutual_matches` | 0.909 | 0.5156 | ❌ |
| `profile_pics_count` | 0.703 | 0.7067 | ❌ |
| `bio_length` | 0.222 | 0.9915 | ❌ |
| `message_sent_count` | 0.836 | 0.5825 | ❌ |
| `emoji_usage_rate` | 0.705 | 0.7050 | ❌ |
| `last_active_hour` | 0.830 | 0.5884 | ❌ |
| `age` | 0.570 | 0.8228 | ❌ |
| `height_cm` | 1.377 | 0.1920 | ❌ |
| `weight_kg` | 1.628 | 0.1009 | ❌ |

## 3. Model Sanity Check — 10-Class Classification

**Random baseline:** 0.100

| Model | Train Accuracy | Test Accuracy |
|---|---:|---:|
| Logistic Regression | 0.111 | 0.097 |
| Random Forest (100 trees) | 1.000 | 0.100 |
| Gradient Boosting (50 estimators) | 0.191 | 0.104 |

## 4. 3-Class Reformulation (Negative / Neutral / Positive)

**Majority-class baseline:** 0.401

| Model | Train Accuracy | Test Accuracy |
|---|---:|---:|
| Logistic Regression | 0.406 | 0.405 |
| Random Forest (100 trees) | 1.000 | 0.393 |
| Gradient Boosting (50 estimators) | 0.455 | 0.402 |

## Summary

- **1/11** categorical features show nominal significance at `p < 0.05`
- **0/12** numeric features show significant differences across outcome groups
- All tested models perform at or near baseline
- Random Forest shows a large train–test gap (`train ≈ 1.0`, `test ≈ 0.1`), which is a classic overfitting-on-noise pattern

## Conclusion

- No practically useful predictive signal for `match_outcome` was found.
- Any isolated univariate significance is weak and does not translate to generalizable predictive performance.
- Model-level evidence, especially chance-level test performance, supports rejecting `match_outcome` as a main supervised target.

## Interpretation

Although one categorical feature (`gender`) reached nominal significance, this isolated result is not enough to support meaningful predictability, especially given the broader pattern of non-significant tests and near-chance model performance. A more defensible conclusion is that **`match_outcome` is not practically learnable from the provided features**.

## Next Step

See `scripts/03_alternative_targets.py` for targets where signal *does* exist.