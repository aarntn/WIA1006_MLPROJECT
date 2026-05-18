# Signal Findings — `match_outcome` prediction

Dataset: 50000 rows × 25 cols
Target: `match_outcome` (10 classes, balanced)
Random baseline: 0.100

## 1. Chi-square tests (categorical features vs target)

| Feature | chi² | dof | p-value | Significant? |
|---|---|---|---|---|
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

## 2. ANOVA (numeric features across outcome groups)

| Feature | F-stat | p-value | Significant? |
|---|---|---|---|
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

## 3. Model sanity check — 10-class classification

Random baseline: 0.100

| Model | Train acc | Test acc |
|---|---|---|
| Logistic Regression | 0.110 | 0.096 |
| Random Forest (100) | 1.000 | 0.100 |
| Gradient Boosting (50) | 0.191 | 0.104 |

## 4. 3-class reformulation (Negative / Neutral / Positive)

Majority-class baseline: 0.401

| Model | Train acc | Test acc |
|---|---|---|
| Logistic Regression | 0.406 | 0.405 |
| Random Forest (100) | 1.000 | 0.393 |
| Gradient Boosting (50) | 0.455 | 0.402 |

## 5. Multiple-Testing Correction

- Total tests performed: 11 chi-square + 12 ANOVA = **23 tests**
- **Bonferroni** threshold (chi-square only): α/m = 0.05/11 = 0.00455
  - gender (p=0.0102) > 0.00455 → **NOT significant after Bonferroni**
  - Result: **0/11** categorical features survive Bonferroni correction
- **Benjamini–Hochberg FDR** threshold (all 23 tests, q=0.05): rank-1 threshold = 0.05/23 ≈ 0.00217
  - Result: **0/23** features survive BH-FDR correction
- **Corrected conclusion: zero of 23 feature–target univariate tests are statistically significant.**
  This is consistent with a synthetic data generator that samples each column independently.

## Summary

- **1/11** categorical features nominally significant (p < 0.05, uncorrected)
- **0/12** numeric features nominally significant (p < 0.05, uncorrected)
- **After multiple-testing correction: 0/23 features significant**
- All models converge to ~random baseline
- Random Forest train/test gap (train≈1.0 / test≈0.1) is the classic no-signal fingerprint

**Conclusion:** No practically useful predictive signal for `match_outcome` was found.
After Bonferroni and BH-FDR correction across all 23 univariate tests, zero features
show a statistically significant association with the target. All three classifier families
converge to the uniform-prior baseline (10%), confirming an unlearnable target.
The dataset is synthetic and labels were assigned independently of features.

Next step: see `scripts/04_train_engagement_models.py` for engagement regression.