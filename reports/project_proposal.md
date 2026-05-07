# Swipe Atlas: From Noisy Match Outcomes to Engagement-Based Dating App Insights

## Proposed Problem Statement

Online dating users often assume that visible profile and behaviour features, such as bio length, number of profile pictures, swipe ratio, message count, app usage time, and emoji usage, can reliably predict dating success. A common project idea is therefore to build a model that predicts exact outcomes such as `Ghosted`, `Mutual Match`, `Date Happened`, or `Relationship Formed`.

However, our exploratory data analysis shows that `match_outcome` is not a suitable primary target for supervised learning in this dataset. Although the dataset contains 50,000 observations and the target is evenly distributed across 10 outcome classes, the available features show little meaningful separation between the outcome groups. Statistical tests also suggest weak association between the predictors and `match_outcome`. Classification models such as Logistic Regression, Random Forest, and Gradient Boosting performed at or near the 10-class random baseline of 0.10 test accuracy. Random Forest also showed severe overfitting, achieving 1.00 training accuracy but only 0.10 test accuracy.

This means that building a model to predict exact match outcomes, including a "ghost detection" model, would likely produce unreliable and misleading results. Since `Ghosted` is part of the same low-signal `match_outcome` variable, a binary ghosting classifier would still suffer from the same limitation unless additional, more relevant behavioural data were collected.

Therefore, this project reframes the problem from direct match-outcome prediction to engagement-based analysis. Instead of forcing a weak classification task, Swipe Atlas investigates whether dating-app behaviour can provide more defensible insights through leakage-aware regression on `mutual_matches` and unsupervised segmentation of user behaviour patterns.

## Problem, Context, Impact, and Goal

**Problem:** Exact dating outcomes such as `Ghosted`, `Catfished`, or `Relationship Formed` are difficult to predict from the available dataset because they are noisy, synthetic, and weakly connected to the provided user/profile features.

**Context:** This issue occurs in dating-app analytics, where users generate many observable behaviours, such as swiping, messaging, app usage, and profile writing, but final relationship outcomes may depend on external human factors that are not captured in the dataset.

**Impact:** For users, unreliable models may lead to misleading advice about how to improve dating success. For example, users may over-invest in longer bios or more profile pictures even though the EDA shows these features do not meaningfully separate match outcomes. For developers and platform designers, using a no-signal target can produce models that overfit training data but fail to generalise, making them unsuitable for real-world decision-making.

**Goal:** The goal of Swipe Atlas is to move beyond unreliable match-outcome classification and develop a more honest engagement-focused machine learning analysis. The project compares multiple supervised models for predicting `mutual_matches`, tests for target leakage, evaluates model performance against baselines, and segments users into interpretable behavioural groups. The ideal outcome is not to force perfect prediction, but to identify which dating-app behaviours contain measurable signal and which assumptions are unsupported by the data.

## How Team Suggestions Are Incorporated

The first team idea, shifting from classification to regression, becomes the main direction of the project. Since `match_outcome` is not practically learnable, the project investigates `mutual_matches` as an engagement-based regression target.

The second idea, optimising profile effort, is included as a supporting EDA insight rather than the main prediction task. The analysis discusses whether features such as `bio_length` and `profile_pics_count` show meaningful relationships with engagement or outcomes. If no strong relationship is found, this is presented as an important finding rather than ignored.

The third idea, classifying user engagement for platform management, is adapted into unsupervised user segmentation. Instead of directly predicting `app_usage_time_label`, which may be derived from `app_usage_time_min`, the project clusters users into behavioural groups such as high-activity users, selective swipers, expressive chatters, or low-like users.

## Proposed Machine Learning Flow

1. **Project Design:** Study whether dating-app behaviour can support engagement prediction and user segmentation better than exact outcome prediction.
2. **Data Collection:** Use the provided Kaggle dating-app behaviour dataset with 50,000 records.
3. **Preprocessing:** Clean and transform the dataset using encoding, scaling, multi-hot encoding of interest tags, and engineered behavioural features.
4. **Feature Extraction:** Apply engineered features and dimensionality reduction methods such as PCA/UMAP for structure exploration and segmentation visualisation.
5. **Model Selection:** Compare at least five models, including Dummy baseline, Ridge/ElasticNet, Poisson Regression, Random Forest, Gradient Boosting, HistGradientBoosting, and XGBoost.
6. **Training and Tuning:** Train models using cross-validation and tune selected models using hyperparameter search.
7. **Evaluation:** Evaluate regression models using R2, MAE, and RMSE. Compare against baseline models and include leakage checks, especially by comparing models with and without closely related engagement features such as `likes_received`.
8. **Auto-sklearn Comparison:** Run auto-sklearn in a Colab/Linux environment and compare its performance with the manually selected models.
9. **Application:** Build a simple Streamlit dashboard to show engagement prediction, user segments, and model evidence.

## Proposed Title

**Swipe Atlas: From Noisy Match Outcomes to Engagement-Based Dating App Insights**
