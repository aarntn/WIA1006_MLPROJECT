# Professor Proposal Message

Dear Prof.,

Our group would like to propose the project **Swipe Atlas: From Noisy Match Outcomes to Engagement-Based Dating App Insights**.

Initially, we considered predicting exact dating outcomes such as `Ghosted`, `Mutual Match`, `Date Happened`, or `Relationship Formed`. However, our exploratory analysis showed that `match_outcome` is not a reliable primary target. The 10 outcome classes are almost evenly balanced, but the available predictors show very weak separation across these classes. Statistical tests found little association between the features and `match_outcome`, and classification models such as Logistic Regression, Random Forest, and Gradient Boosting performed near the 10-class random baseline of 0.10 test accuracy. Random Forest also overfit severely, with 1.00 training accuracy but only around 0.10 test accuracy.

Because of this, we believe that building a direct match-outcome predictor, including a ghosting detector, would produce unreliable and potentially misleading results. Instead, we propose to reframe the project as an engagement-focused machine learning analysis. Our main supervised task will investigate `mutual_matches` using regression models with leakage checks, while our unsupervised task will segment users into interpretable behavioural groups.

The project will still follow the required machine learning flow: data preprocessing, feature engineering, feature extraction/visualisation, comparison of at least five models, hyperparameter tuning, evaluation using suitable metrics, auto-sklearn comparison in Colab/Linux, and a simple Streamlit dashboard.

This direction allows us to show both technical modelling and critical thinking: rather than forcing a low-signal target, we will identify which dating-app behaviours contain measurable signal and which assumptions are not supported by the data.

Thank you.
