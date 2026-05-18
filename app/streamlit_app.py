"""
Streamlit dashboard for Swipe Atlas.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import streamlit as st

from src.data import TARGET, load_raw


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_mutual_matches_model.joblib"
SEGMENTS_PATH = ROOT / "data" / "processed" / "segmentation_assignments.csv"
ENGAGEMENT_RESULTS_PATH = ROOT / "reports" / "engagement_model_results.csv"
SEGMENT_SUMMARY_PATH = ROOT / "reports" / "segmentation_summary.csv"
SIGNAL_PATH = ROOT / "reports" / "signal_findings.md"


@st.cache_data
def load_data() -> pd.DataFrame:
    return load_raw(extended=True)


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def sidebar_input(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Profile")
    row = {}
    row["gender"] = st.sidebar.selectbox("Gender", sorted(df["gender"].unique()))
    row["sexual_orientation"] = st.sidebar.selectbox("Orientation", sorted(df["sexual_orientation"].unique()))
    row["location_type"] = st.sidebar.selectbox("Location", sorted(df["location_type"].unique()))
    row["income_bracket"] = st.sidebar.selectbox("Income", sorted(df["income_bracket"].unique()))
    row["education_level"] = st.sidebar.selectbox("Education", sorted(df["education_level"].unique()))
    row["relationship_intent"] = st.sidebar.selectbox("Intent", sorted(df["relationship_intent"].unique()))
    row["body_type"] = st.sidebar.selectbox("Body type", sorted(df["body_type"].unique()))
    row["zodiac_sign"] = st.sidebar.selectbox("Zodiac", sorted(df["zodiac_sign"].unique()))
    row["app_usage_time_label"] = st.sidebar.selectbox("Usage label", sorted(df["app_usage_time_label"].unique()))
    row["swipe_right_label"] = st.sidebar.selectbox("Swipe label", sorted(df["swipe_right_label"].unique()))
    row["swipe_time_of_day"] = st.sidebar.selectbox("Swipe time", sorted(df["swipe_time_of_day"].unique()))
    row["interest_tags"] = st.sidebar.selectbox("Interest pattern", df["interest_tags"].head(500).sort_values().unique())

    st.sidebar.header("Behavior")
    row["app_usage_time_min"] = st.sidebar.slider("Usage minutes", 0, 300, 90)
    row["swipe_right_ratio"] = st.sidebar.slider("Swipe-right ratio", 0.0, 1.0, 0.45, 0.01)
    row["profile_pics_count"] = st.sidebar.slider("Profile photos", 1, 7, 4)
    row["bio_length"] = st.sidebar.slider("Bio length", 0, 500, 180)
    row["message_sent_count"] = st.sidebar.slider("Messages sent", 0, 100, 35)
    row["emoji_usage_rate"] = st.sidebar.slider("Emoji rate", 0.0, 1.0, 0.35, 0.01)
    row["last_active_hour"] = st.sidebar.slider("Last active hour", 0, 23, 21)
    row["age"] = st.sidebar.slider("Age", 18, 59, 27)
    row["height_cm"] = st.sidebar.slider("Height cm", 145, 200, 170)
    row["weight_kg"] = st.sidebar.slider("Weight kg", 40.0, 130.0, 68.0, 0.5)

    row["likes_received"] = 0
    row["mutual_matches"] = 0
    row[TARGET] = "No Action"
    return pd.DataFrame([row])


def _show_shap_waterfall(model, sample_df: pd.DataFrame) -> None:
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        st.caption("Install `shap` to see feature attribution.")
        return

    try:
        X_feats = model.named_steps["features"].transform(sample_df)
        feature_names = list(model.named_steps["features"].get_feature_names_out())
        estimator = model.named_steps["model"]

        try:
            explainer = shap.TreeExplainer(estimator)
            sv = explainer(X_feats.values, check_additivity=False)
        except Exception:
            bg = X_feats
            explainer = shap.Explainer(estimator, bg.values)
            sv = explainer(X_feats.values)

        sv.feature_names = feature_names
        shap.plots.waterfall(sv[0], max_display=12, show=False)
        fig = plt.gcf()
        st.pyplot(fig, clear_figure=True)
    except Exception as exc:
        st.caption(f"SHAP attribution unavailable: {exc}")


def prediction_tab(df: pd.DataFrame) -> None:
    model = load_model()
    sample = sidebar_input(df)
    st.subheader("Engagement Prediction")
    if model is None:
        st.warning("Run `python scripts/04_train_engagement_models.py` to create the model artifact.")
        return

    pred = float(model.predict(sample)[0])
    st.metric("Predicted mutual matches", f"{pred:.1f}")
    st.dataframe(sample.drop(columns=[TARGET, "mutual_matches"]), use_container_width=True)

    st.subheader("Feature Attribution (SHAP)")
    st.caption("Which features pushed this prediction up or down. With R²≈0, all values land near zero — no feature dominates.")
    _show_shap_waterfall(model, sample)


def segments_tab(df: pd.DataFrame) -> None:
    st.subheader("Behavioral Segments")
    assignments = load_csv(SEGMENTS_PATH)
    summary = load_csv(SEGMENT_SUMMARY_PATH)
    if assignments is None or summary is None:
        st.warning("Run `python scripts/05_segmentation.py` to create segment artifacts.")
        return

    merged = df.reset_index(names="row_id").merge(assignments, on="row_id", how="inner")
    counts = merged["segment_name"].value_counts().rename_axis("segment").reset_index(name="users")
    st.bar_chart(counts, x="segment", y="users")
    st.dataframe(summary, use_container_width=True)


def evidence_tab() -> None:
    st.subheader("Model Evidence")
    results = load_csv(ENGAGEMENT_RESULTS_PATH)
    if results is not None:
        safe = results[results["setting"] == "official safe"].sort_values("cv_r2_mean", ascending=False)
        st.dataframe(safe, use_container_width=True)
        st.bar_chart(safe, x="model", y="cv_r2_mean")
    else:
        st.warning("Run `python scripts/04_train_engagement_models.py` to create model evidence.")

    st.subheader("Match Outcome Signal Check")
    if SIGNAL_PATH.exists():
        text = SIGNAL_PATH.read_text(encoding="utf-8", errors="replace")
        st.markdown(text)
    else:
        st.warning("Run `python scripts/02_signal_test.py` to create signal findings.")


def main() -> None:
    st.set_page_config(page_title="Swipe Atlas", layout="wide")
    st.title("Swipe Atlas")
    df = load_data()

    tab_prediction, tab_segments, tab_evidence = st.tabs(
        ["Prediction", "Segments", "Evidence"]
    )
    with tab_prediction:
        prediction_tab(df)
    with tab_segments:
        segments_tab(df)
    with tab_evidence:
        evidence_tab()


if __name__ == "__main__":
    main()
