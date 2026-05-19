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
AUTOML_RESULTS_PATH = ROOT / "reports" / "automl_results.csv"
SEGMENT_SUMMARY_PATH = ROOT / "reports" / "segmentation_summary.csv"
K_SELECTION_PATH = ROOT / "reports" / "segmentation_k_selection.csv"
SEGMENTATION_FINDINGS_PATH = ROOT / "reports" / "segmentation_findings.md"
SIGNAL_PATH = ROOT / "reports" / "signal_findings.md"


@st.cache_data
def load_data() -> pd.DataFrame:
    return load_raw(extended=True)


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception as exc:
            st.error(f"Could not load model artifact: {exc}")
    return None


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def format_number(value: float | int, decimals: int = 3) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:,.{decimals}f}"


def load_text(path: Path) -> str | None:
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    return None


def best_safe_cv_row() -> pd.Series | None:
    results = load_csv(ENGAGEMENT_RESULTS_PATH)
    if results is None:
        return None
    safe = results[results["setting"] == "official safe"].sort_values("cv_r2_mean", ascending=False)
    return safe.iloc[0] if not safe.empty else None


def tuned_holdout_row() -> pd.Series | None:
    automl = load_csv(AUTOML_RESULTS_PATH)
    if automl is None:
        return None
    row = automl[automl["model"] == "Best manual tuned HistGB"]
    return row.iloc[0] if not row.empty else None


def apply_inter_tight_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #0F172A;
            --surface: #111827;
            --surface-2: #1F2937;
            --border: #334155;
            --text-main: #F8FAFC;
            --text-muted: #CBD5E1;
            --text-soft: #94A3B8;
            --accent: #38BDF8;
            --danger: #F87171;
        }

        html, body, main, section, p, label, small, strong, em,
        h1, h2, h3, h4, h5, h6, button, input, textarea, select, option,
        svg text, canvas,
        .stApp, .stMarkdown, .stMetric, .stButton, .stSelectbox,
        .stSlider, .stRadio, .stTable, .stDataFrame,
        [role="option"], [role="listbox"], [role="menu"],
        [data-testid="stSidebar"], [data-testid="stHeader"],
        [data-testid="stWidgetLabel"], [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"], [data-testid="stCaptionContainer"],
        [data-testid="stMarkdownContainer"] {
            font-family: 'Inter Tight', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }


        .stApp {
            background: var(--bg);
            color: var(--text-main);
        }

        [data-testid="stHeader"] {
            background: rgba(15, 23, 42, 0.9);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.4rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: var(--surface);
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] > div {
            background: var(--surface);
        }

        [data-testid="stSidebar"] h1 {
            font-size: 1.45rem;
            font-weight: 800;
            letter-spacing: 0;
            margin-bottom: 0.15rem;
            color: var(--text-main);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--text-soft);
            line-height: 1.45;
        }

        [data-testid="stSidebar"] [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] {
            color: var(--text-soft);
            font-weight: 600;
        }

        [data-testid="stMetricValue"] {
            color: #E2E8F0;
            font-size: 2.15rem;
            font-weight: 700;
            letter-spacing: 0;
        }

        [data-testid="stSidebar"] [data-testid="stMetricValue"] {
            font-size: 1.95rem;
        }

        [data-testid="stSidebar"] hr {
            border-color: var(--border);
        }

        div[role="radiogroup"] label {
            border-radius: 8px;
            padding: 0.45rem 0.6rem;
            margin-bottom: 0.15rem;
            color: var(--text-muted);
        }

        div[role="radiogroup"] label:hover {
            background: var(--surface-2);
        }

        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--text-main);
        }

        h1 {
            font-weight: 800;
            font-size: 2.45rem;
        }

        h2, h3 {
            font-weight: 700;
        }

        p, li, label, [data-testid="stMarkdownContainer"] {
            color: var(--text-muted);
        }

        .stCaptionContainer, [data-testid="stCaptionContainer"] {
            color: var(--text-soft);
        }

        .exec-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 0.75rem 0 1.35rem;
        }

        .exec-card {
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent);
            border-radius: 8px;
            background: var(--surface);
            padding: 1rem 1.05rem;
            min-height: 122px;
        }

        .exec-card span {
            display: block;
            color: var(--text-soft);
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.35rem;
        }

        .exec-card strong {
            display: block;
            color: var(--text-main);
            font-size: 1.1rem;
            font-weight: 750;
            line-height: 1.15;
            margin-bottom: 0.45rem;
        }

        .exec-card p {
            color: var(--text-soft);
            font-size: 0.95rem;
            line-height: 1.35;
            margin: 0;
        }

        .exec-card.status,
        .exec-card.evidence,
        .exec-card.action {
            border-left-color: var(--accent);
        }

        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-testid="stNumberInput"] input,
        textarea {
            background: var(--surface-2) !important;
            border-color: var(--border) !important;
            color: var(--text-main) !important;
        }

        .stButton button {
            background: transparent;
            border: 0;
            color: var(--text-muted);
            font-weight: 700;
            text-align: left;
            justify-content: flex-start;
            padding-left: 0;
        }

        .stButton button:hover {
            background: transparent;
            border: 0;
            color: var(--text-main);
        }

        [data-testid="stBaseButton-primary"],
        [data-testid="stBaseButton-primary"]:hover,
        [data-testid="stFormSubmitButton"] button,
        [data-testid="stFormSubmitButton"] button:hover {
            background: var(--accent);
            border: 1px solid var(--accent);
            color: #082f49;
            justify-content: center;
            text-align: center;
            padding-left: 0.75rem;
        }

        [data-testid="stTable"] {
            color: var(--text-muted);
        }

        @media (max-width: 900px) {
            .exec-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(df: pd.DataFrame) -> str:
    safe_best = best_safe_cv_row()
    tuned = tuned_holdout_row()

    with st.sidebar:
        st.title("Swipe Atlas")
        st.caption("Engagement prediction, segmentation, and evidence audit.")

        section = st.radio(
            "Section",
            ["Prediction", "Segments", "Evidence"],
            label_visibility="collapsed",
        )

        st.divider()
        st.metric("Dataset Rows", f"{len(df):,}")
        st.metric("Mean Mutual Matches", f"{df['mutual_matches'].mean():.2f}")

        if tuned is not None:
            st.metric("Tuned Holdout R2", format_number(float(tuned["r2"]), 3))

        if safe_best is not None:
            st.caption(
                f"Best safe CV: {safe_best['model']} "
                f"(R2={float(safe_best['cv_r2_mean']):.3f})."
            )

    return section


def render_executive_snapshot(df: pd.DataFrame) -> None:
    safe_best = best_safe_cv_row()
    tuned = tuned_holdout_row()
    k_selection = load_csv(K_SELECTION_PATH)

    safe_r2 = "n/a" if safe_best is None else f"{float(safe_best['cv_r2_mean']):.3f}"
    holdout_r2 = "n/a" if tuned is None else f"{float(tuned['r2']):.3f}"

    if k_selection is not None and not k_selection.empty:
        best_k = k_selection.sort_values("silhouette", ascending=False).iloc[0]
        segment_text = f"k={int(best_k['k'])}, silhouette={float(best_k['silhouette']):.3f}"
    else:
        segment_text = "segmentation pending"

    st.markdown(
        f"""
        <div class="exec-grid">
            <div class="exec-card status">
                <span>Model Status</span>
                <strong>Low Signal Confirmed</strong>
                <p>Safe CV R2 {safe_r2}; tuned holdout R2 {holdout_r2}. Predictions should be treated as audit evidence, not decisions.</p>
            </div>
            <div class="exec-card evidence">
                <span>Current Evidence</span>
                <strong>{len(df):,} rows reviewed</strong>
                <p>Outcome tests, AutoML, and manual models all converge near baseline performance.</p>
            </div>
            <div class="exec-card action">
                <span>Next Action</span>
                <strong>Report limits clearly</strong>
                <p>Use segments descriptively ({segment_text}); avoid claiming individual-level predictive power.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_header(label: str, key: str, default_open: bool = False) -> bool:
    state_key = f"{key}_open"
    if state_key not in st.session_state:
        st.session_state[state_key] = default_open

    arrow = "▾" if st.session_state[state_key] else "▸"
    if st.button(f"{arrow} {label}", key=f"{key}_button", use_container_width=True):
        st.session_state[state_key] = not st.session_state[state_key]
        st.rerun()

    return bool(st.session_state[state_key])


def prediction_tab(df: pd.DataFrame) -> None:
    st.subheader("Profile Customization & Engagement Prediction")

    target_mean = float(df["mutual_matches"].mean())
    target_min = int(df["mutual_matches"].min())
    target_max = int(df["mutual_matches"].max())
    safe_best = best_safe_cv_row()
    tuned = tuned_holdout_row()

    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows", f"{len(df):,}")
    metric_cols[1].metric("Dataset Mean Matches", f"{target_mean:.2f}")
    metric_cols[2].metric("Match Range", f"{target_min}-{target_max}")
    metric_cols[3].metric(
        "Tuned Holdout R2",
        "n/a" if tuned is None else format_number(float(tuned["r2"]), 3),
    )

    if safe_best is not None:
        st.caption(
            "Current official-safe CV winner: "
            f"{safe_best['model']} (R2={float(safe_best['cv_r2_mean']):.3f}, "
            f"MAE={float(safe_best['cv_mae_mean']):.3f})."
        )

    st.markdown("### Customize User Characteristics")

    row = {
        "gender": sorted(df["gender"].unique())[0],
        "sexual_orientation": sorted(df["sexual_orientation"].unique())[0],
        "location_type": sorted(df["location_type"].unique())[0],
        "income_bracket": sorted(df["income_bracket"].unique())[0],
        "education_level": sorted(df["education_level"].unique())[0],
        "relationship_intent": sorted(df["relationship_intent"].unique())[0],
        "body_type": sorted(df["body_type"].unique())[0],
        "zodiac_sign": sorted(df["zodiac_sign"].unique())[0],
        "app_usage_time_label": sorted(df["app_usage_time_label"].unique())[0],
        "swipe_right_label": sorted(df["swipe_right_label"].unique())[0],
        "swipe_time_of_day": sorted(df["swipe_time_of_day"].unique())[0],
        "interest_tags": df["interest_tags"].head(500).sort_values().unique()[0],
        "app_usage_time_min": 90,
        "swipe_right_ratio": 0.45,
        "profile_pics_count": 4,
        "bio_length": 180,
        "message_sent_count": 35,
        "emoji_usage_rate": 0.35,
        "last_active_hour": 21,
        "age": 27,
        "height_cm": 170,
        "weight_kg": 68.0,
    }

    with st.container(border=True):
        personal_open = panel_header("Personal Profile Settings", "personal_profile_settings", True)
        if personal_open:
            row["gender"] = st.selectbox("Gender", sorted(df["gender"].unique()))
            row["sexual_orientation"] = st.selectbox("Orientation", sorted(df["sexual_orientation"].unique()))
            row["location_type"] = st.selectbox("Location", sorted(df["location_type"].unique()))
            row["income_bracket"] = st.selectbox("Income", sorted(df["income_bracket"].unique()))
            row["education_level"] = st.selectbox("Education", sorted(df["education_level"].unique()))
            row["relationship_intent"] = st.selectbox("Intent", sorted(df["relationship_intent"].unique()))
            row["body_type"] = st.selectbox("Body type", sorted(df["body_type"].unique()))
            row["zodiac_sign"] = st.selectbox("Zodiac", sorted(df["zodiac_sign"].unique()))
            row["app_usage_time_label"] = st.selectbox("Usage label", sorted(df["app_usage_time_label"].unique()))
            row["swipe_right_label"] = st.selectbox("Swipe label", sorted(df["swipe_right_label"].unique()))
            row["swipe_time_of_day"] = st.selectbox("Swipe time", sorted(df["swipe_time_of_day"].unique()))
            row["interest_tags"] = st.selectbox("Interest pattern", df["interest_tags"].head(500).sort_values().unique())

    with st.container(border=True):
        behavior_open = panel_header("Behavioral Features Settings", "behavioral_features_settings", True)
        if behavior_open:
            row["app_usage_time_min"] = st.slider("Usage minutes", 0, 300, 90)
            row["swipe_right_ratio"] = st.slider("Swipe-right ratio", 0.0, 1.0, 0.45, 0.01)
            row["profile_pics_count"] = st.slider("Profile photos", 1, 7, 4)
            row["bio_length"] = st.slider("Bio length", 0, 500, 180)
            row["message_sent_count"] = st.slider("Messages sent", 0, 100, 35)
            row["emoji_usage_rate"] = st.slider("Emoji rate", 0.0, 1.0, 0.35, 0.01)
            row["last_active_hour"] = st.slider("Last active hour", 0, 23, 21)
            row["age"] = st.slider("Age", 18, 59, 27)
            row["height_cm"] = st.slider("Height cm", 145, 200, 170)
            row["weight_kg"] = st.slider("Weight kg", 40.0, 130.0, 68.0, 0.5)

    row["likes_received"] = 0
    row["mutual_matches"] = 0
    row[TARGET] = "No Action"
    sample = pd.DataFrame([row])

    submitted = st.button("See Prediction (Check Prediction)", type="primary", use_container_width=True)

    if submitted:
        st.markdown("---") 
        st.markdown("### Model Prediction Result")
        
        model = load_model()
        if model is None:
            st.warning("Run `python scripts/04_train_engagement_models.py` to create the model artifact.")
            return

        pred = float(model.predict(sample)[0])
        st.metric("Predicted Mutual Matches", f"{pred:.1f}")
        
        st.info(
            "**Interpretation:** the official safe model should stay close to the dataset "
            f"mean of {target_mean:.2f} mutual matches. That matches the current evidence: "
            "the best safe cross-validation result is essentially zero R2, and the tuned "
            "holdout model remains slightly below the mean baseline. Small prediction "
            "movement is expected, but the model is not finding a practically useful signal."
        )
        
        st.markdown("#### Selected Feature Vector (Vertical View)")
        st.table(sample.drop(columns=[TARGET, "mutual_matches"]).T)


def segments_tab(df: pd.DataFrame) -> None:
    st.subheader("Behavioral Segments Analysis")
    assignments = load_csv(SEGMENTS_PATH)
    summary = load_csv(SEGMENT_SUMMARY_PATH)
    k_selection = load_csv(K_SELECTION_PATH)
    
    if assignments is None or summary is None:
        st.warning("Run `python scripts/05_segmentation.py` to create segment artifacts.")
        return

    merged = df.reset_index(names="row_id").merge(assignments, on="row_id", how="inner")
    counts = merged["segment_name"].value_counts().rename_axis("segment").reset_index(name="users")

    if k_selection is not None and not k_selection.empty:
        best_k = k_selection.sort_values("silhouette", ascending=False).iloc[0]
        seg_cols = st.columns(4)
        seg_cols[0].metric("Rows Assigned", f"{len(merged):,}")
        seg_cols[1].metric("Selected k", f"{int(best_k['k'])}")
        seg_cols[2].metric("Best Silhouette", format_number(float(best_k["silhouette"]), 3))
        seg_cols[3].metric(
            "Silhouette Range",
            f"{k_selection['silhouette'].min():.3f}-{k_selection['silhouette'].max():.3f}",
        )
        st.caption(
            "Current segmentation finding: silhouette stays below 0.05 across k=2-10, "
            "so the named segments are descriptive profiles, not strong natural clusters."
        )
    
    import altair as alt
    
    seg_chart = alt.Chart(counts).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X('users:Q', title='Number of Users (Group Size)'),
        y=alt.Y('segment:N', sort='-x', title='Behavioral Segment'),
        color=alt.value('#38BDF8'),  
        tooltip=['segment', 'users']  
    ).properties(
        title='User Distribution Across Discovered Segments',
        height=280
    )
    st.altair_chart(seg_chart, theme="streamlit", use_container_width=True)
    
    st.markdown("---") 
    
    summary_clean = summary.copy()
    
    if "segment_id" in summary_clean.columns:
        summary_clean = summary_clean.drop(columns=["segment_id"])
        
    if "segment_name" in summary_clean.columns:
        summary_clean = summary_clean.set_index("segment_name")
        
    with st.container(border=True):
        if panel_header("Comprehensive segment profile table", "segment_profile_table", False):
            st.table(summary_clean.T)

    findings = load_text(SEGMENTATION_FINDINGS_PATH)
    if findings:
        with st.container(border=True):
            if panel_header("Full Segmentation Findings", "segmentation_findings", False):
                st.markdown(findings)


def evidence_tab() -> None:
    st.subheader("Model Evidence")
    results = load_csv(ENGAGEMENT_RESULTS_PATH)
    if results is not None:
        safe = results[results["setting"] == "official safe"].sort_values("cv_r2_mean", ascending=False).copy()
        best_safe = safe.iloc[0]
        evidence_cols = st.columns(4)
        evidence_cols[0].metric("Best Safe CV Model", str(best_safe["model"]))
        evidence_cols[1].metric("Best Safe CV R2", format_number(float(best_safe["cv_r2_mean"]), 3))
        evidence_cols[2].metric("Best Safe CV MAE", format_number(float(best_safe["cv_mae_mean"]), 3))
        evidence_cols[3].metric("Models Compared", f"{len(safe):,}")
        
        with st.container(border=True):
            if panel_header("Official safe CV detail table", "safe_cv_table", False):
                st.table(safe.drop(columns=["color"], errors="ignore"))
        
        import altair as alt
        
        safe['color'] = safe['model'].apply(lambda x: '#64748B' if x == 'Dummy mean' else '#38BDF8')
        
        chart = alt.Chart(safe).mark_bar().encode(
            x=alt.X('cv_r2_mean:Q', title='CV R² Mean (Negative values indicate worse performance)'),
            y=alt.Y('model:N', sort='-x', title='Model'),
            color=alt.Color('color:N', scale=None),
            tooltip=['model', 'cv_r2_mean']
        ).properties(
            title='Model Comparison (Horizontal View)',
            height=400
        )
        st.altair_chart(chart, theme="streamlit", use_container_width=True)
    else:
        st.warning("Run `python scripts/04_train_engagement_models.py` to create model evidence.")

    st.subheader("AutoML and Holdout Comparison")
    automl = load_csv(AUTOML_RESULTS_PATH)
    if automl is not None:
        automl_sorted = automl.sort_values("r2", ascending=False).copy()
        best = automl_sorted.iloc[0]
        automl_cols = st.columns(4)
        automl_cols[0].metric("Best Holdout Row", str(best["model"]))
        automl_cols[1].metric("Best Holdout R2", format_number(float(best["r2"]), 3))
        automl_cols[2].metric("Best Holdout MAE", format_number(float(best["mae"]), 3))
        automl_cols[3].metric("Best Holdout RMSE", format_number(float(best["rmse"]), 3))
        with st.container(border=True):
            if panel_header("AutoML and holdout detail table", "automl_table", False):
                st.table(automl_sorted[["model", "backend", "status", "r2", "mae", "rmse"]])
        st.caption(
            "AutoML confirms the same conclusion as the manual workflow: all safe holdout "
            "models land at approximately zero R2."
        )
    else:
        st.warning("Run `python scripts/06_automl_comparison.py` to create AutoML evidence.")

    st.subheader("Match Outcome Signal Check")
    text = load_text(SIGNAL_PATH)
    if text:
        with st.container(border=True):
            if panel_header("Full match-outcome signal findings", "signal_findings", False):
                st.markdown(text)
    else:
        st.warning("Run `python scripts/02_signal_test.py` to create signal findings.")

def main() -> None:
    st.set_page_config(page_title="Swipe Atlas", layout="wide", initial_sidebar_state="expanded")
    apply_inter_tight_theme()
    
    df = load_data() 
    section = render_sidebar(df)

    st.title("Swipe Atlas")
    render_executive_snapshot(df)

    if section == "Prediction":
        prediction_tab(df)
    elif section == "Segments":
        segments_tab(df)
    else:
        evidence_tab()

if __name__ == "__main__":
    main()
