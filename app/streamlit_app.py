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
MODEL_PATH        = ROOT / "models" / "best_mutual_matches_model.joblib"
SEGMENTS_PATH     = ROOT / "data" / "processed" / "segmentation_assignments.csv"
RESULTS_PATH      = ROOT / "reports" / "engagement_model_results.csv"
AUTOML_PATH       = ROOT / "reports" / "automl_results.csv"
SEG_SUMMARY_PATH  = ROOT / "reports" / "segmentation_summary.csv"
K_SEL_PATH        = ROOT / "reports" / "segmentation_k_selection.csv"
SIGNAL_PATH       = ROOT / "reports" / "signal_findings.md"
FIGDIR            = ROOT / "reports" / "figures"


# ── Data loaders ────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    return load_raw(extended=True)

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None

def load_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else None

def fmt(v, d=3):
    return "n/a" if pd.isna(v) else f"{v:,.{d}f}"


# ── Theme ────────────────────────────────────────────────────────────────────

def apply_theme() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg:       #070E1A;
        --surface:  #0C1526;
        --card:     #101E34;
        --card2:    #152540;
        --border:   #1C2E4A;
        --border2:  #243B5E;
        --txt:      #DDE8F8;
        --txt2:     #7E99C0;
        --txt3:     #4A6180;
        --green:    #22C55E;
        --red:      #EF4444;
        --amber:    #F59E0B;
        --blue:     #3B82F6;
        --blue2:    #1D4ED8;
    }

    /* ── Base ── */
    html, body, .stApp { background: var(--bg) !important; }
    * { font-family: 'Inter Tight', sans-serif !important; box-sizing: border-box; }
    h1,h2,h3,h4,h5 { font-family: 'Inter Tight', sans-serif !important; color: var(--txt); }
    code, pre, .mono { font-family: 'JetBrains Mono', monospace !important; }

    /* ── Streamlit chrome ── */
    [data-testid="stHeader"] { background: var(--bg) !important; border-bottom: 1px solid var(--border); }
    [data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
    [data-testid="stSidebar"] > div { background: var(--surface) !important; }
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px; }

    /* ── Typography ── */
    p, li, label, span, div { color: var(--txt2); }
    [data-testid="stMarkdownContainer"] p { color: var(--txt2); line-height: 1.6; }
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 { color: var(--txt); margin-top: 1.2rem; }

    /* ── Metrics ── */
    [data-testid="stMetricLabel"]  { color: var(--txt3) !important; font-size: 0.78rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.06em; }
    [data-testid="stMetricValue"]  { color: var(--txt) !important; font-family: 'Inter Tight', sans-serif !important; font-size: 1.9rem !important; font-weight: 700 !important; }
    [data-testid="stMetricDelta"]  { font-size: 0.82rem !important; }
    [data-testid="stMetric"]       { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.1rem; }

    /* ── Inputs ── */
    [data-baseweb="select"] > div, [data-baseweb="input"] > div, textarea {
        background: var(--card2) !important; border-color: var(--border2) !important; color: var(--txt) !important; border-radius: 8px !important;
    }
    [data-testid="stSlider"] { accent-color: var(--blue); }
    [data-testid="stWidgetLabel"] { color: var(--txt2) !important; font-size: 0.85rem !important; font-weight: 500 !important; }

    /* ── Buttons ── */
    [data-testid="stBaseButton-primary"] button {
        background: var(--blue) !important; border: none !important; color: #fff !important;
        border-radius: 8px !important; font-weight: 600 !important; font-size: 0.9rem !important;
        transition: background 0.2s;
    }
    [data-testid="stBaseButton-primary"] button:hover { background: var(--blue2) !important; }
    [data-testid="stBaseButton-secondary"] button {
        background: var(--card2) !important; border: 1px solid var(--border2) !important;
        color: var(--txt) !important; border-radius: 8px !important;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid var(--border); gap: 0; }
    [data-testid="stTabs"] button[role="tab"] {
        background: transparent !important; color: var(--txt3) !important;
        border: none !important; border-bottom: 2px solid transparent !important;
        font-weight: 600 !important; font-size: 0.88rem !important;
        padding: 0.6rem 1.1rem !important; border-radius: 0 !important;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--blue) !important; border-bottom-color: var(--blue) !important;
    }
    [data-testid="stTabs"] button[role="tab"]:hover { color: var(--txt) !important; }

    /* ── Containers / expanders ── */
    [data-testid="stExpander"] {
        background: var(--card) !important; border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"] summary { color: var(--txt2) !important; font-weight: 600 !important; }
    [data-testid="stContainer"] [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 10px !important;
    }

    /* ── Tables / dataframes ── */
    [data-testid="stTable"] table { background: var(--card) !important; color: var(--txt2) !important; }
    [data-testid="stTable"] th { background: var(--card2) !important; color: var(--txt3) !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
    [data-testid="stTable"] td { color: var(--txt) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.84rem !important; }
    .dataframe-container { border-radius: 10px; overflow: hidden; }

    /* ── Alerts ── */
    [data-testid="stAlert"] { border-radius: 8px !important; border-left-width: 4px !important; }

    /* ── Dividers ── */
    hr { border-color: var(--border) !important; margin: 1.2rem 0; }

    /* ── Custom components ── */
    .kpi-strip {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.7rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.9rem 1rem;
    }
    .kpi-card .kpi-label {
        font-size: 0.72rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.07em;
        color: var(--txt3); margin-bottom: 0.3rem; display: block;
    }
    .kpi-card .kpi-value {
        font-family: 'Inter Tight', sans-serif;
        font-size: 1.5rem; font-weight: 700;
        color: var(--txt); line-height: 1;
        display: block; margin-bottom: 0.2rem;
    }
    .kpi-card .kpi-sub {
        font-size: 0.75rem; color: var(--txt3); display: block;
    }
    .kpi-card.red   { border-top: 3px solid var(--red); }
    .kpi-card.green { border-top: 3px solid var(--green); }
    .kpi-card.amber { border-top: 3px solid var(--amber); }
    .kpi-card.blue  { border-top: 3px solid var(--blue); }
    .kpi-card.neutral { border-top: 3px solid var(--border2); }

    .section-label {
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.1em; color: var(--txt3);
        margin-bottom: 0.6rem; display: block;
    }
    .finding-badge {
        display: inline-block;
        background: rgba(239,68,68,0.12);
        border: 1px solid rgba(239,68,68,0.3);
        color: #FCA5A5;
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        font-size: 0.78rem; font-weight: 600;
        margin-right: 0.4rem; margin-bottom: 0.4rem;
    }
    .pred-result {
        background: var(--card2);
        border: 1px solid var(--border2);
        border-left: 4px solid var(--blue);
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin: 1rem 0;
    }
    .pred-result .pred-label { font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; color: var(--txt3); display: block; margin-bottom: 0.3rem; }
    .pred-result .pred-value { font-family: 'Inter Tight', sans-serif; font-size: 2.8rem; font-weight: 800; color: var(--txt); display: block; line-height: 1; }
    .pred-result .pred-note  { font-size: 0.8rem; color: var(--txt3); display: block; margin-top: 0.4rem; }

    @media (max-width: 900px) {
        .kpi-strip { grid-template-columns: repeat(2, 1fr); }
    }
    </style>
    """, unsafe_allow_html=True)


# ── KPI Strip ────────────────────────────────────────────────────────────────

def kpi_strip(df: pd.DataFrame) -> None:
    results = load_csv(RESULTS_PATH)
    automl  = load_csv(AUTOML_PATH)
    k_sel   = load_csv(K_SEL_PATH)

    tuned_r2 = "n/a"
    if automl is not None:
        row = automl[automl["model"] == "Best manual tuned HistGB"]
        if not row.empty:
            tuned_r2 = f"{float(row.iloc[0]['r2']):.3f}"

    best_cv = "n/a"
    if results is not None:
        safe = results[results["setting"] == "official safe"].sort_values("cv_r2_mean", ascending=False)
        if not safe.empty:
            best_cv = f"{float(safe.iloc[0]['cv_r2_mean']):.3f}"

    seg_k = "k=3"
    sil   = "0.019"
    if k_sel is not None and not k_sel.empty:
        best = k_sel.sort_values("silhouette", ascending=False).iloc[0]
        seg_k = f"k={int(best['k'])}"
        sil   = f"{float(best['silhouette']):.3f}"

    st.markdown(f"""
    <div class="kpi-strip">
        <div class="kpi-card neutral">
            <span class="kpi-label">Dataset</span>
            <span class="kpi-value">{len(df):,}</span>
            <span class="kpi-sub">rows · 25 features</span>
        </div>
        <div class="kpi-card red">
            <span class="kpi-label">Tuned Holdout R²</span>
            <span class="kpi-value">{tuned_r2}</span>
            <span class="kpi-sub">≈ 0 · no useful signal</span>
        </div>
        <div class="kpi-card red">
            <span class="kpi-label">Best CV R²</span>
            <span class="kpi-value">{best_cv}</span>
            <span class="kpi-sub">safe feature set · 5-fold</span>
        </div>
        <div class="kpi-card amber">
            <span class="kpi-label">Segmentation</span>
            <span class="kpi-value">{seg_k}</span>
            <span class="kpi-sub">silhouette {sil} · weak structure</span>
        </div>
        <div class="kpi-card green">
            <span class="kpi-label">AutoML Confirmed</span>
            <span class="kpi-value">3 / 3</span>
            <span class="kpi-sub">FLAML · AutoGluon · auto-sklearn</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("""
        <div style="padding: 0.5rem 0 1rem">
            <div style="font-family:'Inter Tight',sans-serif; font-size:1.4rem; font-weight:800; color:#DDE8F8; line-height:1;">Swipe Atlas</div>
            <div style="font-size:0.78rem; color:#4A6180; margin-top:0.25rem; font-weight:500;">ML Engagement Dashboard</div>
        </div>
        """, unsafe_allow_html=True)

        section = st.radio(
            "Navigate",
            ["Overview", "Predict", "Evidence", "Segments"],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown('<span class="section-label">Key Finding</span>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#0F1E35;border:1px solid #1C2E4A;border-left:3px solid #EF4444;border-radius:8px;padding:0.8rem 0.9rem;">
            <div style="font-size:0.82rem;color:#FCA5A5;font-weight:600;margin-bottom:0.3rem;">No Predictive Signal</div>
            <div style="font-size:0.78rem;color:#7E99C0;line-height:1.5;">
            0/23 statistical tests survive correction.<br>
            All models converge at R²≈0.<br>
            Data is synthetic.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown('<span class="section-label">Leakage Rule</span>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.78rem;color:#7E99C0;line-height:1.6;">
        <code style="background:#152540;padding:1px 5px;border-radius:4px;color:#93C5FD;">likes_received</code> excluded<br>
        Including it inflates R² from<br>
        −0.001 → <span style="color:#F59E0B;font-weight:600;">0.127</span> (leakage)
        </div>
        """, unsafe_allow_html=True)

    return section


# ── SHAP Waterfall ────────────────────────────────────────────────────────────

def show_shap_waterfall(model, sample_df: pd.DataFrame) -> None:
    try:
        import shap, matplotlib
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
            explainer = shap.Explainer(estimator, X_feats.values)
            sv = explainer(X_feats.values)
        sv.feature_names = feature_names
        shap.plots.waterfall(sv[0], max_display=12, show=False)
        fig = plt.gcf()
        st.pyplot(fig, clear_figure=True)
    except Exception as exc:
        st.caption(f"SHAP unavailable: {exc}")


# ── Tabs ─────────────────────────────────────────────────────────────────────

def tab_overview(df: pd.DataFrame) -> None:
    st.markdown("### Project Summary")
    st.markdown("""
    <div style="background:#0C1526;border:1px solid #1C2E4A;border-radius:10px;padding:1rem 1.2rem;margin-bottom:1rem;">
    <span style="color:#DDE8F8;font-size:0.92rem;line-height:1.7;">
    <b style="color:#DDE8F8;">Swipe Atlas</b> tests whether dating-app profile and behaviour features can predict
    engagement outcomes. After running 10 regression models, 3 AutoML systems, statistical correction across
    23 tests, and unsupervised clustering — every method independently confirms
    <b style="color:#FCA5A5;">R²≈0</b>. The dataset is synthetic, which explains the absence of signal.
    </span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<span class="section-label">Key Findings</span>', unsafe_allow_html=True)
        findings = [
            "match_outcome has no predictive signal",
            "0/23 statistical tests survive Bonferroni / BH-FDR correction",
            "All 10 models converge at CV R²≈0 (safe feature set)",
            "likes_received leakage inflates R² from −0.001 → 0.127",
            "FLAML, AutoGluon, auto-sklearn all confirm R²≈0",
            "K-Means silhouette = 0.019 (below 0.25 'weak' threshold)",
            "HDBSCAN classifies all 50k users as noise",
            "Data is synthetic — perfect balance, uniform tag frequencies",
        ]
        for f in findings:
            st.markdown(f'<span class="finding-badge">{f}</span>', unsafe_allow_html=True)

    with col2:
        st.markdown('<span class="section-label">Figure Gallery</span>', unsafe_allow_html=True)
        fig_cols = st.columns(2)
        figs = [
            ("10_signal_summary.png",     "Signal Test Summary"),
            ("13_leakage_comparison.png", "Leakage Comparison"),
            ("12_engagement_model_comparison.png", "Model Comparison"),
            ("16_segment_umap.png",       "UMAP Segments"),
        ]
        for i, (fname, caption) in enumerate(figs):
            path = FIGDIR / fname
            if path.exists():
                fig_cols[i % 2].image(str(path), caption=caption, use_container_width=True)


def tab_predict(df: pd.DataFrame) -> None:
    st.markdown("### Engagement Prediction")
    st.warning("This model has R²≈0 on held-out data. Predictions are illustrative only — the model cannot reliably predict individual outcomes on this synthetic dataset.")

    model = load_model()
    if model is None:
        st.error("Model artifact not found. Run `python scripts/04_train_engagement_models.py` first.")
        return

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<span class="section-label">Profile</span>', unsafe_allow_html=True)
        gender      = st.selectbox("Gender",      sorted(df["gender"].unique()))
        orientation = st.selectbox("Orientation", sorted(df["sexual_orientation"].unique()))
        location    = st.selectbox("Location",    sorted(df["location_type"].unique()))
        intent      = st.selectbox("Intent",      sorted(df["relationship_intent"].unique()))
        income      = st.selectbox("Income",      sorted(df["income_bracket"].unique()))
        education   = st.selectbox("Education",   sorted(df["education_level"].unique()))
        body_type   = st.selectbox("Body type",   sorted(df["body_type"].unique()))
        zodiac      = st.selectbox("Zodiac",      sorted(df["zodiac_sign"].unique()))
        usage_lbl   = st.selectbox("Usage label", sorted(df["app_usage_time_label"].unique()))
        swipe_lbl   = st.selectbox("Swipe label", sorted(df["swipe_right_label"].unique()))
        swipe_tod   = st.selectbox("Swipe time",  sorted(df["swipe_time_of_day"].unique()))
        tags        = st.selectbox("Interest pattern", df["interest_tags"].head(500).sort_values().unique())

        st.markdown('<span class="section-label" style="margin-top:0.8rem;">Behaviour</span>', unsafe_allow_html=True)
        usage_min  = st.slider("Usage minutes",     0,   300,  90)
        swipe_r    = st.slider("Swipe-right ratio", 0.0, 1.0,  0.45, 0.01)
        pics       = st.slider("Profile photos",    0,   6,    4)
        bio        = st.slider("Bio length",        0,   500,  180)
        msgs       = st.slider("Messages sent",     0,   100,  35)
        emoji      = st.slider("Emoji rate",        0.0, 1.0,  0.35, 0.01)
        hour       = st.slider("Last active hour",  0,   23,   21)
        age        = st.slider("Age",               18,  59,   27)
        height     = st.slider("Height cm",         145, 200,  170)
        weight     = st.slider("Weight kg",         40.0, 130.0, 68.0, 0.5)

    row = {
        "gender": gender, "sexual_orientation": orientation,
        "location_type": location, "income_bracket": income,
        "education_level": education, "relationship_intent": intent,
        "body_type": body_type, "zodiac_sign": zodiac,
        "app_usage_time_label": usage_lbl, "swipe_right_label": swipe_lbl,
        "swipe_time_of_day": swipe_tod, "interest_tags": tags,
        "app_usage_time_min": usage_min, "swipe_right_ratio": swipe_r,
        "profile_pics_count": pics, "bio_length": bio,
        "message_sent_count": msgs, "emoji_usage_rate": emoji,
        "last_active_hour": hour, "age": age, "height_cm": height,
        "weight_kg": weight, "likes_received": 0,
        "mutual_matches": 0, TARGET: "No Action",
    }
    sample = pd.DataFrame([row])

    with col_right:
        pred = float(model.predict(sample)[0])
        mean = float(df["mutual_matches"].mean())
        delta = pred - mean

        st.markdown('<span class="section-label">Prediction Result</span>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="pred-result">
            <span class="pred-label">Predicted Mutual Matches</span>
            <span class="pred-value">{pred:.1f}</span>
            <span class="pred-note">Dataset mean: {mean:.2f} · Δ from mean: {delta:+.2f}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<span class="section-label" style="margin-top:1rem;">SHAP Feature Attribution</span>', unsafe_allow_html=True)
        st.caption("All SHAP values land near zero — no feature meaningfully drives the prediction.")
        show_shap_waterfall(model, sample)


def tab_evidence() -> None:
    st.markdown("### Model Evidence")

    results = load_csv(RESULTS_PATH)
    automl  = load_csv(AUTOML_PATH)

    # ── KPI row ──
    if results is not None and automl is not None:
        safe = results[results["setting"] == "official safe"].sort_values("cv_r2_mean", ascending=False)
        tuned_row = automl[automl["model"] == "Best manual tuned HistGB"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Models Compared", f"{len(safe)}")
        c2.metric("Best CV R²",      fmt(float(safe.iloc[0]["cv_r2_mean"])) if not safe.empty else "n/a")
        c3.metric("Tuned R²",        fmt(float(tuned_row.iloc[0]["r2"])) if not tuned_row.empty else "n/a")
        c4.metric("AutoML Systems",  "3")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<span class="section-label">Model Comparison — Safe Feature Set</span>', unsafe_allow_html=True)
        fig_path = FIGDIR / "12_engagement_model_comparison.png"
        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True)
        elif results is not None:
            safe_disp = results[results["setting"] == "official safe"].sort_values("cv_r2_mean", ascending=False)
            st.dataframe(safe_disp[["model", "cv_r2_mean", "cv_mae_mean"]].round(4), use_container_width=True)

    with col2:
        st.markdown('<span class="section-label">Leakage Demonstration</span>', unsafe_allow_html=True)
        fig_path = FIGDIR / "13_leakage_comparison.png"
        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True)
        else:
            st.caption("Run script 04 to generate this figure.")

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<span class="section-label">Signal Test — match_outcome</span>', unsafe_allow_html=True)
        fig_path = FIGDIR / "10_signal_summary.png"
        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True)

    with col4:
        st.markdown('<span class="section-label">AutoML Comparison</span>', unsafe_allow_html=True)
        fig_path = FIGDIR / "20_automl_comparison.png"
        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True)
        elif automl is not None:
            st.dataframe(automl[["model", "backend", "r2", "mae"]].round(4), use_container_width=True)

    st.divider()
    st.markdown('<span class="section-label">Learning Curve</span>', unsafe_allow_html=True)
    fig_path = FIGDIR / "14b_learning_curve.png"
    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)
        st.caption("R² stays flat as training size increases — more data cannot fix zero signal.")

    with st.expander("Full model results table"):
        if results is not None:
            safe = results[results["setting"] == "official safe"].sort_values("cv_r2_mean", ascending=False)
            st.dataframe(safe[["model","cv_r2_mean","cv_r2_std","cv_mae_mean","cv_rmse_mean"]].round(4), use_container_width=True)

    with st.expander("AutoML detailed results"):
        if automl is not None:
            st.dataframe(automl[["model","backend","status","r2","mae","rmse"]].round(4), use_container_width=True)


def tab_segments(df: pd.DataFrame) -> None:
    st.markdown("### User Segmentation")
    st.info("Silhouette score = 0.019 (below the 0.25 'weak structure' threshold). Segments are descriptive profiles, not tight natural clusters.")

    summary     = load_csv(SEG_SUMMARY_PATH)
    k_sel       = load_csv(K_SEL_PATH)
    assignments = load_csv(SEGMENTS_PATH)

    # ── KPI row ──
    if k_sel is not None and not k_sel.empty:
        best = k_sel.sort_values("silhouette", ascending=False).iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Selected k",       f"{int(best['k'])}")
        c2.metric("Best Silhouette",  f"{float(best['silhouette']):.3f}")
        c3.metric("Silhouette Range", f"{k_sel['silhouette'].min():.3f}–{k_sel['silhouette'].max():.3f}")
        c4.metric("Total Users",      f"{len(df):,}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<span class="section-label">UMAP Projection</span>', unsafe_allow_html=True)
        umap_path = FIGDIR / "16_segment_umap.png"
        if umap_path.exists():
            st.image(str(umap_path), use_container_width=True)
            st.caption("Diffuse, overlapping clusters confirm weak structure.")

    with col2:
        st.markdown('<span class="section-label">Segment Profiles</span>', unsafe_allow_html=True)
        profile_path = FIGDIR / "17_segment_profiles.png"
        if profile_path.exists():
            st.image(str(profile_path), use_container_width=True)

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<span class="section-label">K Selection — Silhouette Sweep</span>', unsafe_allow_html=True)
        k_path = FIGDIR / "15_kmeans_selection.png"
        if k_path.exists():
            st.image(str(k_path), use_container_width=True)

    with col4:
        st.markdown('<span class="section-label">GMM BIC / AIC</span>', unsafe_allow_html=True)
        gmm_path = FIGDIR / "18_gmm_bic_aic.png"
        if gmm_path.exists():
            st.image(str(gmm_path), use_container_width=True)
            st.caption("Monotone BIC/AIC decrease — no preferred cluster count.")

    if summary is not None:
        st.divider()
        st.markdown('<span class="section-label">Segment Summary Table</span>', unsafe_allow_html=True)
        disp = summary.copy()
        if "segment_id" in disp.columns:
            disp = disp.drop(columns=["segment_id"])
        st.dataframe(disp.round(2), use_container_width=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Swipe Atlas",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()

    df = load_data()
    section = render_sidebar()

    # Page header
    st.markdown("""
    <div style="display:flex;align-items:baseline;gap:0.8rem;margin-bottom:0.5rem;">
        <span style="font-family:'Inter Tight',sans-serif;font-size:1.9rem;font-weight:800;color:#DDE8F8;">Swipe Atlas</span>
        <span style="font-size:0.82rem;color:#4A6180;font-weight:500;">Engagement Prediction · User Segmentation · Model Evidence</span>
    </div>
    """, unsafe_allow_html=True)

    kpi_strip(df)

    if "Overview" in section:
        tab_overview(df)
    elif "Predict" in section:
        tab_predict(df)
    elif "Evidence" in section:
        tab_evidence()
    elif "Segments" in section:
        tab_segments(df)


if __name__ == "__main__":
    main()
