"""
Streamlit dashboard for Swipe Atlas.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

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
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=block');

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

    /* ── Material Symbols: global icon font preservation ──────────────────────
       Streamlit assigns .material-symbols-rounded to EVERY icon span it renders
       (expander chevrons, sidebar arrows, button icons, tabs, etc).
       Our * rule wins by !important — so we must win back here with higher
       specificity AND !important to beat the cascade.
    ──────────────────────────────────────────────────────────────────────────── */
    .material-symbols-rounded,
    .material-symbols-outlined,
    .material-icons,
    .material-icons-outlined,
    [class*="material-symbols"],
    [class*="material-icons"] {
        font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
        font-style: normal !important;
        font-weight: normal !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        direction: ltr !important;
        -webkit-font-smoothing: antialiased !important;
        -webkit-font-feature-settings: 'liga' 1 !important;
        font-feature-settings: 'liga' 1 !important;
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24 !important;
    }

    /* ── Streamlit chrome ── */
    [data-testid="stHeader"] { background: var(--bg) !important; border-bottom: 1px solid var(--border); }
    [data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
    [data-testid="stSidebar"] > div { background: var(--surface) !important; overflow-y: auto !important; overflow-x: hidden !important; }
    /* ── Kill every layer of Streamlit's sidebar top padding ── */
    [data-testid="stSidebarContent"]                             { padding-top: 0 !important; }
    [data-testid="stSidebarHeader"]                              { height: 0 !important; min-height: 0 !important; padding: 0 !important; }
    [data-testid="stLogoSpacer"]                                 { display: none !important; }
    [data-testid="stSidebarCollapseButton"]                      { position: absolute !important; top: 0.45rem !important; right: 0.55rem !important; z-index: 5 !important; }
    [data-testid="stSidebarUserContent"]                         { padding-top: 0 !important; }
    [data-testid="stSidebar"] > div > div:first-child            { padding-top: 0 !important; margin-top: 0 !important; }
    [data-testid="stSidebar"] > div > div:first-child > div      { padding-top: 0 !important; margin-top: 0 !important; }
    [data-testid="stSidebar"] section                            { padding-top: 0 !important; margin-top: 0 !important; }
    [data-testid="stSidebar"] section > div                      { padding-top: 0 !important; padding-bottom: 0.5rem !important; }
    [data-testid="stSidebar"] section > div > div:first-child    { padding-top: 0 !important; margin-top: 0 !important; }
    .block-container { padding-top: 0.75rem; padding-bottom: 3rem; max-width: 1200px; }

    /* ── Sidebar collapse / expand arrows ─────────────────────────────────────
       Belt-and-suspenders: explicit test-ids + wildcard substring match.
       The icon spans here have NO class in most Streamlit versions, so the
       global .material-symbols-rounded rule above doesn't reach them.
       We need data-testid targeting specifically.
    ──────────────────────────────────────────────────────────────────────────── */
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="stSidebarCollapseButton"] p,
    [data-testid="stSidebarCollapseButton"] div,
    [data-testid="stSidebarCollapsedControl"] span,
    [data-testid="stSidebarCollapsedControl"] p,
    [data-testid="stSidebarCollapsedControl"] div,
    [data-testid="collapsedControl"] span,
    [data-testid="collapsedControl"] p,
    [data-testid="collapsedControl"] div,
    [data-testid*="Collapse"] span,
    [data-testid*="Collapse"] p,
    [data-testid*="collapsed"] span,
    [data-testid*="collapsed"] p {
        font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
        font-style: normal !important;
        font-size: 1.25rem !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        -webkit-font-feature-settings: 'liga' 1 !important;
        font-feature-settings: 'liga' 1 !important;
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24 !important;
    }
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] button,
    [data-testid*="Collapse"] button {
        background: transparent !important; border: none !important; color: var(--txt3) !important;
    }
    [data-testid="stSidebarCollapseButton"] button > span,
    [data-testid="stExpandSidebarButton"] > span,
    [data-testid="stSidebarCollapsedControl"] button > span,
    [data-testid="collapsedControl"] button > span,
    [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
    [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"],
    [data-testid="stSidebarCollapsedControl"] [data-testid="stIconMaterial"],
    [data-testid="collapsedControl"] [data-testid="stIconMaterial"] {
        display: none !important;
    }
    [data-testid="stSidebarCollapseButton"] button::before,
    [data-testid="stExpandSidebarButton"]::before,
    [data-testid="stSidebarCollapsedControl"] button::before,
    [data-testid="collapsedControl"] button::before {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 1.5rem !important;
        height: 1.5rem !important;
        color: var(--txt3) !important;
        font-family: 'Inter Tight', sans-serif !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        line-height: 1 !important;
    }
    [data-testid="stSidebarCollapseButton"] button::before {
        content: '‹';
    }
    [data-testid="stExpandSidebarButton"]::before,
    [data-testid="stSidebarCollapsedControl"] button::before,
    [data-testid="collapsedControl"] button::before {
        content: '›';
    }

    /* ── Sidebar markdown containers — always stretch full width ── */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] > div {
        width: 100% !important; max-width: 100% !important; box-sizing: border-box !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] div,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
        max-width: 100% !important; box-sizing: border-box !important;
        word-break: break-word !important; overflow-wrap: break-word !important;
    }

    /* ── Sidebar nav: pure HTML anchors — zero Streamlit button interference ── */
    .nav-menu {
        display: flex;
        flex-direction: column;
        gap: 1px;
        width: 100%;
        margin: 0;
        padding: 0;
    }
    .nav-item {
        display: block !important;
        width: 100% !important;
        box-sizing: border-box !important;
        padding: 0.55rem 0.85rem !important;
        border-radius: 6px !important;
        text-decoration: none !important;
        color: var(--txt3) !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        line-height: 1.1 !important;
        letter-spacing: 0.01em !important;
        transition: background 0.1s ease, color 0.1s ease !important;
        cursor: pointer !important;
    }
    .nav-item:hover {
        background: rgba(255,255,255,0.05) !important;
        color: var(--txt2) !important;
        text-decoration: none !important;
    }
    .nav-active {
        background: rgba(255,255,255,0.09) !important;
        color: var(--txt) !important;
        font-weight: 700 !important;
    }
    .nav-active:hover {
        background: rgba(255,255,255,0.12) !important;
        color: var(--txt) !important;
    }

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

    /* ── Expanders ─────────────────────────────────────────────────────────────
       Design-intentional: CSS ::after chevron so we're never dependent on
       Material Symbols loading. Any broken icon span inside summary is hidden;
       the ::after ›  rotates on open — clean, precise, always renders.
    ──────────────────────────────────────────────────────────────────────────── */
    [data-testid="stExpander"] {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        transition: border-color 0.15s ease !important;
    }
    [data-testid="stExpander"]:hover {
        border-color: var(--border2) !important;
    }
    /* Summary row */
    [data-testid="stExpander"] summary {
        list-style: none !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        padding: 0.85rem 1.1rem !important;
        color: var(--txt2) !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        user-select: none !important;
        position: relative !important;
    }
    [data-testid="stExpander"] summary::-webkit-details-marker,
    [data-testid="stExpander"] summary::marker { display: none !important; }
    /* Hide Streamlit's icon span/wrapper - we replace it with ::after */
    [data-testid="stExpander"] summary [data-testid="stExpanderToggleIcon"],
    [data-testid="stExpander"] summary [data-testid="stIconMaterial"],
    [data-testid="stExpander"] summary .material-symbols-rounded,
    [data-testid="stExpander"] summary .material-symbols-outlined,
    [data-testid="stExpander"] summary [class*="material-symbols"] {
        display: none !important;
    }
    [data-testid="stExpander"] summary [data-testid="stIconMaterial"] {
        width: 0 !important;
        min-width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }
    /* Our chevron — Inter Tight, always renders, no font dependency */
    [data-testid="stExpander"] summary::after {
        content: '›';
        font-family: 'Inter Tight', sans-serif !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: var(--txt3);
        margin-left: auto;
        padding-left: 0.75rem;
        transition: transform 0.18s ease, color 0.15s ease;
        display: inline-block;
        line-height: 1;
        flex-shrink: 0;
    }
    [data-testid="stExpander"][open] > summary::after {
        transform: rotate(90deg);
        color: var(--txt2);
    }
    [data-testid="stExpander"] summary:hover { color: var(--txt) !important; }
    [data-testid="stExpander"] summary:hover::after { color: var(--txt2); }
    /* Content padding */
    [data-testid="stExpander"] > div[data-testid] {
        padding: 0 1.1rem 1rem !important;
    }

    /* ── Other containers ── */
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

    /* ── Page header ── */
    .page-header {
        margin-bottom: 1.5rem;
        padding-bottom: 1.3rem;
        border-bottom: 1px solid var(--border);
    }
    .page-header-eyebrow {
        font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.15em; color: var(--blue);
        margin-bottom: 0.5rem; display: block;
    }
    .page-header-title {
        font-size: 1.85rem; font-weight: 800;
        color: var(--txt); letter-spacing: -0.03em;
        line-height: 1; margin-bottom: 0.5rem; display: block;
    }
    .page-header-subtitle {
        font-size: 0.9rem; color: var(--txt2);
        line-height: 1.65; font-weight: 400; max-width: 640px; display: block;
    }

    @media (max-width: 900px) {
        .kpi-strip { grid-template-columns: repeat(2, 1fr); }
        .page-header-title { font-size: 1.45rem; }
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

_NAV_ITEMS = [
    ("Overview", "▣"),
    ("Analysis", "◈"),
    ("Predict",  "◎"),
    ("Evidence", "≡"),
    ("Segments", "◉"),
]

def render_sidebar() -> str:
    # URL-based routing: persists across browser refreshes, no session state needed
    _valid = {item[0] for item in _NAV_ITEMS}
    page = st.query_params.get("page", "Overview")
    if page not in _valid:
        page = "Overview"

    with st.sidebar:
        # ── Brand — flush to top, no gap above ──
        st.markdown("""
        <div style="width:100%;box-sizing:border-box;
                    padding:0.75rem 2.3rem 0.85rem 0.1rem;
                    border-bottom:1px solid #1C2E4A;margin-bottom:1rem;">
            <div style="display:flex;align-items:center;gap:0.7rem;">
                <div style="flex-shrink:0;width:32px;height:32px;background:#3B82F6;border-radius:8px;
                            display:flex;align-items:center;justify-content:center;">
                    <span style="font-family:'Inter Tight',sans-serif;font-size:0.95rem;font-weight:900;
                                 color:#fff;line-height:1;">S</span>
                </div>
                <div>
                    <div style="font-family:'Inter Tight',sans-serif;font-size:1.3rem;font-weight:800;
                                color:#DDE8F8;line-height:1;letter-spacing:-0.02em;">Swipe Atlas</div>
                    <div style="font-size:0.6rem;color:#3B82F6;font-weight:700;text-transform:uppercase;
                                letter-spacing:0.11em;margin-top:0.2rem;">ML Dashboard</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── GENERAL section label ──
        st.markdown("""
        <p style="font-size:0.62rem;font-weight:700;text-transform:uppercase;letter-spacing:0.13em;
                  color:#4A6180;margin:0 0 0.25rem 0.25rem;padding:0;line-height:1;">General</p>
        """, unsafe_allow_html=True)

        # ── Nav: HTML anchors with ?page= routing — pure CSS, zero Streamlit button override ──
        nav_html = '<div class="nav-menu">'
        for name, icon in _NAV_ITEMS:
            cls = "nav-item nav-active" if page == name else "nav-item"
            nav_html += f'<a href="?page={name}" class="{cls}" target="_self">{icon}&nbsp;&nbsp;{name}</a>'
        nav_html += '</div>'
        st.markdown(nav_html, unsafe_allow_html=True)

        st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

        # ── Key Finding ──
        st.markdown("""
        <p style="font-size:0.63rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;
                  color:#4A6180;margin:0.6rem 0 0.35rem 0.1rem;padding:0;">Key Finding</p>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="width:100%;box-sizing:border-box;background:#0F1E35;border:1px solid #1C2E4A;
                    border-left:3px solid #EF4444;border-radius:8px;padding:0.75rem 0.85rem;">
            <div style="font-size:0.78rem;color:#FCA5A5;font-weight:700;margin-bottom:0.3rem;">No Predictive Signal</div>
            <div style="font-size:0.74rem;color:#7E99C0;line-height:1.65;word-break:break-word;">
                0/23 tests survive correction<br>
                All 10 models: CV R²≈0<br>
                3/3 AutoML confirm<br>
                Data is synthetic
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Leakage Rule ──
        st.markdown("""
        <p style="font-size:0.63rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;
                  color:#4A6180;margin:0.8rem 0 0.35rem 0.1rem;padding:0;">Leakage Rule</p>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="width:100%;box-sizing:border-box;background:#0F1E35;border:1px solid #1C2E4A;
                    border-left:3px solid #F59E0B;border-radius:8px;padding:0.75rem 0.85rem;">
            <div style="font-size:0.74rem;color:#7E99C0;line-height:1.75;word-break:break-word;">
                <code style="background:#152540;padding:2px 6px;border-radius:4px;
                             color:#93C5FD;font-size:0.7rem;">likes_received</code> excluded<br>
                Without: R² = −0.001<br>
                With:&nbsp; R² = <span style="color:#F59E0B;font-weight:700;">0.127</span>&nbsp;(leakage)
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Dataset quick stats ──
        st.markdown("""
        <p style="font-size:0.63rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;
                  color:#4A6180;margin:0.8rem 0 0.35rem 0.1rem;padding:0;">Dataset</p>
        <div style="width:100%;box-sizing:border-box;display:grid;grid-template-columns:1fr 1fr;gap:0.35rem;">
            <div style="background:#101E34;border:1px solid #1C2E4A;border-radius:7px;padding:0.5rem 0.65rem;box-sizing:border-box;">
                <div style="font-size:0.62rem;color:#4A6180;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">Rows</div>
                <div style="font-size:1.05rem;font-weight:700;color:#DDE8F8;line-height:1.2;">50,000</div>
            </div>
            <div style="background:#101E34;border:1px solid #1C2E4A;border-radius:7px;padding:0.5rem 0.65rem;box-sizing:border-box;">
                <div style="font-size:0.62rem;color:#4A6180;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">Features</div>
                <div style="font-size:1.05rem;font-weight:700;color:#DDE8F8;line-height:1.2;">25</div>
            </div>
            <div style="background:#101E34;border:1px solid #1C2E4A;border-radius:7px;padding:0.5rem 0.65rem;box-sizing:border-box;">
                <div style="font-size:0.62rem;color:#4A6180;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">Models</div>
                <div style="font-size:1.05rem;font-weight:700;color:#DDE8F8;line-height:1.2;">10</div>
            </div>
            <div style="background:#101E34;border:1px solid #1C2E4A;border-radius:7px;padding:0.5rem 0.65rem;box-sizing:border-box;">
                <div style="font-size:0.62rem;color:#4A6180;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">Stat Tests</div>
                <div style="font-size:1.05rem;font-weight:700;color:#DDE8F8;line-height:1.2;">23</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    return page


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


# ── Page headers ─────────────────────────────────────────────────────────────

_PAGE_META: dict[str, tuple[str, str, str]] = {
    "Overview": (
        "Swipe Atlas · ML Project",
        "Project Overview",
        "What we set out to find, what every model told us, and why the result — "
        "zero predictive signal — is the honest answer.",
    ),
    "Predict": (
        "Engagement Prediction",
        "Try the Predictor",
        "Enter a user profile and get an estimated engagement score. "
        "The model is trained but predicts near-chance — use this to explore inputs, not to draw conclusions.",
    ),
    "Evidence": (
        "Model Evidence",
        "What the Numbers Show",
        "10 regression models, 3 AutoML frameworks, and 23 statistical tests — "
        "all run on the same data, all arriving at the same answer.",
    ),
    "Analysis": (
        "Exploratory Data Analysis",
        "Data Analysis",
        "How we explored the data before building any models — quality checks, "
        "correlations, the features we engineered, and 23 statistical tests that all came back empty.",
    ),
    "Segments": (
        "User Segmentation",
        "Behavioral Profiles",
        "Explore how users cluster by behavior. "
        "With a silhouette score of 0.019, these are loose descriptive profiles, not tight natural groups.",
    ),
}


def page_header(section: str) -> None:
    eyebrow, title, subtitle = _PAGE_META.get(section, ("", section, ""))
    st.markdown(f"""
    <div class="page-header">
        <span class="page-header-eyebrow">{eyebrow}</span>
        <span class="page-header-title">{title}</span>
        <span class="page-header-subtitle">{subtitle}</span>
    </div>
    """, unsafe_allow_html=True)


# ── Tabs ─────────────────────────────────────────────────────────────────────

def tab_overview(df: pd.DataFrame) -> None:

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
                fig_cols[i % 2].image(str(path), caption=caption, width="stretch")


def tab_predict(df: pd.DataFrame) -> None:
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
            st.image(str(fig_path), width="stretch")
        elif results is not None:
            safe_disp = results[results["setting"] == "official safe"].sort_values("cv_r2_mean", ascending=False)
            st.dataframe(safe_disp[["model", "cv_r2_mean", "cv_mae_mean"]].round(4), width="stretch")

    with col2:
        st.markdown('<span class="section-label">Leakage Demonstration</span>', unsafe_allow_html=True)
        fig_path = FIGDIR / "13_leakage_comparison.png"
        if fig_path.exists():
            st.image(str(fig_path), width="stretch")
        else:
            st.caption("Run script 04 to generate this figure.")

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<span class="section-label">Signal Test — match_outcome</span>', unsafe_allow_html=True)
        fig_path = FIGDIR / "10_signal_summary.png"
        if fig_path.exists():
            st.image(str(fig_path), width="stretch")

    with col4:
        st.markdown('<span class="section-label">AutoML Comparison</span>', unsafe_allow_html=True)
        fig_path = FIGDIR / "20_automl_comparison.png"
        if fig_path.exists():
            st.image(str(fig_path), width="stretch")
        elif automl is not None:
            st.dataframe(automl[["model", "backend", "r2", "mae"]].round(4), width="stretch")

    st.divider()
    st.markdown('<span class="section-label">Learning Curve</span>', unsafe_allow_html=True)
    fig_path = FIGDIR / "14b_learning_curve.png"
    if fig_path.exists():
        st.image(str(fig_path), width="stretch")
        st.caption("R² stays flat as training size increases — more data cannot fix zero signal.")

    with st.expander("Full model results table"):
        if results is not None:
            safe = results[results["setting"] == "official safe"].sort_values("cv_r2_mean", ascending=False)
            st.dataframe(safe[["model","cv_r2_mean","cv_r2_std","cv_mae_mean","cv_rmse_mean"]].round(4), width="stretch")

    with st.expander("AutoML detailed results"):
        if automl is not None:
            st.dataframe(automl[["model","backend","status","r2","mae","rmse"]].round(4), width="stretch")


def tab_analysis(df: pd.DataFrame) -> None:
    audit = load_csv(ROOT / "reports" / "eda_quality_audit.csv")

    # ── Section 1: Data Quality ───────────────────────────────────────────────
    st.markdown('<span class="section-label">Data Quality at a Glance</span>', unsafe_allow_html=True)

    num_cols_count = int((audit["outlier count (IQR rule for numeric)"] != "").sum()) if audit is not None else 12
    invalid_count  = int((audit["invalid values found (yes/no)"] == "yes").sum()) if audit is not None else 0

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.7rem;margin-bottom:1.2rem;">
        <div class="kpi-card neutral">
            <span class="kpi-label">Total Columns</span>
            <span class="kpi-value">25</span>
            <span class="kpi-sub">features + target</span>
        </div>
        <div class="kpi-card green">
            <span class="kpi-label">Missing Values</span>
            <span class="kpi-value">0</span>
            <span class="kpi-sub">across all 50,000 rows</span>
        </div>
        <div class="kpi-card green">
            <span class="kpi-label">Invalid Values</span>
            <span class="kpi-value">{invalid_count}</span>
            <span class="kpi-sub">no inf / empty strings</span>
        </div>
        <div class="kpi-card amber">
            <span class="kpi-label">Outlier Features</span>
            <span class="kpi-value">1</span>
            <span class="kpi-sub">emoji_usage_rate · 311 rows</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Full data quality audit — all 25 columns"):
        if audit is not None:
            st.dataframe(audit, width="stretch")
        else:
            st.caption("Run `python scripts/01_eda.py` to generate this table.")

    st.divider()

    # ── Section 2: Distributions & Correlations ───────────────────────────────
    st.markdown('<span class="section-label">Distributions &amp; Correlations</span>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <p style="font-size:0.8rem;color:var(--txt2);margin-bottom:0.5rem;">
        <b style="color:var(--txt);">Numeric distributions</b> — all 12 numeric features are
        uniformly spread with nearly identical mean and median lines. This is a strong
        fingerprint of synthetic data: real distributions are skewed and lumpy.
        </p>""", unsafe_allow_html=True)
        p = FIGDIR / "02_numeric_distributions.png"
        if p.exists():
            st.image(str(p), width="stretch")

    with col2:
        st.markdown("""
        <p style="font-size:0.8rem;color:var(--txt2);margin-bottom:0.5rem;">
        <b style="color:var(--txt);">Correlation heatmap</b> — only two pairs show |r| ≥ 0.05:
        <code>height_cm ↔ weight_kg</code> (physical, expected) and
        <code>likes_received ↔ mutual_matches</code> (which revealed the leakage risk).
        Everything else is noise-level.
        </p>""", unsafe_allow_html=True)
        p = FIGDIR / "03_correlation_heatmap.png"
        if p.exists():
            st.image(str(p), width="stretch")

    st.divider()

    # ── Section 3: Engineered Features ───────────────────────────────────────
    st.markdown('<span class="section-label">Features We Engineered</span>', unsafe_allow_html=True)
    if False:
        st.markdown("\n".join(line.lstrip() for line in dedent("""
    <p style="font-size:0.85rem;color:var(--txt2);margin-bottom:1rem;">
    We added 6 behavioural features on top of the 25 raw columns. None improved CV R² —
    confirming the absence of signal rather than a feature-engineering gap.
    </p>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.7rem;margin-bottom:0.5rem;">

        <div style="background:var(--card);border:1px solid var(--border);border-left:3px solid var(--blue);
                    border-radius:9px;padding:0.85rem 1rem;">
            <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;
                        color:var(--blue);margin-bottom:0.35rem;">bio_effort</div>
            <code style="font-size:0.75rem;color:var(--txt);background:var(--card2);padding:2px 6px;
                         border-radius:4px;">bio_length × profile_pics_count</code>
            <div style="font-size:0.78rem;color:var(--txt2);margin-top:0.4rem;line-height:1.55;">
                Profile completeness proxy — users who write more and upload more photos may signal higher intent.
            </div>
        </div>

        <div style="background:var(--card);border:1px solid var(--border);border-left:3px solid var(--blue);
                    border-radius:9px;padding:0.85rem 1rem;">
            <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;
                        color:var(--blue);margin-bottom:0.35rem;">night_user</div>
            <code style="font-size:0.75rem;color:var(--txt);background:var(--card2);padding:2px 6px;
                         border-radius:4px;">last_active_hour ≥ 22 or ≤ 4 → 1</code>
            <div style="font-size:0.78rem;color:var(--txt2);margin-top:0.4rem;line-height:1.55;">
                Binary flag for late-night activity — different usage patterns might correlate with match behaviour.
            </div>
        </div>

        <div style="background:var(--card);border:1px solid var(--border);border-left:3px solid var(--blue);
                    border-radius:9px;padding:0.85rem 1rem;">
            <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;
                        color:var(--blue);margin-bottom:0.35rem;">emoji_heavy</div>
            <code style="font-size:0.75rem;color:var(--txt);background:var(--card2);padding:2px 6px;
                         border-radius:4px;">emoji_usage_rate &gt; 0.5 → 1</code>
            <div style="font-size:0.78rem;color:var(--txt2);margin-top:0.4rem;line-height:1.55;">
                Communication style flag — heavy emoji use may indicate a more expressive, approachable persona.
            </div>
        </div>

        <div style="background:var(--card);border:1px solid var(--border);border-left:3px solid var(--amber);
                    border-radius:9px;padding:0.85rem 1rem;">
            <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;
                        color:var(--amber);margin-bottom:0.35rem;">likes_per_usage_min</div>
            <code style="font-size:0.75rem;color:var(--txt);background:var(--card2);padding:2px 6px;
                         border-radius:4px;">likes_received ÷ app_usage_time_min</code>
            <div style="font-size:0.78rem;color:var(--txt2);margin-top:0.4rem;line-height:1.55;">
                Engagement efficiency — likes earned per minute of app use, normalising for session length.
            </div>
        </div>

        <div style="background:var(--card);border:1px solid var(--border);border-left:3px solid var(--amber);
                    border-radius:9px;padding:0.85rem 1rem;">
            <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;
                        color:var(--amber);margin-bottom:0.35rem;">messages_per_match</div>
            <code style="font-size:0.75rem;color:var(--txt);background:var(--card2);padding:2px 6px;
                         border-radius:4px;">message_sent_count ÷ mutual_matches</code>
            <div style="font-size:0.78rem;color:var(--txt2);margin-top:0.4rem;line-height:1.55;">
                Conversation rate — how many messages a user sends per match, capturing follow-through behaviour.
            </div>
        </div>

        <div style="background:var(--card);border:1px solid var(--border);border-left:3px solid var(--amber);
                    border-radius:9px;padding:0.85rem 1rem;">
            <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;
                        color:var(--amber);margin-bottom:0.35rem;">match_yield_from_likes</div>
            <code style="font-size:0.75rem;color:var(--txt);background:var(--card2);padding:2px 6px;
                         border-radius:4px;">mutual_matches ÷ likes_received</code>
            <div style="font-size:0.78rem;color:var(--txt2);margin-top:0.4rem;line-height:1.55;">
                Conversion rate — what fraction of incoming likes turn into mutual matches.
            </div>
        </div>

    </div>
    <p style="font-size:0.78rem;color:var(--txt3);margin-top:0.5rem;">
        Blue = profile / behaviour flags &nbsp;·&nbsp; Amber = ratio / efficiency features.
        All 6 excluded the active target from their denominators to prevent leakage.
    </p>
    """).strip().splitlines()), unsafe_allow_html=True)

    st.write(
        "We added 6 behavioural features on top of the 25 raw columns. "
        "None improved CV R^2, confirming the absence of signal rather "
        "than a feature-engineering gap."
    )

    feature_cards = [
        (
            "bio_effort",
            "bio_length x profile_pics_count",
            "Profile completeness proxy: users who write more and upload more photos may signal higher intent.",
            "Profile / behaviour flag",
        ),
        (
            "night_user",
            "last_active_hour >= 22 or <= 4 -> 1",
            "Binary flag for late-night activity: different usage patterns might correlate with match behaviour.",
            "Profile / behaviour flag",
        ),
        (
            "emoji_heavy",
            "emoji_usage_rate > 0.5 -> 1",
            "Communication style flag: heavy emoji use may indicate a more expressive, approachable persona.",
            "Profile / behaviour flag",
        ),
        (
            "likes_per_usage_min",
            "likes_received / app_usage_time_min",
            "Engagement efficiency: likes earned per minute of app use, normalising for session length.",
            "Ratio / efficiency feature",
        ),
        (
            "messages_per_match",
            "message_sent_count / mutual_matches",
            "Conversation rate: how many messages a user sends per match, capturing follow-through behaviour.",
            "Ratio / efficiency feature",
        ),
        (
            "match_yield_from_likes",
            "mutual_matches / likes_received",
            "Conversion rate: what fraction of incoming likes turn into mutual matches.",
            "Ratio / efficiency feature",
        ),
    ]

    for start in range(0, len(feature_cards), 3):
        cols = st.columns(3)
        for col, (name, formula, description, feature_type) in zip(cols, feature_cards[start:start + 3]):
            with col.container(border=True):
                st.markdown(f"**{name}**")
                st.code(formula, language=None)
                st.caption(feature_type)
                st.write(description)

    st.caption(
        "Profile / behaviour flags and ratio / efficiency features all excluded "
        "the active target from their denominators to prevent leakage."
    )

    st.divider()

    # ── Section 4: Statistical Tests ──────────────────────────────────────────
    st.markdown('<span class="section-label">Statistical Tests — 23 Total, 0 Survive Correction</span>', unsafe_allow_html=True)

    chi_data = pd.DataFrame([
        ("gender",             69.85, 45,  0.0102, True),
        ("sexual_orientation", 80.44, 63,  0.0684, False),
        ("zodiac_sign",       107.98, 99,  0.2525, False),
        ("education_level",    71.50, 72,  0.4943, False),
        ("income_bracket",     52.20, 54,  0.5442, False),
        ("body_type",          41.87, 45,  0.6055, False),
        ("relationship_intent",40.56, 45,  0.6604, False),
        ("swipe_time_of_day",  41.22, 45,  0.6330, False),
        ("swipe_right_label",  23.01, 27,  0.6842, False),
        ("app_usage_time_label",30.96,54,  0.9950, False),
        ("location_type",      29.30, 45,  0.9662, False),
    ], columns=["Feature", "χ²", "dof", "p-value", "Nom. sig."])

    anova_data = pd.DataFrame([
        ("height_cm",          1.377, 0.1920, False),
        ("weight_kg",          1.628, 0.1009, False),
        ("swipe_right_ratio",  1.446, 0.1621, False),
        ("mutual_matches",     0.909, 0.5156, False),
        ("message_sent_count", 0.836, 0.5825, False),
        ("last_active_hour",   0.830, 0.5884, False),
        ("likes_received",     0.777, 0.6374, False),
        ("profile_pics_count", 0.703, 0.7067, False),
        ("emoji_usage_rate",   0.705, 0.7050, False),
        ("app_usage_time_min", 0.655, 0.7506, False),
        ("age",                0.570, 0.8228, False),
        ("bio_length",         0.222, 0.9915, False),
    ], columns=["Feature", "F-stat", "p-value", "Nom. sig."])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <p style="font-size:0.8rem;color:var(--txt2);margin-bottom:0.5rem;">
        <b style="color:var(--txt);">Chi-square</b> — categorical features vs <code>match_outcome</code>.
        Only <code>gender</code> reached p &lt; 0.05 (uncorrected), but p = 0.0102 &gt; Bonferroni
        threshold of 0.00455 — so even that one fails correction.
        </p>""", unsafe_allow_html=True)
        chi_disp = chi_data.copy()
        chi_disp["p-value"] = chi_disp["p-value"].apply(lambda x: f"{x:.4f}")
        chi_disp["χ²"]      = chi_disp["χ²"].apply(lambda x: f"{x:.2f}")
        chi_disp["Nom. sig."] = chi_disp["Nom. sig."].map({True: "✓ p<0.05", False: "—"})
        st.dataframe(chi_disp, width="stretch", hide_index=True)

    with col2:
        st.markdown("""
        <p style="font-size:0.8rem;color:var(--txt2);margin-bottom:0.5rem;">
        <b style="color:var(--txt);">One-way ANOVA</b> — numeric features vs <code>match_outcome</code>.
        Zero features reach p &lt; 0.05 even before any correction. F-statistics are all
        near 1.0, indicating group means are indistinguishable.
        </p>""", unsafe_allow_html=True)
        anova_disp = anova_data.copy()
        anova_disp["p-value"] = anova_disp["p-value"].apply(lambda x: f"{x:.4f}")
        anova_disp["F-stat"]  = anova_disp["F-stat"].apply(lambda x: f"{x:.3f}")
        anova_disp["Nom. sig."] = anova_disp["Nom. sig."].map({True: "✓ p<0.05", False: "—"})
        st.dataframe(anova_disp, width="stretch", hide_index=True)

    # Correction summary
    st.markdown("""
    <div style="background:var(--card);border:1px solid var(--border);border-left:4px solid var(--red);
                border-radius:9px;padding:1rem 1.2rem;margin-top:0.8rem;">
        <div style="font-size:0.78rem;font-weight:700;color:#FCA5A5;margin-bottom:0.5rem;
                    text-transform:uppercase;letter-spacing:0.07em;">Multiple-Testing Correction Results</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.5rem;">
            <div style="font-size:0.82rem;color:var(--txt2);">
                <b style="color:var(--txt);">Total tests:</b> 11 chi-square + 12 ANOVA = <b style="color:var(--txt);">23</b>
            </div>
            <div style="font-size:0.82rem;color:var(--txt2);">
                <b style="color:var(--txt);">Bonferroni</b> threshold: α/11 = 0.00455 →
                <b style="color:#FCA5A5;">0 / 11</b> survive
            </div>
            <div style="font-size:0.82rem;color:var(--txt2);">
                <b style="color:var(--txt);">BH-FDR</b> (q = 0.05, all 23 tests) →
                <b style="color:#FCA5A5;">0 / 23</b> survive
            </div>
        </div>
        <div style="font-size:0.8rem;color:var(--txt3);margin-top:0.6rem;line-height:1.6;">
            Corrected conclusion: <b style="color:var(--txt2);">zero of 23 feature–target tests are statistically significant</b>.
            This is consistent with a synthetic data generator that samples each column independently of the target.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Section 5: Target distribution ───────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<span class="section-label">Target Distribution</span>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size:0.8rem;color:var(--txt2);margin-bottom:0.5rem;">
        10 outcome classes with a balance ratio of <b style="color:var(--txt);">1.000</b> — each class has
        almost exactly 5,000 rows. Perfect balance in a real dataset is extraordinarily unlikely.
        It is the clearest fingerprint that labels were assigned synthetically.
        </p>""", unsafe_allow_html=True)
        p = FIGDIR / "01_target_distribution.png"
        if p.exists():
            st.image(str(p), width="stretch")
    with col2:
        st.markdown('<span class="section-label">Feature Means by Outcome</span>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size:0.8rem;color:var(--txt2);margin-bottom:0.5rem;">
        Numeric feature means grouped by match outcome. If any feature predicted the target,
        you would see clear separation between the coloured lines. All lines overlap completely.
        </p>""", unsafe_allow_html=True)
        p = FIGDIR / "06_means_by_outcome.png"
        if p.exists():
            st.image(str(p), width="stretch")


def tab_segments(df: pd.DataFrame) -> None:
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
            st.image(str(umap_path), width="stretch")
            st.caption("Diffuse, overlapping clusters confirm weak structure.")

    with col2:
        st.markdown('<span class="section-label">Segment Profiles</span>', unsafe_allow_html=True)
        profile_path = FIGDIR / "17_segment_profiles.png"
        if profile_path.exists():
            st.image(str(profile_path), width="stretch")

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<span class="section-label">K Selection — Silhouette Sweep</span>', unsafe_allow_html=True)
        k_path = FIGDIR / "15_kmeans_selection.png"
        if k_path.exists():
            st.image(str(k_path), width="stretch")

    with col4:
        st.markdown('<span class="section-label">GMM BIC / AIC</span>', unsafe_allow_html=True)
        gmm_path = FIGDIR / "18_gmm_bic_aic.png"
        if gmm_path.exists():
            st.image(str(gmm_path), width="stretch")
            st.caption("Monotone BIC/AIC decrease — no preferred cluster count.")

    if summary is not None:
        st.divider()
        st.markdown('<span class="section-label">Segment Summary Table</span>', unsafe_allow_html=True)
        disp = summary.copy()
        if "segment_id" in disp.columns:
            disp = disp.drop(columns=["segment_id"])
        st.dataframe(disp.round(2), width="stretch")


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

    # Compact breadcrumb
    _section_labels = {
        "Overview": "Project Overview",
        "Analysis": "Data Analysis",
        "Predict":  "Engagement Prediction",
        "Evidence": "Model Evidence",
        "Segments": "User Segmentation",
    }
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1.2rem;padding-bottom:0.45rem;border-bottom:1px solid #1C2E4A;">
        <span style="font-size:0.7rem;color:#4A6180;font-weight:700;text-transform:uppercase;letter-spacing:0.09em;">Swipe Atlas</span>
        <span style="color:#1C2E4A;font-size:0.85rem;line-height:1;">›</span>
        <span style="font-size:0.7rem;color:#7E99C0;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;">{_section_labels.get(section, section)}</span>
    </div>
    """, unsafe_allow_html=True)

    page_header(section)

    kpi_strip(df)

    if section == "Overview":
        tab_overview(df)
    elif section == "Analysis":
        tab_analysis(df)
    elif section == "Predict":
        tab_predict(df)
    elif section == "Evidence":
        tab_evidence()
    elif section == "Segments":
        tab_segments(df)


if __name__ == "__main__":
    main()
