# =============================================================================
# app.py
# Equally Accurate, Unequally Fair
# COMPAS Rashomon Fairness Audit — Interactive Streamlit Application
# =============================================================================
#
# ── HOW TO RUN ───────────────────────────────────────────────────────────────
#
#   streamlit run app.py
#
# ── WHAT THIS APP DOES ───────────────────────────────────────────────────────
#
# Five pages that tell one complete argument:
#
#   Page 1: The Case        — The ProPublica story + our key finding
#   Page 2: The Audit       — The central scatter plot: accuracy vs bias
#   Page 3: Model Explorer  — Drill into any model's decision logic
#   Page 4: Who Bears It    — Defendant impact + impossibility theorem
#   Page 5: About           — Methodology, citations, limitations
#
# ── DATA LOADING STRATEGY ────────────────────────────────────────────────────
#
# We use Streamlit's @st.cache_data to avoid re-running expensive computations
# on every user interaction. The loading sequence is:
#
#   load_fast()                   ~1 sec  — download and clean COMPAS data
#   load_rashomon(_X, _y, _race)  ~5 sec  — train 49 models (ONCE ever)
#   load_fairness(_result)        ~2 sec  — compute all fairness metrics (ONCE)
#
# Total first load: ~8 seconds. Every subsequent interaction: instant.
# The underscore prefix (_X, _y, etc.) tells Streamlit NOT to hash those
# arguments — sklearn objects and DataFrames are not hashable by Streamlit.
#
# ── AESTHETIC: "EVIDENCE ROOM" ───────────────────────────────────────────────
#
# Warm off-white background. Serif headings. Deep charcoal text.
# Terracotta red for high bias. Forest green for low bias. Steel blue accents.
# Feels like: a courtroom exhibit. A published audit. Investigative journalism.
# Deliberately unlike P1 (cold navy) and P2 (dark editorial).
#
# ── CITING HER WORK ──────────────────────────────────────────────────────────
#
# This project is an interactive demonstration of:
# Semenova, Hsu, Chen, Zhong — "The Double-Edged Nature of the Rashomon Set
# for Trustworthy Machine Learning" (arXiv 2025)
#
# =============================================================================

import streamlit as st
import pandas    as pd
import numpy     as np
import plotly.graph_objects as go
import sys
import os

# Add project root to path so we can import our audit package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audit.data     import load_data, get_dataset_summary, get_race_stats
from audit.rashomon import build_rashomon_set, summarise_rashomon_set
from audit.fairness import compute_all_fairness, compute_defendant_disagreement
from audit.analyze  import (get_scatter_data, compute_pareto_frontier,
                             get_summary_stats, get_model_detail,
                             get_impossibility_data, MODEL_TYPE_COLORS)


# =============================================================================
# PAGE CONFIG
# Must be the FIRST Streamlit call in the script — nothing can come before it.
# =============================================================================

st.set_page_config(
    page_title = "Equally Accurate, Unequally Fair",
    page_icon  = "⚖",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)


# =============================================================================
# CSS — THE EVIDENCE ROOM AESTHETIC
# =============================================================================
# We inject a <style> block into the page using st.markdown(unsafe_allow_html).
# This is the standard way to customise Streamlit's appearance.
#
# Design tokens (CSS variables) keep colors and fonts consistent.
# Change a token here and it updates everywhere automatically.

st.markdown("""
<style>
/* ── Google Fonts ──────────────────────────────────────────────────────── */
/* Libre Baskerville: authoritative serif for headings (like a report title) */
/* IBM Plex Mono: precise monospace for numbers and data values              */
/* Inter: clean sans-serif for body text (readable at small sizes)          */
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Mono:wght@300;400;500&family=Inter:wght@300;400;500;600&display=swap');

/* ── Design Tokens (CSS variables) ────────────────────────────────────── */
:root {
    --bg:           #fafaf8;   /* warm off-white — main background          */
    --bg-card:      #ffffff;   /* pure white — card backgrounds             */
    --bg-raised:    #f4f4f0;   /* slightly darker — code blocks, alt rows   */
    --border:       #e5e5e0;   /* light warm gray — card borders            */
    --border-strong:#c8c8c2;   /* slightly darker border for emphasis       */
    --text:         #1a1a2e;   /* deep charcoal — main text                 */
    --text-muted:   #52525e;   /* medium gray — secondary text              */
    --text-faint:   #8a8a96;   /* light gray — captions, footnotes          */
    --danger:       #b83225;   /* terracotta red — high bias, warnings      */
    --danger-bg:    #fdf2f1;   /* light red tint — danger card backgrounds  */
    --danger-border:#e8b4b0;   /* danger card borders                       */
    --safe:         #1a6b3c;   /* forest green — low bias, positive         */
    --safe-bg:      #f0f9f4;   /* light green tint — safe card backgrounds  */
    --safe-border:  #a8dbc0;   /* safe card borders                         */
    --accent:       #2c5f8a;   /* steel blue — links, highlights, accents   */
    --accent-bg:    #eef4fa;   /* light blue tint                           */
    --accent-border:#a8c4de;   /* accent card borders                       */
    --amber:        #b8770c;   /* amber — warnings, outside-set models      */
    --amber-bg:     #fdf8ed;   /* light amber tint                          */
    --font-head:    'Libre Baskerville', Georgia, serif;
    --font-mono:    'IBM Plex Mono', 'Courier New', monospace;
    --font-body:    'Inter', system-ui, sans-serif;
}

/* ── Global Reset ──────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}
[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
/* Hide Streamlit's default menu bar, footer, and top decoration */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"]  { display: none; }

/* ── Typography ────────────────────────────────────────────────────────── */
.headline {
    font-family: var(--font-head);
    font-size: 42px;
    font-weight: 700;
    color: #1a1a2e;          /* light text for dark hero background */
    line-height: 1.15;
    margin-bottom: 8px;
}
.headline-italic {
    font-family: var(--font-head);
    font-size: 42px;
    font-style: italic;
    font-weight: 400;
    color: #3a3a6a;
    line-height: 1.15;
}
.section-title {
    font-family: var(--font-head);
    font-size: 24px;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 6px 0;
}
/* Eyebrow: small uppercase label above a section title */
.eyebrow {
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--accent);
    display: block;
    margin-bottom: 8px;
}
.body-text {
    font-family: var(--font-body);
    font-size: 14px;
    line-height: 1.8;
    color: var(--text-muted);
}

/* ── Cards ─────────────────────────────────────────────────────────────── */
/* Plain white card with a subtle border */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 14px;
}
/* Coloured left-border cards for emphasis */
.card-danger {
    background: var(--danger-bg);
    border: 1px solid var(--danger-border);
    border-left: 4px solid var(--danger);
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin-bottom: 14px;
}
.card-safe {
    background: var(--safe-bg);
    border: 1px solid var(--safe-border);
    border-left: 4px solid var(--safe);
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin-bottom: 14px;
}
.card-accent {
    background: var(--accent-bg);
    border: 1px solid var(--accent-border);
    border-left: 4px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin-bottom: 14px;
}
.card-amber {
    background: var(--amber-bg);
    border: 1px solid #e8d4a0;
    border-left: 4px solid var(--amber);
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin-bottom: 14px;
}

/* ── Stat Boxes ─────────────────────────────────────────────────────────── */
/* Large number + small label, used in a row of 5 on the home page */
.stat-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 18px 14px;
    text-align: center;
}
.stat-value {
    font-family: var(--font-mono);
    font-size: 28px;
    font-weight: 500;
    display: block;
    line-height: 1.1;
}
.stat-label {
    font-family: var(--font-body);
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-top: 5px;
    display: block;
}
/* Color variants for stat values */
.stat-value.danger { color: var(--danger); }
.stat-value.safe   { color: var(--safe);   }
.stat-value.accent { color: var(--accent); }
.stat-value.amber  { color: var(--amber);  }
.stat-value.dark   { color: var(--text);   }

/* ── Pullquote ──────────────────────────────────────────────────────────── */
.pullquote {
    font-family: var(--font-head);
    font-size: 19px;
    font-style: italic;
    color: var(--text);
    border-left: 4px solid var(--danger);
    padding-left: 20px;
    margin: 20px 0;
    line-height: 1.6;
}

/* ── Horizontal divider ─────────────────────────────────────────────────── */
.divider {
    height: 1px;
    background: var(--border);
    margin: 28px 0;
}

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar       { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-strong);
                             border-radius: 3px; }

/* ── Lock the sidebar open ───────────────────────────────────────────────── */
/* We hide the collapse button INSIDE the sidebar (the < arrow on its edge).
   This prevents the sidebar from ever being collapsed.
   We do NOT hide the expand button in the main content, so it is always
   recoverable if needed. */
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}
/* Belt-and-suspenders: also target the button by its aria label */
button[aria-label="Close sidebar"] {
    display: none !important;
}
                        
</style>
""", unsafe_allow_html=True)


# =============================================================================
# PLOTLY CHART THEME
# =============================================================================
# All charts use white backgrounds and minimal styling to look like
# figures from an academic paper or audit report — not a dashboard.

def _apply_chart_theme(fig, height=420, title_text=None):
    """
    Applies the Evidence Room chart theme to any Plotly figure.
    Call this on every chart before st.plotly_chart().
    """
    fig.update_layout(
        paper_bgcolor = "#ffffff",
        plot_bgcolor  = "#ffffff",
        height        = height,
        font          = dict(family="IBM Plex Mono, monospace",
                             color="#52525e", size=11),
        margin        = dict(l=55, r=20, t=50, b=55),
        hoverlabel    = dict(bgcolor="#1a1a2e", bordercolor="#1a1a2e",
                             font=dict(family="IBM Plex Mono",
                                       color="#f8f8f4", size=11)),
        legend        = dict(bgcolor="rgba(255,255,255,0.9)",
                             bordercolor="#e5e5e0", borderwidth=1,
                             font=dict(color="#52525e", size=10)),
    )
    if title_text:
        fig.update_layout(title=dict(
            text=title_text,
            font=dict(family="Libre Baskerville, Georgia, serif",
                      size=14, color="#1a1a2e"),
        ))
    fig.update_xaxes(
        gridcolor="#eeeeea", linecolor="#c8c8c2",
        tickfont=dict(color="#52525e", size=10), zeroline=False,
    )
    fig.update_yaxes(
        gridcolor="#eeeeea", linecolor="#c8c8c2",
        tickfont=dict(color="#52525e", size=10), zeroline=False,
    )
    return fig


# =============================================================================
# DATA LOADING WITH CACHING
# =============================================================================

@st.cache_data(show_spinner=False)
def load_fast():
    """
    Loads and cleans the COMPAS dataset. (~1 second, cached permanently.)
    Called once on first load, then served from cache every time.
    """
    X, y, df, race = load_data(verbose=False)
    summary        = get_dataset_summary(df, X, y, race)
    race_stats     = get_race_stats(df, race)
    return X, y, df, race, summary, race_stats


@st.cache_data(show_spinner=False)
def load_rashomon(_X, _y, _race):
    """
    Trains 49 models and builds the Rashomon set. (~5 seconds, cached.)

    The underscore prefix on arguments tells Streamlit: "do not try to
    hash these objects for the cache key." sklearn DataFrames and Series
    are not hashable by Streamlit. We accept that the cache key is based
    on function identity only — this is safe because the data never changes.
    """
    result  = build_rashomon_set(_X, _y, _race, verbose=False)
    summary = summarise_rashomon_set(result)
    return result, summary


@st.cache_data(show_spinner=False)
def load_fairness(_result):
    """
    Computes all fairness metrics. (~2 seconds, cached.)
    Returns everything the app pages need.
    """
    fairness_df = compute_all_fairness(_result, verbose=False)
    scatter_df  = get_scatter_data(fairness_df)
    stats       = get_summary_stats(fairness_df, _result)
    disagree    = compute_defendant_disagreement(fairness_df, _result)
    impossib    = get_impossibility_data(fairness_df)
    pareto      = compute_pareto_frontier(fairness_df)
    return fairness_df, scatter_df, stats, disagree, impossib, pareto


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

def render_sidebar(stats):
    """
    Renders the sidebar: branding, navigation buttons, and key stats.
    """
    with st.sidebar:

        # ── Project branding ───────────────────────────────────────────────
        st.markdown("""
        <div style="padding: 4px 0 20px; border-bottom: 1px solid #e5e5e0;
                    margin-bottom: 20px;">
            <div style="font-family: 'Libre Baskerville', Georgia, serif;
                        font-size: 16px; font-weight: 700; color: #1a1a2e;
                        line-height: 1.35;">
                Equally Accurate,<br>Unequally Fair
            </div>
            <div style="font-family: 'IBM Plex Mono', monospace;
                        font-size: 9px; color: #2c5f8a; margin-top: 6px;
                        text-transform: uppercase; letter-spacing: 0.13em;">
                COMPAS · Broward County FL · 6,172 defendants
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Navigation ─────────────────────────────────────────────────────
        # Each button sets session_state["page"] and triggers a rerun.
        st.markdown('<span class="eyebrow">Navigate</span>',
                    unsafe_allow_html=True)

        pages = [
            ("⚖",  "The Case",       "The story & our finding"),
            ("◎",  "The Audit",      "49 models: accuracy vs bias"),
            ("↳",  "Model Explorer", "Drill into any model"),
            ("◈",  "Who Bears It",   "Defendants & the theorem"),
            ("○",  "About",          "Methodology & citations"),
        ]

        current_page = st.session_state.get("page", "The Case")

        for icon, name, desc in pages:
            # Primary button for active page, secondary for others
            btn_type = "primary" if current_page == name else "secondary"
            if st.button(f"{icon}  {name}", key=f"nav_{name}",
                         type=btn_type, use_container_width=True):
                st.session_state["page"] = name
                st.rerun()

        # ── Live key stats (only shown after fairness metrics load) ───────
        if stats:
            st.markdown("""
            <div style="margin-top: 24px; padding-top: 18px;
                        border-top: 1px solid #e5e5e0;">
                <span class="eyebrow">Key Finding</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="stat-box" style="margin-bottom: 6px;">
                <span class="stat-value danger">
                    {stats['fpr_gap_max']:.3f}</span>
                <span class="stat-label">Worst FPR bias gap</span>
            </div>
            <div class="stat-box" style="margin-bottom: 6px;">
                <span class="stat-value safe">
                    {stats['fpr_gap_min']:.3f}</span>
                <span class="stat-label">Best FPR bias gap</span>
            </div>
            <div class="stat-box">
                <span class="stat-value accent">
                    {stats['auc_cost_of_fairness']:.4f}</span>
                <span class="stat-label">AUC cost of fairness</span>
            </div>
            """, unsafe_allow_html=True)

        # ── Footer ─────────────────────────────────────────────────────────
        st.markdown("""
        <div style="margin-top: 28px; padding-top: 14px;
                    border-top: 1px solid #eeeeea;">
            <div style="font-size: 9px; color: #b0b0b8; line-height: 2;
                        font-family: 'IBM Plex Mono', monospace;">
                Data: ProPublica (2016)<br>
                Theory: Semenova et al. 2025<br>
                ε-Rashomon · COMPAS
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 1 — THE CASE
# =============================================================================

def page_the_case(summary, race_stats, stats, rashomon_summary):
    """
    The landing page. Sets the scene with the ProPublica story,
    then immediately shows our key finding.
    """

    # ── Dark hero banner ───────────────────────────────────────────────────
    # This dark section grabs attention and frames the project immediately.
    st.markdown("""
    <div style="background: #c6cbf5; padding: 44px 44px 40px;
                margin-bottom: 36px; border-radius: 8px;">
        <span style="font-family: 'IBM Plex Mono', monospace; font-size: 10px;
                     color: #333436; text-transform: uppercase;
                     letter-spacing: 0.16em; display: block; margin-bottom: 14px;">
            Algorithmic Fairness · Rashomon Sets · Criminal Justice
        </span>
        <div class="headline">Equally Accurate,</div>
        <div class="headline-italic">Unequally Fair</div>
        <div style="font-size: 15px; color: #333436; margin-top: 16px;
                    max-width: 640px; line-height: 1.85;
                    font-family: 'Inter', sans-serif;">
            In Florida, an algorithm called COMPAS decided whether defendants
            received bail. ProPublica found it was racially biased.
            But they only looked at one model.
            <strong style="color: #333436;">
                We looked at all 40 equally accurate models.</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Five headline statistics ───────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""<div class="stat-box">
            <span class="stat-value dark">{summary['total_defendants']:,}</span>
            <span class="stat-label">Defendants</span>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="stat-box">
            <span class="stat-value accent">
                {rashomon_summary['n_in_set']}</span>
            <span class="stat-label">Equally-Good Models</span>
        </div>""", unsafe_allow_html=True)
    with c3:
        if stats:
            st.markdown(f"""<div class="stat-box">
                <span class="stat-value danger">
                    {stats['fpr_gap_max']:.3f}</span>
                <span class="stat-label">Worst Bias Gap</span>
            </div>""", unsafe_allow_html=True)
    with c4:
        if stats:
            st.markdown(f"""<div class="stat-box">
                <span class="stat-value safe">
                    {stats['fpr_gap_min']:.3f}</span>
                <span class="stat-label">Best Bias Gap</span>
            </div>""", unsafe_allow_html=True)
    with c5:
        if stats:
            st.markdown(f"""<div class="stat-box">
                <span class="stat-value amber">
                    {stats['auc_cost_of_fairness']:.4f}</span>
                <span class="stat-label">AUC Cost of Fairness</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two columns: story (left) + data context (right) ──────────────────
    col_story, col_data = st.columns([3, 2], gap="large")

    with col_story:
        st.markdown('<span class="eyebrow">The Story</span>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <p style="font-family:'Libre Baskerville',serif; font-size:18px;
                      color:#1a1a2e; line-height:1.65; margin-bottom:14px;">
                A judge in Florida uses an algorithm to decide bail.
                Two defendants with similar records appear before the court.
                One is Black. One is White.
            </p>
            <p class="body-text" style="margin-bottom: 12px;">
                The algorithm — COMPAS — outputs a risk score from 1 to 10.
                High score means "likely to reoffend." High score means
                less likely to get bail. The judge sees the score.
                The defendant does not know how it was calculated.
            </p>
            <p class="body-text" style="margin-bottom: 12px;">
                In 2016, ProPublica investigated 7,000 defendants in
                Broward County and found that COMPAS falsely labelled
                Black defendants as high-risk at
                <strong style="color:#1a1a2e;">nearly twice the rate</strong>
                of white defendants. Among people who did not reoffend,
                44.7% of Black defendants were labelled dangerous
                versus 23.2% of white defendants.
            </p>
            <p class="body-text">
                Northpointe, the company behind COMPAS, said the algorithm
                was fair by a different measure. Both sides were technically
                correct. The debate became one of the most famous in
                AI ethics.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="pullquote">
            "But both sides were arguing about a single deployed model.
            We asked a deeper question: within all the equally-accurate
            models that could have been deployed, how much did racial
            bias vary?"
        </div>
        """, unsafe_allow_html=True)

        if stats:
            st.markdown(f"""
            <div class="card-safe">
                <div style="font-family:'Libre Baskerville',serif; font-size:17px;
                            color:var(--safe); margin-bottom:8px; font-weight:700;">
                    Our finding</div>
                <p class="body-text">
                    Within the Rashomon set — the
                    <strong style="color:#1a1a2e;">
                    {rashomon_summary['n_in_set']} equally-accurate models</strong>
                    — the racial bias gap varies from
                    <strong style="color:var(--safe);">
                    {stats['fpr_gap_min']:.3f}</strong> to
                    <strong style="color:var(--danger);">
                    {stats['fpr_gap_max']:.3f}</strong>.
                    Choosing the fairest model costs just
                    <strong style="color:#1a1a2e;">
                    {stats['auc_cost_of_fairness']:.4f} AUC points</strong>
                    — a difference invisible by any standard benchmark.
                    The bias was not inevitable. It was a choice made
                    by omission.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with col_data:
        st.markdown('<span class="eyebrow">The Data</span>',
                    unsafe_allow_html=True)

        bw = race_stats.get("African-American", {})
        wh = race_stats.get("Caucasian", {})

        st.markdown(f"""
        <div class="card">
            <div style="font-family:'Libre Baskerville',serif; font-size:14px;
                        font-weight:700; color:#1a1a2e; margin-bottom:14px;">
                {summary['total_defendants']:,} defendants ·
                Broward County, FL
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr;
                        gap:10px; margin-bottom:14px;">
                <div style="background:var(--danger-bg);
                            border:1px solid var(--danger-border);
                            border-radius:6px; padding:12px; text-align:center;">
                    <div style="font-family:'IBM Plex Mono',monospace;
                                font-size:22px; color:var(--danger);">
                        {bw.get('count',0):,}</div>
                    <div style="font-size:10px; color:var(--text-faint);
                                margin-top:4px; text-transform:uppercase;
                                letter-spacing:.08em;">Black defendants</div>
                    <div style="font-size:12px; color:var(--text-muted);
                                margin-top:3px;">
                        recid rate: {bw.get('recid_rate',0):.1f}%</div>
                </div>
                <div style="background:var(--accent-bg);
                            border:1px solid var(--accent-border);
                            border-radius:6px; padding:12px; text-align:center;">
                    <div style="font-family:'IBM Plex Mono',monospace;
                                font-size:22px; color:var(--accent);">
                        {wh.get('count',0):,}</div>
                    <div style="font-size:10px; color:var(--text-faint);
                                margin-top:4px; text-transform:uppercase;
                                letter-spacing:.08em;">White defendants</div>
                    <div style="font-size:12px; color:var(--text-muted);
                                margin-top:3px;">
                        recid rate: {wh.get('recid_rate',0):.1f}%</div>
                </div>
            </div>
            <div style="font-size:11px; color:var(--text-faint);
                        border-top:1px solid var(--border); padding-top:10px;
                        line-height:1.7;">
                The 13.2 percentage point base rate difference (52.3% vs 39.1%)
                is the reason the ProPublica vs Northpointe debate is
                mathematically irresolvable — Chouldechova's impossibility
                theorem explains why. See "Who Bears It" for the full story.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ProPublica reference — the benchmark we compare against
        st.markdown(f"""
        <div class="card-danger">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:9px;
                        color:var(--danger); text-transform:uppercase;
                        letter-spacing:.12em; margin-bottom:10px;">
                ProPublica 2016 · Deployed COMPAS system</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;
                        margin-bottom:10px;">
                <div style="text-align:center;">
                    <div style="font-family:'IBM Plex Mono',monospace;
                                font-size:26px; color:var(--danger);
                                font-weight:500;">44.7%</div>
                    <div style="font-size:10px; color:var(--text-faint);
                                margin-top:4px; line-height:1.5;">
                        innocent Black<br>defendants flagged</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-family:'IBM Plex Mono',monospace;
                                font-size:26px; color:var(--accent);
                                font-weight:500;">23.2%</div>
                    <div style="font-size:10px; color:var(--text-faint);
                                margin-top:4px; line-height:1.5;">
                        innocent White<br>defendants flagged</div>
                </div>
            </div>
            <div style="text-align:center; font-family:'IBM Plex Mono',monospace;
                        font-size:15px; color:var(--danger); font-weight:500;
                        border-top:1px solid var(--danger-border);
                        padding-top:8px;">
                FPR Gap = 0.215
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── What We Did section ────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="eyebrow">What We Did</span>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-family:'Libre Baskerville',serif; font-size:22px;
                font-weight:700; color:#1a1a2e; margin-bottom:20px;">
        We trained {rashomon_summary['n_total']} models.
        <span style="font-style:italic; font-weight:400;">
            {rashomon_summary['n_in_set']} of them are equally accurate.
        </span>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3, gap="medium")
    steps = [
        ("Step 1", "Build the Rashomon Set",
         f"We trained {rashomon_summary['n_total']} classifiers: "
         f"Decision Trees at 7 depths, Logistic Regression at 10 regularisation "
         f"strengths, Random Forests, and Gradient Boosted Trees. "
         f"The {rashomon_summary['n_in_set']} models within 2% of peak accuracy "
         f"form the ε-Rashomon set — all are defensibly 'the best model.'"),
        ("Step 2", "Measure Bias for Each",
         "For every model we compute the ProPublica metric: the False Positive "
         "Rate gap between Black and White defendants. Among innocent people, "
         "what fraction does each model wrongly call dangerous? "
         "Always on held-out test data — never on training data."),
        ("Step 3", "Compare the Range",
         f"The FPR gap varies from "
         f"<strong style='color:var(--safe)'>{stats['fpr_gap_min']:.3f}</strong> "
         f"to <strong style='color:var(--danger)'>{stats['fpr_gap_max']:.3f}</strong> "
         f"within equally-accurate models. All have the same accuracy. "
         f"Accuracy alone cannot tell fair from biased." if stats else
         "Loading…"),
    ]
    for col, (eyebrow, title, body) in zip([s1, s2, s3], steps):
        with col:
            st.markdown(f"""
            <div class="card-accent" style="height:100%;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:9px;
                            color:var(--accent); text-transform:uppercase;
                            letter-spacing:.12em; margin-bottom:6px;">
                    {eyebrow}</div>
                <div style="font-family:'Libre Baskerville',serif; font-size:15px;
                            font-weight:700; color:#1a1a2e; margin-bottom:8px;">
                    {title}</div>
                <p class="body-text" style="font-size:13px; margin:0;">
                    {body}</p>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PAGE 2 — THE AUDIT
# =============================================================================

def page_the_audit(scatter_df, stats, rashomon_summary):
    """
    The central evidence page: the scatter plot of all 49 models
    with AUC on X and FPR disparity on Y.

    The visual argument: all dots are in a narrow band on X (accuracy is
    similar) but spread widely on Y (bias varies enormously).
    """

    st.markdown("""
    <div class="section-title">The Audit</div>
    <p class="body-text" style="max-width:680px; margin-bottom:24px;">
        Every dot is a trained model. Models inside the shaded band achieve
        within 2% of peak accuracy — they are all equally deployable.
        Look at how spread they are on the vertical axis.
        <em>Accuracy cannot distinguish the fair from the biased.</em>
    </p>
    """, unsafe_allow_html=True)

    # ── Chart controls ─────────────────────────────────────────────────────
    ctrl1, ctrl2 = st.columns([3, 1])
    with ctrl1:
        all_types     = sorted(scatter_df["model_type"].unique().tolist())
        selected_types = st.multiselect(
            "Show model types:",
            options  = all_types,
            default  = all_types,
            help     = "Filter which algorithm families appear on the plot",
        )
    with ctrl2:
        show_outside = st.toggle(
            "Show models outside Rashomon set",
            value = True,
            help  = "Gray dots are models below the 2% accuracy threshold",
        )

    # ── Filter to selected types ───────────────────────────────────────────
    plot_df = scatter_df[scatter_df["model_type"].isin(selected_types)].copy()
    if not show_outside:
        plot_df = plot_df[plot_df["in_rashomon"] == True]

    # ── Build the scatter plot ─────────────────────────────────────────────
    fig = go.Figure()

    if stats:
        # Shaded band showing the Rashomon set region on the X axis
        x_min = stats["rashomon_auc_min"] - 0.003
        x_max = stats["best_auc"]         + 0.003
        y_max = (plot_df["fpr_disparity"].max() + 0.025
                 if len(plot_df) > 0 else 0.25)

        fig.add_shape(
            type="rect",
            x0=x_min, x1=x_max, y0=0, y1=y_max,
            fillcolor="rgba(44,95,138,0.05)",
            line=dict(color="rgba(44,95,138,0.25)", width=1, dash="dot"),
        )
        fig.add_annotation(
            x=x_max, y=y_max - 0.006,
            text="← Rashomon set (ε = 2%)",
            font=dict(family="IBM Plex Mono", color="#2c5f8a", size=9),
            showarrow=False, xanchor="right",
        )

        # Horizontal reference line: where the deployed COMPAS system sat
        fig.add_hline(
            y=stats["propublica_fpr_gap"],
            line_dash="dash", line_color="#b83225", line_width=1.5,
            annotation_text=(f"Deployed COMPAS (2016): "
                             f"{stats['propublica_fpr_gap']:.3f}"),
            annotation_font=dict(family="IBM Plex Mono",
                                  color="#b83225", size=9),
            annotation_position="top right",
        )

    # Plot models OUTSIDE the Rashomon set first (grayed out, in background)
    if show_outside:
        outside = plot_df[plot_df["in_rashomon"] == False]
        if len(outside) > 0:
            fig.add_trace(go.Scatter(
                x     = outside["test_auc"],
                y     = outside["fpr_disparity"],
                mode  = "markers",
                name  = "Outside Rashomon set",
                marker = dict(
                    color="rgba(160,160,160,0.35)",
                    size=9, symbol="circle-open",
                    line=dict(width=1, color="rgba(150,150,150,0.5)"),
                ),
                hovertemplate = (
                    "<b>%{customdata}</b><br>"
                    "AUC: %{x:.4f}<br>"
                    "FPR gap: %{y:.3f}<br>"
                    "<i>Outside Rashomon set</i>"
                    "<extra></extra>"
                ),
                customdata = outside["name"],
            ))

    # Plot each model type in its color (only Rashomon set members)
    for mtype in selected_types:
        subset = plot_df[
            (plot_df["model_type"]  == mtype) &
            (plot_df["in_rashomon"] == True)
        ]
        if len(subset) == 0:
            continue

        color = MODEL_TYPE_COLORS.get(mtype, "#888888")

        fig.add_trace(go.Scatter(
            x     = subset["test_auc"],
            y     = subset["fpr_disparity"],
            mode  = "markers",
            name  = mtype,
            marker = dict(
                color  = color,
                size   = subset["dot_size"] + 4,
                symbol = "circle",
                line   = dict(width=1.5, color="white"),
                opacity= 0.88,
            ),
            hovertext      = subset["hover_text"],
            hoverinfo      = "text",
        ))

    fig.update_layout(
        xaxis_title = "Test AUC — accuracy (higher = better)",
        yaxis_title = "FPR gap: |FPR_Black − FPR_White| (lower = fairer)",
        legend      = dict(orientation="h", y=-0.22),
    )
    _apply_chart_theme(fig, height=480,
                       title_text="Accuracy vs Racial Bias — All 49 Models")
    st.plotly_chart(fig, use_container_width=True)

    # ── Reading guide ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="card-accent" style="margin-top:6px;">
        <strong>How to read this chart:</strong>
        The shaded band contains all models within 2% of the best accuracy.
        On the horizontal axis they are nearly indistinguishable.
        On the vertical axis they span 0.116 to 0.200 — a gap of 0.084.
        The red dashed line shows where the actually deployed COMPAS system
        sat (0.215). Most models in the Rashomon set would have been
        <em>fairer</em> than the system that was deployed — with equal accuracy.
    </div>
    """, unsafe_allow_html=True)

    # ── Table of Rashomon set members ──────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="eyebrow">Rashomon Set — All 40 Models</span>',
                unsafe_allow_html=True)
    st.markdown("""
    <p class="body-text" style="margin-bottom:12px;">
        Sorted from fairest to most biased. Every row is an equally
        defensible deployment choice by accuracy alone.
    </p>
    """, unsafe_allow_html=True)

    in_set = scatter_df[scatter_df["in_rashomon"] == True].copy()
    table  = in_set[[
        "short_name", "model_type", "test_auc",
        "fpr_disparity", "fpr_black", "fpr_white",
    ]].sort_values("fpr_disparity").copy()

    table.columns = [
        "Model", "Type", "AUC",
        "FPR Gap ↓ fairer", "FPR (Black)", "FPR (White)",
    ]

    st.dataframe(
        table,
        use_container_width = True,
        hide_index          = True,
        height              = 380,
        column_config = {
            "AUC":           st.column_config.NumberColumn(format="%.4f"),
            "FPR Gap ↓ fairer":
                             st.column_config.NumberColumn(format="%.3f"),
            "FPR (Black)":   st.column_config.NumberColumn(format="%.3f"),
            "FPR (White)":   st.column_config.NumberColumn(format="%.3f"),
        },
    )


# =============================================================================
# PAGE 3 — MODEL EXPLORER
# =============================================================================

def page_model_explorer(fairness_df, rashomon_result, stats):
    """
    Drill-down: pick any model from the Rashomon set, see its fairness
    profile, feature importances, and (for trees) its decision rules.
    Compares the selected model side-by-side with the fairest and most biased.
    """

    st.markdown("""
    <div class="section-title">Model Explorer</div>
    <p class="body-text" style="max-width:680px; margin-bottom:24px;">
        Select any model from the Rashomon set to inspect its decision logic.
        All models here have the same accuracy. Their fairness profiles differ
        dramatically. For Decision Trees, you can read the exact rules used
        to label any defendant as high risk.
    </p>
    """, unsafe_allow_html=True)

    in_set = fairness_df[fairness_df["in_rashomon"] == True]

    # Sort dropdown from fairest to most biased so user sees spectrum clearly
    model_options = (in_set.sort_values("fpr_disparity")["name"].tolist())

    selected_name = st.selectbox(
        "Select a model from the Rashomon set (sorted: fairest → most biased):",
        model_options,
    )
    selected_key = in_set[in_set["name"] == selected_name]["key"].values[0]

    # Get the detailed record for the selected model
    detail = get_model_detail(selected_key, fairness_df, rashomon_result)
    if detail is None:
        st.error("Could not load model detail.")
        return

    # Identify fairest and most biased for comparison
    fairest_key   = in_set.loc[in_set["fpr_disparity"].idxmin(), "key"]
    unfairest_key = in_set.loc[in_set["fpr_disparity"].idxmax(), "key"]
    sel_row       = in_set[in_set["key"] == selected_key].iloc[0]
    fair_row      = in_set[in_set["key"] == fairest_key].iloc[0]
    unfair_row    = in_set[in_set["key"] == unfairest_key].iloc[0]

    # ── Three-column comparison ────────────────────────────────────────────
    col_sel, col_fair, col_unfair = st.columns(3, gap="medium")

    def _metric_card(row, label, card_class):
        gap  = row["fpr_disparity"]
        gcol = ("#b83225" if gap > 0.17 else
                "#b8770c" if gap > 0.14 else "#1a6b3c")
        return f"""
        <div class="{card_class}">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:9px;
                        color:{gcol}; text-transform:uppercase;
                        letter-spacing:.1em; margin-bottom:7px;">{label}</div>
            <div style="font-family:'Libre Baskerville',serif; font-size:13px;
                        font-weight:700; color:#1a1a2e; margin-bottom:12px;
                        line-height:1.35;">{row['name'][:52]}</div>
            <div style="display:grid; grid-template-columns:1fr 1fr;
                        gap:5px; font-size:12px; font-family:'IBM Plex Mono',
                        monospace;">
                <span style="color:var(--text-faint);">AUC</span>
                <span style="color:#1a1a2e;">{row['test_auc']:.4f}</span>
                <span style="color:var(--text-faint);">FPR gap</span>
                <span style="color:{gcol}; font-weight:500;">
                    {row['fpr_disparity']:.3f}</span>
                <span style="color:var(--text-faint);">FPR Black</span>
                <span style="color:var(--danger);">{row['fpr_black']:.3f}</span>
                <span style="color:var(--text-faint);">FPR White</span>
                <span style="color:var(--accent);">{row['fpr_white']:.3f}</span>
            </div>
        </div>
        """

    with col_sel:
        st.markdown(_metric_card(sel_row, "Selected model", "card"),
                    unsafe_allow_html=True)
    with col_fair:
        st.markdown(_metric_card(fair_row, "Fairest in set →",
                                 "card-safe"), unsafe_allow_html=True)
    with col_unfair:
        st.markdown(_metric_card(unfair_row, "Most biased in set →",
                                 "card-danger"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature importances + decision logic ──────────────────────────────
    col_feat, col_rules = st.columns([1, 1], gap="large")

    with col_feat:
        st.markdown('<span class="eyebrow">What This Model Relies On</span>',
                    unsafe_allow_html=True)

        importances = detail["feature_importances"]
        # Readable labels for the chart
        feat_labels = {
            "age":             "Age",
            "priors_count":    "Prior Offenses",
            "charge_degree":   "Charge Degree",
            "sex_male":        "Sex (Male)",
            "juv_fel_count":   "Juvenile Felonies",
            "juv_misd_count":  "Juvenile Misdemeanors",
            "juv_other_count": "Juvenile Other",
        }
        labels = [feat_labels.get(f, f) for f in importances.keys()]
        values = list(importances.values())
        max_v  = max(values) if values else 1

        # Horizontal bar chart — feature importance
        fig_imp = go.Figure(go.Bar(
            x = values,
            y = labels,
            orientation = "h",
            marker_color = [
                f"rgba(44,95,138,{0.25 + 0.75 * v / max_v})"
                for v in values
            ],
            marker_line_width = 0,
            hovertemplate = "%{y}: %{x:.3f}<extra></extra>",
        ))
        fig_imp.update_layout(
            xaxis_title = "Importance score",
            xaxis       = dict(range=[0, max_v * 1.18]),
            yaxis_title = "",
            showlegend  = False,
        )
        _apply_chart_theme(fig_imp, height=260,
                           title_text=f"Feature weights — {detail['model_type']}")
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("""
        <p class="body-text" style="font-size:12px;">
            <strong>Note:</strong> Prior offenses is consistently the most
            important feature. It is a proxy for race due to unequal policing
            — the model never sees race, yet absorbs the racial pattern
            through this feature.
        </p>
        """, unsafe_allow_html=True)

    with col_rules:
        st.markdown('<span class="eyebrow">Decision Logic</span>',
                    unsafe_allow_html=True)

        if detail["model_type"] == "Decision Tree" and detail["decision_rules"]:
            st.markdown(f"""
            <div style="font-family:'IBM Plex Mono',monospace; font-size:12px;
                        background:#f4f4f0; border:1px solid #e5e5e0;
                        border-radius:6px; padding:14px; white-space:pre;
                        overflow-x:auto; max-height:310px; overflow-y:auto;
                        line-height:1.65; color:#1a1a2e;">
{detail['decision_rules']}</div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <p class="body-text" style="font-size:12px; margin-top:8px;">
                This <strong>{detail['n_leaves']}-leaf tree</strong> has
                {detail['n_leaves']} fully auditable decision paths.
                Every prediction follows a branch you can read and verify.
            </p>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="card-amber">
                <div style="font-family:'Libre Baskerville',serif;
                            font-size:14px; font-weight:700;
                            color:var(--amber); margin-bottom:6px;">
                    Black-box model</div>
                <p class="body-text" style="font-size:13px; margin:0;">
                    This <strong>{detail['model_type']}</strong> cannot be
                    expressed as a simple set of rules. Its decisions
                    emerge from the combination of many internal parameters
                    that no human can easily audit.
                    The feature importances chart shows an approximation
                    of what drives its predictions — but not the exact
                    reasoning for any individual defendant.
                </p>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PAGE 4 — WHO BEARS IT
# =============================================================================

def page_who_bears_it(disagree, impossib_df, fairness_df):
    """
    Two tabs:
      1. Defendant Disagreement — real people affected by model choice
      2. The Impossibility Theorem — why ProPublica and Northpointe
         were both simultaneously right
    """

    st.markdown("""
    <div class="section-title">Who Bears the Cost?</div>
    <p class="body-text" style="max-width:680px; margin-bottom:24px;">
        Abstract bias metrics become concrete when you ask: which specific
        defendants get different outcomes depending on which equally-accurate
        model was deployed? And why were both sides of the COMPAS debate
        simultaneously correct?
    </p>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Defendant Disagreement", "The Impossibility Theorem"])

    # ── TAB 1: Defendant Disagreement ─────────────────────────────────────
    with tab1:
        st.markdown('<span class="eyebrow">People at the mercy of model choice</span>',
                    unsafe_allow_html=True)

        if not disagree:
            st.info("Disagreement data not available.")
        else:
            # Headline statistics
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.markdown(f"""<div class="stat-box">
                    <span class="stat-value danger">
                        {disagree['n_disagreements']}</span>
                    <span class="stat-label">Defendants affected</span>
                </div>""", unsafe_allow_html=True)
            with d2:
                st.markdown(f"""<div class="stat-box">
                    <span class="stat-value dark">
                        {disagree['disagree_pct']}%</span>
                    <span class="stat-label">Of the test set</span>
                </div>""", unsafe_allow_html=True)
            with d3:
                st.markdown(f"""<div class="stat-box">
                    <span class="stat-value danger">
                        {disagree['disagree_black_pct']}%</span>
                    <span class="stat-label">Of Black defendants</span>
                </div>""", unsafe_allow_html=True)
            with d4:
                st.markdown(f"""<div class="stat-box">
                    <span class="stat-value accent">
                        {disagree['disagree_white_pct']}%</span>
                    <span class="stat-label">Of White defendants</span>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # The two models being compared
            st.markdown(f"""
            <div style="display:grid; grid-template-columns:1fr 1fr;
                        gap:14px; margin-bottom:20px;">
                <div class="card-safe">
                    <div style="font-family:'IBM Plex Mono',monospace;
                                font-size:9px; color:var(--safe);
                                text-transform:uppercase; letter-spacing:.1em;
                                margin-bottom:6px;">Fairest model in set</div>
                    <div style="font-family:'Libre Baskerville',serif;
                                font-size:13px; font-weight:700; color:#1a1a2e;
                                margin-bottom:5px;">
                        {disagree['fairest_name'][:55]}</div>
                    <div style="font-family:'IBM Plex Mono',monospace;
                                font-size:11px; color:var(--text-muted);">
                        AUC = {disagree['auc_fairest']:.4f} ·
                        FPR gap = {disagree['fpr_gap_fairest']:.3f}</div>
                </div>
                <div class="card-danger">
                    <div style="font-family:'IBM Plex Mono',monospace;
                                font-size:9px; color:var(--danger);
                                text-transform:uppercase; letter-spacing:.1em;
                                margin-bottom:6px;">Most biased model in set</div>
                    <div style="font-family:'Libre Baskerville',serif;
                                font-size:13px; font-weight:700; color:#1a1a2e;
                                margin-bottom:5px;">
                        {disagree['unfairest_name'][:55]}</div>
                    <div style="font-family:'IBM Plex Mono',monospace;
                                font-size:11px; color:var(--text-muted);">
                        AUC = {disagree['auc_unfairest']:.4f} ·
                        FPR gap = {disagree['fpr_gap_unfairest']:.3f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Bar chart — disagreement rate by race
            fig_dis = go.Figure()
            groups  = ["African-American", "Caucasian"]
            values  = [disagree["disagree_black_pct"],
                       disagree["disagree_white_pct"]]
            colors  = ["#b83225", "#2c5f8a"]

            fig_dis.add_trace(go.Bar(
                x     = groups,
                y     = values,
                marker_color     = colors,
                marker_line_width= 0,
                text  = [f"{v:.1f}%" for v in values],
                textposition     = "outside",
                textfont = dict(family="IBM Plex Mono", size=12),
                hovertemplate    = "%{x}: %{y:.1f}% of defendants affected<extra></extra>",
            ))
            fig_dis.update_layout(
                xaxis_title = "",
                yaxis_title = "% of group with different prediction",
                yaxis_range = [0, max(values) * 1.3],
                showlegend  = False,
            )
            _apply_chart_theme(fig_dis, height=300,
                               title_text="Disagreement Rate by Race")
            st.plotly_chart(fig_dis, use_container_width=True)

            st.markdown(f"""
            <div class="card-danger" style="margin-top:6px;">
                <strong>What this means:</strong>
                {disagree['n_disagreements']} real defendants in our test set
                receive a different bail recommendation depending solely on
                which equally-accurate model the court chose to deploy.
                Black defendants are disproportionately affected
                ({disagree['disagree_black_pct']}% vs
                {disagree['disagree_white_pct']}% for White defendants).
                Both models have the same accuracy. Their human costs are
                not the same.
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 2: The Impossibility Theorem ──────────────────────────────────
    with tab2:
        st.markdown('<span class="eyebrow">Chouldechova (2017) Impossibility Theorem</span>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="card" style="margin-bottom:18px;">
            <div style="font-family:'Libre Baskerville',serif; font-size:17px;
                        font-weight:700; color:#1a1a2e; margin-bottom:10px;">
                Why ProPublica and Northpointe were both right</div>
            <p class="body-text" style="margin-bottom:10px;">
                ProPublica found COMPAS had a higher
                <strong>False Positive Rate (FPR)</strong> for Black defendants
                — innocent Black people were more often labelled dangerous.
                Northpointe responded that their model had equal
                <strong>False Negative Rate (FNR)</strong> across races
                — people who actually reoffended were equally likely to have
                been labelled low-risk regardless of race.
                Both were measuring real things. Neither was lying.
            </p>
            <p class="body-text">
                Chouldechova (2017) proved this is mathematically inevitable.
                When recidivism base rates differ between groups
                — as they do here (52.3% Black vs 39.1% White) —
                no algorithm can simultaneously achieve equal FPR
                <em>and</em> equal FNR <em>and</em> equal predictive parity.
                The choice of which criterion to prioritise is a normative,
                political decision — not a technical one.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # FPR vs FNR scatter — the empirical tradeoff
        in_set_imp = impossib_df[impossib_df["in_rashomon"] == True]
        fig_imp    = go.Figure()

        for mtype in in_set_imp["model_type"].unique():
            subset = in_set_imp[in_set_imp["model_type"] == mtype]
            fig_imp.add_trace(go.Scatter(
                x         = subset["fpr_disparity"],
                y         = subset["fnr_disparity"],
                mode      = "markers",
                name      = mtype,
                marker    = dict(color=MODEL_TYPE_COLORS.get(mtype, "#888"),
                                 size=10,
                                 line=dict(width=1, color="white")),
                hovertext = subset["hover_text"],
                hoverinfo = "text",
            ))

        fig_imp.update_layout(
            xaxis_title = "FPR gap — ProPublica criterion (lower = fairer by this measure)",
            yaxis_title = "FNR gap — Northpointe criterion (lower = fairer by this measure)",
        )
        _apply_chart_theme(fig_imp, height=380,
                           title_text="FPR vs FNR Disparity — Rashomon Set Members")
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("""
        <div class="card-amber">
            <strong>Reading this chart:</strong>
            If both fairness criteria could be satisfied simultaneously,
            all dots would cluster near (0, 0) — zero disparity on both.
            Instead, they spread across both axes. No single model achieves
            low disparity on both measures at once. This is not a modelling
            failure — it is a mathematical consequence of the different base
            rates between the two racial groups.
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 5 — ABOUT
# =============================================================================

def page_about():
    """
    Methodology, research citations, limitations, and reproducibility numbers.
    """

    st.markdown("""
    <div class="section-title">About This Project</div>
    <p class="body-text" style="max-width:680px; margin-bottom:28px;">
        The technical methodology, direct research connections, and honest
        limitations of this audit.
    </p>
    """, unsafe_allow_html=True)

    col_method, col_papers = st.columns([3, 2], gap="large")

    with col_method:
        st.markdown('<span class="eyebrow">Technical Pipeline</span>',
                    unsafe_allow_html=True)

        pipeline_steps = [
            ("1. Data",
             "ProPublica COMPAS dataset (compas-scores-two-years.csv). "
             "Downloaded from GitHub. Filtered following ProPublica's exact "
             "4-step methodology: days_b_screening_arrest ∈ [−30,+30], "
             "is_recid ≠ −1, charge degree ≠ 'O', score_text ≠ 'N/A'. "
             "7,214 raw rows → 6,172 after filtering."),
            ("2. Features",
             "Age, prior offense count, charge degree (felony/misdemeanor), "
             "sex, and juvenile offense counts. Race is excluded from model "
             "inputs — it is the protected attribute for fairness analysis."),
            ("3. Model Family",
             "49 models: Decision Trees (depth 2–8 × min_leaf 5/10/20 = 21), "
             "Logistic Regression (C = 0.001 to 100 = 10), Random Forest "
             "(4 depths × 3 leaf sizes = 12), Gradient Boosting "
             "(3 depths × 2 learning rates = 6). Total: 49 configurations."),
            ("4. Rashomon Set",
             "ε = 0.02: models with test AUC ≥ best_AUC − 0.02 are in the set. "
             "40 of 49 models qualify. Train/test split: 80/20, stratified "
             "by race × label. All fairness metrics on test set only."),
            ("5. Fairness Metric",
             "Primary: FPR disparity = |FPR_Black − FPR_White|, where FPR = "
             "false positive rate (fraction of non-recidivists predicted "
             "high-risk). This is ProPublica's metric. Threshold: P ≥ 0.5. "
             "We also compute FNR and FDR disparities for the theorem page."),
        ]

        for title, body in pipeline_steps:
            st.markdown(f"""
            <div style="display:flex; gap:12px; margin-bottom:10px;
                        background:#ffffff; border:1px solid #e5e5e0;
                        border-radius:6px; padding:12px;">
                <div style="width:3px; min-width:3px; background:var(--accent);
                            border-radius:2px;"></div>
                <div>
                    <div style="font-family:'Libre Baskerville',serif;
                                font-size:13px; color:#1a1a2e; font-weight:700;
                                margin-bottom:3px;">{title}</div>
                    <div class="body-text" style="font-size:12px; margin:0;">
                        {body}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<span class="eyebrow" style="margin-top:16px;">Limitations</span>',
                    unsafe_allow_html=True)
        limits = [
            ("Threshold choice (P ≥ 0.5)",
             "FPR/FNR values depend on the classification threshold. COMPAS "
             "uses a 10-point scale with a different effective threshold. "
             "Our qualitative finding — that bias varies within the Rashomon "
             "set — is threshold-independent."),
            ("Not the actual COMPAS algorithm",
             "We train our own models on COMPAS features. The deployed COMPAS "
             "system used additional proprietary inputs. Our results demonstrate "
             "that bias variation within the Rashomon set is a property of "
             "this data space, not a specific claim about Northpointe's code."),
            ("Historical data",
             "The 2013–2014 Broward County data reflects policing practices "
             "of that era. Patterns may differ in other jurisdictions or "
             "time periods."),
        ]
        for title, body in limits:
            st.markdown(f"""
            <div class="card-amber" style="margin-bottom:8px;">
                <strong>{title}:</strong>
                <span class="body-text"> {body}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_papers:
        st.markdown('<span class="eyebrow">Research Foundation</span>',
                    unsafe_allow_html=True)

        papers = [
            ("#2c5f8a",
             "The Double-Edged Nature of the Rashomon Set for Trustworthy ML",
             "Semenova, Hsu, Chen, Zhong · arXiv 2025",
             "The direct theoretical foundation for this project. Shows that "
             "the Rashomon set creates both fairness risk (models span a wide "
             "bias range) and fairness opportunity (fairer models are almost "
             "always available within it)."),
            ("#2d7a45",
             "On the Existence of Simpler Machine Learning Models",
             "Semenova, Rudin, Parr · FAccT 2022",
             "Proves that the Rashomon set almost always contains simple, "
             "interpretable models with near-identical accuracy to complex "
             "black-box alternatives."),
            ("#b83225",
             "Machine Bias",
             "Angwin, Larson, Mattu, Kirchner · ProPublica 2016",
             "The original investigation that revealed COMPAS's racial bias "
             "and provided the public dataset this project uses."),
            ("#7c3f8c",
             "A Study of the Impossibility of Fairness",
             "Chouldechova · Criminal Justice & Behavior 2017",
             "Proves that equal FPR and equal FNR cannot be simultaneously "
             "achieved when base rates differ between groups."),
        ]

        for color, title, citation, desc in papers:
            st.markdown(f"""
            <div style="background:#ffffff; border:1px solid {color}33;
                        border-left:3px solid {color};
                        border-radius:0 6px 6px 0; padding:12px 14px;
                        margin-bottom:10px;">
                <div style="font-family:'Libre Baskerville',serif; font-size:13px;
                            font-weight:700; color:#1a1a2e; margin-bottom:3px;
                            line-height:1.35;">{title}</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:9px;
                            color:{color}; margin-bottom:6px;">{citation}</div>
                <div style="font-size:12px; color:#8a8a96;
                            line-height:1.65;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        # Reproducibility numbers
        st.markdown('<span class="eyebrow" style="margin-top:16px;">Reproducibility</span>',
                    unsafe_allow_html=True)
        repro = [
            ("6,172",  "Defendants (ProPublica filter)"),
            ("49",     "Models trained"),
            ("40",     "In Rashomon set (ε=0.02)"),
            ("0.7317", "Best test AUC"),
            ("0.116",  "Min FPR gap (fairest model)"),
            ("0.200",  "Max FPR gap (most biased)"),
            ("0.0016", "AUC cost of fairness"),
            ("205",    "Defendants with different outcomes"),
        ]
        rr1, rr2 = st.columns(2)
        for i, (val, label) in enumerate(repro):
            col = rr1 if i % 2 == 0 else rr2
            with col:
                st.markdown(f"""
                <div style="text-align:center; padding:8px;
                            background:#ffffff; border:1px solid #e5e5e0;
                            border-radius:6px; margin-bottom:7px;">
                    <div style="font-family:'IBM Plex Mono',monospace;
                                font-size:16px; color:var(--accent);">{val}</div>
                    <div style="font-size:9px; color:#8a8a96;
                                text-transform:uppercase;
                                letter-spacing:.07em; margin-top:2px;">
                        {label}</div>
                </div>
                """, unsafe_allow_html=True)


# =============================================================================
# MAIN — entry point
# =============================================================================

def main():
    """
    Loads all data (with Streamlit caching), renders the sidebar,
    and routes to the active page.

    First run: ~8 seconds (download + train 49 models + compute metrics).
    Every subsequent interaction: instant (all cached).
    """

    # Initialise page state
    if "page" not in st.session_state:
        st.session_state["page"] = "The Case"

    # ── Load data sequentially (each step cached) ─────────────────────────
    with st.spinner("Loading COMPAS dataset…"):
        X, y, df, race, summary, race_stats = load_fast()

    with st.spinner("Training 49 models and building Rashomon set (~5 sec on first run)…"):
        rashomon_result, rashomon_summary = load_rashomon(X, y, race)

    with st.spinner("Computing fairness metrics for all models…"):
        (fairness_df, scatter_df, stats,
         disagree, impossib_df, pareto) = load_fairness(rashomon_result)

    # ── Sidebar (always visible) ──────────────────────────────────────────
    render_sidebar(stats)

    # ── Route to the active page ──────────────────────────────────────────
    page = st.session_state.get("page", "The Case")

    if page == "The Case":
        page_the_case(summary, race_stats, stats, rashomon_summary)

    elif page == "The Audit":
        page_the_audit(scatter_df, stats, rashomon_summary)

    elif page == "Model Explorer":
        page_model_explorer(fairness_df, rashomon_result, stats)

    elif page == "Who Bears It":
        page_who_bears_it(disagree, impossib_df, fairness_df)

    elif page == "About":
        page_about()


# Only run main() when this script is executed directly
if __name__ == "__main__":
    main()
