"""Portfolio Advisor — Streamlit UI (Marathon-inspired tactical design)"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from main import SAMPLE_PORTFOLIOS
from orchestrator import run_analysis
from schemas import RiskOutput, MarketIntelOutput
from agents.risk_agent import compute_metrics

# ── Colour palette ─────────────────────────────────────────────────────────────
BG      = "#F0EFE9"
SURFACE = "#FAFAF7"
PANEL   = "#FFFFFF"
BORDER  = "#D6D4CB"
BDACC   = "#B8A07A"
TEXT    = "#1A1918"
MUTED   = "#6A6660"
ACCENT  = "#B8924A"
BLUE    = "#6A8EAD"
LOW     = "#5A9660"
MID     = "#C49030"
HIGH    = "#BC4A36"

PLOTLY_COLORS = [ACCENT, BLUE, "#7AB87A", HIGH, "#8A6AAD", "#4A9696", MID, "#AD7A6A"]


# ── CSS injection ──────────────────────────────────────────────────────────────

def _inject_css() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;500;600;700&family=Barlow:wght@300;400;500&family=Share+Tech+Mono&display=swap');

/* ── Global ── */
.stApp {{ background: {BG}; font-family: 'Barlow', sans-serif; color: {TEXT}; }}
#MainMenu, footer, [data-testid="stDecoration"],
[data-testid="stToolbar"], .stAppDeployButton,
[data-testid="stStatusWidget"] {{ visibility: hidden !important; height: 0 !important; }}
header[data-testid="stHeader"] {{
    background: transparent !important;
    height: 0 !important;
    min-height: 0 !important;
}}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}
[data-testid="stMainBlockContainer"] {{ padding: 0 !important; }}
/* Ensure all text in main area is visible */
p, span, div, label {{ color: inherit; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {SURFACE} !important;
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] > div:first-child {{
    padding: 1.5rem 1.2rem 1.5rem;
}}
.sidebar-head {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: {MUTED};
    padding-bottom: 0.6rem;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 1.2rem;
}}

/* ── Buttons ── */
[data-testid="stButton"] > button {{
    background: {ACCENT} !important;
    color: {PANEL} !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.25em !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    width: 100% !important;
    padding: 0.65rem !important;
    transition: background 0.15s ease;
}}
[data-testid="stButton"] > button:hover {{
    background: {TEXT} !important;
}}

/* ── Text inputs / textareas ── */
.stTextArea textarea {{
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 0 !important;
    color: {TEXT} !important;
}}

/* ── Selectbox / radio — widget labels ── */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
[data-testid="stWidgetLabel"] label {{
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    color: {MUTED} !important;
}}

/* ── Radio option labels ── */
[data-testid="stRadio"] div[role="radiogroup"] label p,
[data-testid="stRadio"] div[role="radiogroup"] label span,
[data-testid="stRadio"] div[role="radiogroup"] [data-testid="stMarkdownContainer"] p,
[data-testid="stRadio"] label div p {{
    font-family: 'Barlow', sans-serif !important;
    font-size: 0.83rem !important;
    letter-spacing: 0.03em !important;
    text-transform: none !important;
    color: {TEXT} !important;
}}

/* ── Selectbox display value ── */
[data-testid="stSelectbox"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSelectbox"] span,
[data-testid="stSelectbox"] div[data-baseweb="select"] span {{
    font-family: 'Barlow', sans-serif !important;
    font-size: 0.85rem !important;
    color: {TEXT} !important;
}}
/* ── Selectbox — prevent text editing ── */
[data-testid="stSelectbox"] input {{
    caret-color: transparent !important;
    user-select: none !important;
    cursor: pointer !important;
}}

/* ── All sidebar text fallback ── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not([data-testid]),
[data-testid="stSidebar"] div[class*="label"] {{
    color: {TEXT} !important;
}}

/* ── Spinner — hide the default widget ── */
[data-testid="stSpinner"] {{ display: none !important; }}

/* ── Analysis overlay ── */
#analysis-overlay {{
    position: fixed;
    inset: 0;
    background: rgba(26,25,24,0.6);
    backdrop-filter: blur(3px);
    z-index: 99999;
    display: flex;
    align-items: center;
    justify-content: center;
}}
#analysis-overlay .card {{
    background: {PANEL};
    border-top: 3px solid {ACCENT};
    padding: 2.2rem 3.5rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1.2rem;
    min-width: 320px;
}}
#analysis-overlay .spinner-ring {{
    width: 2rem; height: 2rem;
    border: 3px solid {BORDER};
    border-top-color: {ACCENT};
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
#analysis-overlay .label {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.82rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: {TEXT};
    margin: 0;
}}

/* ── Info / warning / error boxes ── */
[data-testid="stAlert"] {{
    border-radius: 0 !important;
    font-family: 'Barlow', sans-serif !important;
    font-size: 0.85rem !important;
}}

/* ── App header ── */
.app-header {{
    background: {PANEL};
    border-bottom: 2px solid {BDACC};
    padding: 0.85rem 1.75rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0;
    position: relative;
    z-index: 1;
}}
.app-header-gem {{
    width: 24px; height: 24px;
    background: {ACCENT};
    clip-path: polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);
    flex-shrink: 0;
}}
.app-header-title {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: {TEXT};
    line-height: 1;
}}
.app-header-sub {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    color: {MUTED};
    text-transform: uppercase;
    margin-left: auto;
}}
.app-header-divider {{
    width: 1px; height: 20px; background: {BORDER};
}}

/* ── Section label ── */
.sec-lbl {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: {MUTED};
    padding: 0.7rem 0 0.3rem;
    border-top: 1px solid {BORDER};
    margin-top: 0.6rem;
}}

/* ── Portfolio compact table ── */
.port-wrap {{
    border: 1px solid {BORDER};
    border-left: 3px solid {ACCENT};
    background: {PANEL};
    overflow: hidden;
}}
.port-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.76rem;
}}
.port-table th {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.57rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: {MUTED};
    padding: 0.28rem 0.65rem;
    text-align: left;
    border-bottom: 1px solid {BORDER};
    background: {SURFACE};
}}
.port-table td {{
    padding: 0.18rem 0.65rem;
    border-bottom: 1px solid {BORDER};
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.73rem;
    color: {TEXT};
}}
.port-table td:first-child {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: {ACCENT};
}}
.port-table tr:last-child td {{ border-bottom: none; }}
.port-footer {{
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    color: {MUTED};
    padding: 0.3rem 0.65rem;
    border-top: 1px solid {BORDER};
    background: {SURFACE};
    text-align: right;
    letter-spacing: 0.08em;
}}

/* ── Advisory blocks ── */
.adv-block {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-top: 2px solid {ACCENT};
    padding: 1rem 1.25rem;
    margin-bottom: 0.65rem;
}}
.adv-block-title {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: {ACCENT};
    margin-bottom: 0.55rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid {BORDER};
}}
.adv-block p {{
    font-size: 0.87rem;
    line-height: 1.7;
    color: {TEXT};
    margin: 0;
}}

/* ── Recommendations ── */
.rec-block {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-top: 2px solid {BLUE};
    padding: 1rem 1.25rem;
    margin-bottom: 0.65rem;
}}
.rec-item {{
    display: flex;
    gap: 0.7rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid {BORDER};
    align-items: flex-start;
}}
.rec-item:last-child {{ border-bottom: none; }}
.rec-n {{
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    color: {BLUE};
    min-width: 1.4rem;
    padding-top: 0.12rem;
    font-weight: 700;
    flex-shrink: 0;
}}
.rec-t {{
    font-size: 0.83rem;
    line-height: 1.55;
    color: {TEXT};
}}

/* ── Risk flags ── */
.flag {{
    font-size: 0.79rem;
    line-height: 1.45;
    padding: 0.42rem 0.72rem;
    margin-bottom: 0.3rem;
}}
.flag-critical {{
    border-left: 3px solid {HIGH};
    background: rgba(188,74,54,0.06);
    color: {TEXT};
}}
.flag-warning {{
    border-left: 3px solid {MID};
    background: rgba(196,144,48,0.06);
    color: {TEXT};
}}

/* ── Right panel risk score ── */
.risk-score-card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    padding: 1.1rem 1rem;
    text-align: center;
    margin-bottom: 0.5rem;
}}
.risk-num {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 4.2rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.02em;
}}
.risk-denom {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.1rem;
    font-weight: 300;
    color: {MUTED};
    margin-left: 0.1rem;
}}
.risk-lbl {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.58rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: {MUTED};
    margin-top: 0.2rem;
}}
.risk-lvl {{
    display: inline-block;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    padding: 0.18rem 0.55rem;
    margin-top: 0.45rem;
    border: 1px solid;
}}
.rl-low  {{ color: {LOW};  border-color: {LOW};  background: rgba(90,150,96,0.1); }}
.rl-mid  {{ color: {MID};  border-color: {MID};  background: rgba(196,144,48,0.1); }}
.rl-high {{ color: {HIGH}; border-color: {HIGH}; background: rgba(188,74,54,0.1); }}

/* ── Mini metrics row (right panel) ── */
.mini-metrics {{
    display: grid;
    grid-template-columns: repeat(3,1fr);
    gap: 0.4rem;
    margin-bottom: 0.5rem;
}}
.mm-cell {{
    background: {PANEL};
    border: 1px solid {BORDER};
    padding: 0.55rem 0.4rem;
    text-align: center;
}}
.mm-lbl {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.52rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 0.18rem;
}}
.mm-val {{
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.05rem;
    color: {TEXT};
    font-weight: 700;
}}

/* ── Chart panel title strip ── */
.chart-hd {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: {MUTED};
    padding: 0.4rem 0.75rem;
    border-bottom: 1px solid {BORDER};
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-bottom: none;
}}

/* ── Pending / empty state ── */
.pending-state {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 4rem 1rem;
    text-align: center;
    color: {MUTED};
    background: {SURFACE};
    border: 1px solid {BORDER};
}}
.pending-icon {{
    font-size: 2rem;
    color: {ACCENT};
    opacity: 0.45;
    margin-bottom: 0.75rem;
}}
.pending-txt {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: {MUTED};
}}

/* ── News ticker ── */
.ticker-wrap {{
    position: fixed;
    bottom: 0; left: 0; right: 0;
    height: 40px;
    background: {TEXT};
    border-top: 2px solid {ACCENT};
    display: flex;
    align-items: center;
    overflow: hidden;
    z-index: 9999;
    font-size: 0;            /* collapse whitespace */
}}
.ticker-badge {{
    background: {ACCENT};
    color: #FFF;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.63rem;
    font-weight: 700;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    padding: 0 1.1rem;
    height: 100%;
    display: inline-flex;
    align-items: center;
    white-space: nowrap;
    flex-shrink: 0;
    min-width: 110px;
    justify-content: center;
}}
.ticker-scroll {{
    flex: 1;
    overflow: hidden;
    height: 100%;
    display: flex;
    align-items: center;
    font-size: 14px;
}}
.ticker-track {{
    display: inline-flex;
    align-items: center;
    animation: ticker-run 70s linear infinite;
    white-space: nowrap;
}}
.ticker-track:hover {{ animation-play-state: paused; }}
.ticker-item {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0 1.8rem;
    font-family: 'Barlow', sans-serif;
    font-size: 0.74rem;
    color: #E8E6DE;
}}
.ti-sym {{
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    color: {ACCENT};
}}
.ti-score-pos {{ color: {LOW};  font-family: 'Share Tech Mono', monospace; font-size: 0.64rem; }}
.ti-score-neg {{ color: {HIGH}; font-family: 'Share Tech Mono', monospace; font-size: 0.64rem; }}
.ti-score-neu {{ color: {BLUE}; font-family: 'Share Tech Mono', monospace; font-size: 0.64rem; }}
.ti-dot {{ color: {BDACC}; font-size: 0.4rem; padding: 0 0.5rem; }}
@keyframes ticker-run {{
    0%   {{ transform: translateX(0); }}
    100% {{ transform: translateX(-50%); }}
}}

/* ── Plotly chart borders ── */
.stPlotlyChart > div {{
    border: 1px solid {BORDER} !important;
}}

/* ── Remove column default padding gaps ── */
[data-testid="stHorizontalBlock"] {{
    gap: 1rem !important;
    padding: 0 1.5rem 60px !important;
    margin-top: 0.6rem !important;
}}
[data-testid="stHorizontalBlock"] > div {{
    padding: 0 !important;
}}

/* ── Scrollable container styling ── */
[data-testid="stVerticalBlockBorderWrapper"] {{
    border: none !important;
    background: transparent !important;
}}
</style>
""", unsafe_allow_html=True)


# ── Plotly theme helpers ────────────────────────────────────────────────────────

def _plt_layout(height: int = 240, margin_top: int = 12) -> dict:
    return dict(
        height=height,
        margin=dict(t=margin_top, b=20, l=12, r=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=SURFACE,
        font=dict(family="Barlow, sans-serif", size=10, color=TEXT),
        showlegend=True,
        legend=dict(
            font=dict(family="Barlow Condensed, sans-serif", size=10, color=MUTED),
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom", y=-0.25, xanchor="center", x=0.5,
        ),
        colorway=PLOTLY_COLORS,
        xaxis=dict(
            gridcolor=BORDER, gridwidth=0.5,
            zerolinecolor=BORDER,
            tickfont=dict(family="Share Tech Mono, monospace", size=9, color=MUTED),
        ),
        yaxis=dict(
            gridcolor=BORDER, gridwidth=0.5,
            zerolinecolor=BORDER,
            tickfont=dict(family="Share Tech Mono, monospace", size=9, color=MUTED),
        ),
    )


# ── Helper functions ────────────────────────────────────────────────────────────

def parse_holdings_text(text: str) -> list[dict]:
    holdings = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) < 3:
            continue
        try:
            holdings.append({
                "ticker": parts[0].upper(),
                "shares": float(parts[1]),
                "cost": float(parts[2]),
            })
        except ValueError:
            continue
    return holdings


def _risk_level(score: float) -> tuple[str, str]:
    """Returns (label, css_class)."""
    if score < 40:
        return "LOW RISK", "rl-low"
    elif score < 70:
        return "MODERATE", "rl-mid"
    else:
        return "HIGH RISK", "rl-high"


def _risk_color(score: float) -> str:
    if score < 40:
        return LOW
    elif score < 70:
        return MID
    return HIGH


def _avg_corr(matrix: dict) -> float:
    vals = [v for r in matrix.values() for k, v in r.items() if k != list(r.keys())[0]]
    return sum(vals) / len(vals) if vals else 0.0


def compute_fallback_risk(metrics) -> RiskOutput:
    score = 50
    score += int((metrics.portfolio_beta - 1.0) * 25)
    score += int(metrics.avg_pairwise_correlation * 20)
    if metrics.drawdowns:
        worst = min(d.max_drawdown for d in metrics.drawdowns)
        score += int(abs(worst) * 15)
    score = max(0, min(100, score))
    return RiskOutput(
        risk_score=float(score),
        sector_concentration={e.sector: e.portfolio_weight for e in metrics.sector_exposures},
        correlation_matrix=metrics.correlation_matrix,
        portfolio_beta=metrics.portfolio_beta,
        max_drawdowns={d.ticker: d.max_drawdown for d in metrics.drawdowns},
        flags=[],
    )


# ── Header ─────────────────────────────────────────────────────────────────────

def _render_header() -> None:
    st.markdown("""
<div class="app-header">
    <div class="app-header-gem"></div>
    <div class="app-header-title">MERIDIAN</div>
    <div class="app-header-divider"></div>
    <div class="app-header-sub" style="margin-left:0">Portfolio Intelligence</div>
    <div class="app-header-sub">MARKET ADVISOR v1.0</div>
</div>
""", unsafe_allow_html=True)


# ── News ticker ─────────────────────────────────────────────────────────────────

def _render_ticker(holdings_sentiment: list | None) -> None:
    if not holdings_sentiment:
        items_html = "".join([
            f'<span class="ticker-item"><span class="ti-sym">{sym}</span>'
            f'<span class="ti-score-neu">AWAITING ANALYSIS</span>'
            f'<span class="ti-dot">◆</span></span>'
            for sym in ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "JPM"]
        ])
    else:
        def _item(hs) -> str:
            s = hs.sentiment_score
            if s > 0.05:
                score_cls = "ti-score-pos"
                score_txt = f"+{s:.2f}"
            elif s < -0.05:
                score_cls = "ti-score-neg"
                score_txt = f"{s:.2f}"
            else:
                score_cls = "ti-score-neu"
                score_txt = f"{s:+.2f}"
            summary = hs.summary[:80] + "…" if len(hs.summary) > 80 else hs.summary
            return (
                f'<span class="ticker-item">'
                f'<span class="ti-sym">{hs.ticker}</span>'
                f'<span class="{score_cls}">{score_txt}</span>'
                f'&nbsp;{summary}'
                f'<span class="ti-dot">◆</span></span>'
            )
        items_html = "".join(_item(hs) for hs in holdings_sentiment)

    # Duplicate track for seamless loop
    track = items_html * 2

    st.markdown(f"""
<div class="ticker-wrap">
    <div class="ticker-badge">MARKET INTEL</div>
    <div class="ticker-scroll">
        <div class="ticker-track">{track}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Portfolio table ─────────────────────────────────────────────────────────────

def _render_portfolio_table(holdings: list[dict]) -> None:
    rows = "".join(
        f"<tr>"
        f"<td>{h['ticker']}</td>"
        f"<td>{h['shares']:.0f}</td>"
        f"<td>${h['cost']:.2f}</td>"
        f"</tr>"
        for h in holdings
    )
    total_cost = sum(h["shares"] * h["cost"] for h in holdings)
    st.markdown(f"""
<div class="sec-lbl">Portfolio Holdings</div>
<div class="port-wrap">
<table class="port-table">
<thead><tr><th>Ticker</th><th>Shares</th><th>Cost Basis</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<div class="port-footer">{len(holdings)} positions · estimated cost ${total_cost:,.0f}</div>
</div>
""", unsafe_allow_html=True)


# ── Left panel — summary, advisory, recommendations ────────────────────────────

def _render_left_analysis(advisory: dict | None, risk: RiskOutput) -> None:
    if advisory:
        summary = advisory.get("summary", "")
        recs = advisory.get("recommendations", [])
    else:
        summary = None
        recs = []

    # ── Summary ──
    st.markdown('<div class="sec-lbl">Analysis Summary</div>', unsafe_allow_html=True)
    if summary:
        st.markdown(f"""
<div class="adv-block">
    <div class="adv-block-title">Portfolio Overview</div>
    <p>{summary}</p>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class="adv-block">
    <div class="adv-block-title">Portfolio Overview</div>
    <p style="color:{MUTED};font-style:italic">LLM advisory unavailable — showing computed metrics only.</p>
</div>
""", unsafe_allow_html=True)

    # ── Recommendations ──
    if recs:
        st.markdown('<div class="sec-lbl">Recommendations</div>', unsafe_allow_html=True)
        items_html = "".join(
            f'<div class="rec-item">'
            f'<span class="rec-n">0{i}.</span>'
            f'<span class="rec-t">{rec}</span>'
            f'</div>'
            for i, rec in enumerate(recs, 1)
        )
        st.markdown(f"""
<div class="rec-block">
    <div class="adv-block-title" style="color:{BLUE}">Action Items</div>
    {items_html}
</div>
""", unsafe_allow_html=True)

    # ── Risk flags (critical risks + warnings from the flags list) ──
    all_flags = risk.flags if risk.flags else []
    critical = [f for f in all_flags if f.category == "critical_risk"]
    warnings = [f for f in all_flags if f.category == "warning"]

    if critical or warnings:
        st.markdown('<div class="sec-lbl">Risk Flags</div>', unsafe_allow_html=True)
        flags_html = ""
        for f in critical:
            flags_html += f'<div class="flag flag-critical">⚑ {f.message}</div>'
        for w in warnings:
            flags_html += f'<div class="flag flag-warning">⚐ {w.message}</div>'
        st.markdown(flags_html, unsafe_allow_html=True)


# ── Right panel — charts & risk ────────────────────────────────────────────────

def _render_right_panel(risk: RiskOutput, container) -> None:
    with container:
        score = risk.risk_score
        lvl_label, lvl_cls = _risk_level(score)
        risk_color = _risk_color(score)

        # ── Risk score card ──
        st.markdown(f"""
<div class="risk-score-card">
    <div class="risk-num" style="color:{risk_color}">{score:.0f}<span class="risk-denom">/100</span></div>
    <div class="risk-lbl">Portfolio Risk Score</div>
    <div class="risk-lvl {lvl_cls}">{lvl_label}</div>
</div>
""", unsafe_allow_html=True)

        # ── Mini metrics ──
        avg_c = _avg_corr(risk.correlation_matrix)
        worst_dd = min(risk.max_drawdowns.values(), default=0.0)
        st.markdown(f"""
<div class="mini-metrics">
    <div class="mm-cell">
        <div class="mm-lbl">Beta</div>
        <div class="mm-val">{risk.portfolio_beta:.2f}</div>
    </div>
    <div class="mm-cell">
        <div class="mm-lbl">Avg Corr</div>
        <div class="mm-val">{avg_c:.2f}</div>
    </div>
    <div class="mm-cell">
        <div class="mm-lbl">Worst DD</div>
        <div class="mm-val">{worst_dd:.1%}</div>
    </div>
</div>
""", unsafe_allow_html=True)

        # ── Gauge ──
        st.markdown('<div class="chart-hd">Risk Gauge</div>', unsafe_allow_html=True)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"family": "Share Tech Mono", "color": risk_color, "size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 0.5, "tickcolor": MUTED,
                          "tickfont": {"family": "Share Tech Mono", "size": 8}},
                "bar": {"color": risk_color, "thickness": 0.22},
                "bgcolor": SURFACE,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40],  "color": "rgba(90,150,96,0.18)"},
                    {"range": [40, 70], "color": "rgba(196,144,48,0.18)"},
                    {"range": [70, 100],"color": "rgba(188,74,54,0.18)"},
                ],
            },
        ))
        gauge.update_layout(**_plt_layout(height=180, margin_top=8))
        gauge.update_layout(showlegend=False)
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

        # ── Sector concentration ──
        if risk.sector_concentration:
            st.markdown('<div class="chart-hd">Sector Concentration</div>', unsafe_allow_html=True)
            sec_df = pd.DataFrame(
                list(risk.sector_concentration.items()),
                columns=["Sector", "Weight"],
            ).sort_values("Weight", ascending=False)
            fig_sec = px.pie(
                sec_df, values="Weight", names="Sector",
                hole=0.42,
                color_discrete_sequence=PLOTLY_COLORS,
            )
            fig_sec.update_traces(
                textposition="inside",
                textinfo="percent+label",
                textfont=dict(family="Barlow Condensed", size=10, color=PANEL),
                insidetextorientation="horizontal",
                automargin=True,
            )
            _sec_layout = _plt_layout(height=300)
            _sec_layout["margin"] = dict(t=16, b=16, l=16, r=16)
            _sec_layout["legend"].update(y=-0.18, yanchor="top")
            fig_sec.update_layout(**_sec_layout)
            st.plotly_chart(fig_sec, use_container_width=True, config={"displayModeBar": False})

        # ── Correlation heatmap ──
        if risk.correlation_matrix:
            st.markdown('<div class="chart-hd">Correlation Heatmap</div>', unsafe_allow_html=True)
            corr_df = pd.DataFrame(risk.correlation_matrix).fillna(0.0)
            fig_corr = px.imshow(
                corr_df,
                color_continuous_scale=[[0, BLUE], [0.5, SURFACE], [1, HIGH]],
                zmin=-1, zmax=1,
                text_auto=".2f",
            )
            fig_corr.update_traces(
                textfont=dict(family="Share Tech Mono", size=8),
            )
            fig_corr.update_xaxes(side="top", tickangle=-30)
            fig_corr.update_layout(**_plt_layout(height=240))
            fig_corr.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})

        # ── Max drawdowns ──
        if risk.max_drawdowns:
            st.markdown('<div class="chart-hd">Max Drawdown by Holding</div>', unsafe_allow_html=True)
            dd_df = pd.DataFrame(
                list(risk.max_drawdowns.items()),
                columns=["Ticker", "Drawdown"],
            )
            dd_df["Drawdown_abs"] = dd_df["Drawdown"].abs()
            dd_df = dd_df.sort_values("Drawdown_abs", ascending=True)
            fig_dd = px.bar(
                dd_df, x="Drawdown_abs", y="Ticker",
                orientation="h",
                text=dd_df["Drawdown_abs"].map("{:.1%}".format),
                color="Drawdown_abs",
                color_continuous_scale=[[0, LOW], [0.5, MID], [1, HIGH]],
            )
            fig_dd.update_traces(
                textfont=dict(family="Share Tech Mono", size=9),
                textposition="outside",
            )
            fig_dd.update_layout(**_plt_layout(height=max(180, len(dd_df) * 28 + 30)))
            fig_dd.update_layout(
                xaxis_tickformat=".0%",
                coloraxis_showscale=False,
                yaxis=dict(tickfont=dict(family="Barlow Condensed", size=10, color=MUTED)),
            )
            st.plotly_chart(fig_dd, use_container_width=True, config={"displayModeBar": False})


def _render_right_pending(container) -> None:
    with container:
        st.markdown("""
<div class="pending-state">
    <div class="pending-icon">◈</div>
    <div class="pending-txt">Awaiting Analysis</div>
</div>
""", unsafe_allow_html=True)


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="MERIDIAN · Portfolio Intelligence",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()
    _render_header()

    # ── Sidebar ─────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="sidebar-head">Portfolio Input</div>', unsafe_allow_html=True)

        input_mode = st.radio(
            "Mode",
            ["Preset portfolio", "Custom holdings"],
            label_visibility="collapsed",
        )

        portfolio_data: dict | None = None

        if input_mode == "Preset portfolio":
            preset_key = st.selectbox(
                "Preset",
                list(SAMPLE_PORTFOLIOS.keys()),
                format_func=lambda k: k.replace("_", " ").title(),
            )
            portfolio_data = SAMPLE_PORTFOLIOS[preset_key]
        else:
            text_input = st.text_area(
                "Holdings",
                value="AAPL, 10, 150\nMSFT, 5, 260\nNVDA, 3, 180",
                height=180,
                help="TICKER, SHARES, COST — one per line",
            )
            portfolio_data = {"holdings": parse_holdings_text(text_input)}

        st.markdown("---")

        if not portfolio_data or not portfolio_data.get("holdings"):
            st.warning("No valid holdings.")
        else:
            if st.button("▶  Run Analysis"):
                st.session_state.run_analysis = True
                st.session_state.portfolio_data = portfolio_data
                st.session_state.analysis_done = False
                st.session_state.analysis_result = None
                st.session_state.analysis_error = None

        st.markdown("""
<div style="margin-top:1.5rem; font-size:0.68rem; color:{muted}; font-family:'Barlow Condensed',sans-serif; letter-spacing:0.12em; line-height:1.7;">
    Powered by Gemini · yfinance · LangGraph
</div>
""".format(muted=MUTED), unsafe_allow_html=True)

    # ── Resolve portfolio ────────────────────────────────────────────────────────
    active_portfolio = st.session_state.get("portfolio_data", portfolio_data) or portfolio_data
    holdings = active_portfolio.get("holdings", []) if active_portfolio else []

    # ── Run analysis ─────────────────────────────────────────────────────────────
    if st.session_state.get("run_analysis") and not st.session_state.get("analysis_done"):
        _overlay = st.empty()
        _overlay.markdown("""
<div id="analysis-overlay">
  <div class="card">
    <div class="spinner-ring"></div>
    <p class="label">Running portfolio analysis — please wait…</p>
  </div>
</div>
""", unsafe_allow_html=True)
        try:
            advisory_obj = run_analysis(active_portfolio)
            st.session_state.analysis_result = {
                "risk": advisory_obj.risk,
                "advisory": {
                    "summary": advisory_obj.summary,
                    "recommendations": advisory_obj.recommendations,
                },
                "market_intel": advisory_obj.market_intel,
                "error": None,
            }
        except Exception as exc:
            try:
                metrics = compute_metrics(active_portfolio)
                fallback = compute_fallback_risk(metrics)
                st.session_state.analysis_result = {
                    "risk": fallback,
                    "advisory": None,
                    "market_intel": None,
                    "error": str(exc),
                }
            except Exception as exc2:
                st.session_state.analysis_result = {
                    "risk": None,
                    "advisory": None,
                    "market_intel": None,
                    "error": f"{exc} | {exc2}",
                }
        finally:
            _overlay.empty()
        st.session_state.analysis_done = True

    result = st.session_state.get("analysis_result")

    # ── News ticker ───────────────────────────────────────────────────────────────
    intel: MarketIntelOutput | None = result["market_intel"] if result else None
    _render_ticker(intel.holdings_sentiment if intel else None)

    # ── Two-column layout ─────────────────────────────────────────────────────────
    col_left, col_right = st.columns([0.65, 0.35])

    # ── Left column ──────────────────────────────────────────────────────────────
    with col_left:
        if holdings:
            _render_portfolio_table(holdings)

        if result:
            if result.get("error"):
                st.error(f"Partial results — {result['error']}")

            if result.get("risk"):
                _render_left_analysis(result.get("advisory"), result["risk"])
            else:
                st.error("Analysis failed — no risk data available.")

        elif not st.session_state.get("run_analysis"):
            st.markdown(f"""
<div style="padding:2rem 0;">
    <div class="sec-lbl">Instructions</div>
    <div style="font-size:0.85rem;line-height:1.7;color:{MUTED};margin-top:0.5rem;">
        Select a preset portfolio or enter custom holdings above, then click
        <strong style="color:{TEXT}">▶ Run Analysis</strong> to fetch live market data
        and generate a full risk assessment with LLM advisory.
    </div>
</div>
""", unsafe_allow_html=True)

    # ── Right column (scrollable) ─────────────────────────────────────────────────
    with col_right:
        # st.container(height=...) creates a scrollable box in Streamlit ≥ 1.31
        right_container = st.container(height=680, border=False)
        if result and result.get("risk"):
            _render_right_panel(result["risk"], right_container)
        else:
            _render_right_pending(right_container)


if __name__ == "__main__":
    main()
