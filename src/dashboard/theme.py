"""Design system for the dashboard: one palette, one plotly template, one CSS layer.

Everything visual flows from the tokens defined here so every page shares the same
language — deep slate neutrals, a single indigo primary, cyan as the secondary series
color, and semantic green/amber/rose reserved for meaning (growth, caution, anomaly),
never used decoratively.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ------------------------------------------------------------------------ color tokens
# One palette drives every surface, chart, and semantic accent. Neutrals carry the
# structure; the indigo primary and cyan accent carry the data; green/amber/red are
# reserved for meaning (growth, caution, anomaly) and never used decoratively.

INK = "#111827"          # primary text, headings
TEXT = "#374151"         # body copy
MUTED = "#6B7280"        # captions, axis labels, secondary text
BORDER = "#E5E7EB"       # hairline borders
SURFACE = "#FFFFFF"      # cards
CANVAS = "#F5F7FA"       # app background

PRIMARY = "#4F46E5"      # indigo — primary data series, actions, focus
PRIMARY_SOFT = "#EEF2FF"
ACCENT = "#06B6D4"       # cyan — secondary data series
VIOLET = "#7C3AED"       # tertiary categorical series
SLATE = "#94A3B8"        # neutral context series (rolling means, baselines)

GOOD = "#10B981"         # growth, positive deltas, held-out actuals
WARN = "#F59E0B"         # caution states
BAD = "#EF4444"          # anomalies, negative deltas

# Soft tints paired to each semantic color, for callout and chip backgrounds.
GOOD_SOFT = "#ECFDF5"
WARN_SOFT = "#FFFBEB"
BAD_SOFT = "#FEF2F2"
NEUTRAL_SOFT = "#F8FAFC"

# Categorical colorway for multi-series charts. Cohesive cool-to-warm ordering rather
# than a rainbow; the semantic colors appear only at the tail, where a chart genuinely
# has enough distinct series to need them.
COLORWAY = [PRIMARY, ACCENT, VIOLET, GOOD, WARN, BAD]

# Grid and axis lines: a hair lighter than the border so gridlines recede behind data.
GRID = "#EEF1F5"

FONT_STACK = (
    "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "'Helvetica Neue', Arial, sans-serif"
)


# -------------------------------------------------------------------- plotly template


def register_plotly_template() -> None:
    """Register and activate the project plotly template. Idempotent."""
    pio.templates["demand_intel"] = go.layout.Template(
        layout=go.Layout(
            font=dict(family=FONT_STACK, size=12.5, color=TEXT),
            title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=COLORWAY,
            margin=dict(l=8, r=8, t=42, b=8),
            xaxis=dict(
                gridcolor=GRID, linecolor=BORDER, zeroline=False,
                tickfont=dict(color=MUTED),
            ),
            yaxis=dict(
                gridcolor=GRID, linecolor=BORDER, zeroline=False,
                tickfont=dict(color=MUTED),
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                font=dict(size=12, color=TEXT), bgcolor="rgba(0,0,0,0)",
            ),
            hoverlabel=dict(
                bgcolor=INK, font=dict(family=FONT_STACK, size=12, color="#F1F5F9"),
                bordercolor=INK,
            ),
            hovermode="x unified",
        )
    )
    pio.templates.default = "demand_intel"


PLOTLY_CONFIG = {"displayModeBar": False}


# --------------------------------------------------------------------------- app CSS

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {{
    --radius: 12px;
    --radius-sm: 10px;
    --shadow-card: 0 1px 2px rgba(17, 24, 39, 0.04), 0 1px 3px rgba(17, 24, 39, 0.03);
    --shadow-hover: 0 2px 6px rgba(17, 24, 39, 0.06), 0 1px 2px rgba(17, 24, 39, 0.04);
}}

html, body, [data-testid="stAppViewContainer"] * {{
    font-family: {FONT_STACK};
}}

/* The blanket font override above must not capture Streamlit's icon ligature font. */
[data-testid="stIconMaterial"] {{
    font-family: "Material Symbols Rounded" !important;
}}

/* Layout rhythm */
.block-container {{
    padding-top: 1.4rem;
    padding-bottom: 4rem;
    max-width: 1200px;
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbar"], #MainMenu, footer {{ visibility: hidden; }}

/* Headings */
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}

/* Sidebar polish on top of the dark theme from config.toml */
[data-testid="stSidebar"] {{ border-right: 1px solid #1E293B; }}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a span {{ font-weight: 500; }}
[data-testid="stSidebarNav"]::before {{ content: ""; display: block; height: 4px; }}

/* Bordered containers read as cards */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {SURFACE};
    border: 1px solid {BORDER} !important;
    border-radius: var(--radius);
    box-shadow: var(--shadow-card);
}}

/* KPI cards — a subtle lift on hover signals the surface is a live tile, not static ink */
.kpi-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: var(--radius);
    padding: 17px 18px 15px 18px;
    box-shadow: var(--shadow-card);
    height: 100%;
    transition: box-shadow 140ms ease, transform 140ms ease;
}}
.kpi-card:hover {{ box-shadow: var(--shadow-hover); transform: translateY(-1px); }}
.kpi-label {{
    font-size: 11px; font-weight: 600; letter-spacing: 0.07em;
    text-transform: uppercase; color: {MUTED}; margin-bottom: 7px;
}}
.kpi-value {{
    font-size: 26px; font-weight: 700; color: {INK};
    line-height: 1.15; letter-spacing: -0.02em;
}}
.kpi-sub {{ font-size: 12.5px; color: {MUTED}; margin-top: 6px; }}
.kpi-sub .pos {{ color: {GOOD}; font-weight: 600; }}
.kpi-sub .neg {{ color: {BAD}; font-weight: 600; }}

/* Callouts */
.callout {{
    border-radius: var(--radius-sm); padding: 14px 16px; font-size: 13.5px;
    line-height: 1.55; color: {TEXT}; margin: 4px 0 8px 0;
    border: 1px solid {BORDER}; border-left-width: 3px;
}}
.callout b {{ color: {INK}; }}
.callout code {{
    background: rgba(17, 24, 39, 0.05); border-radius: 4px;
    padding: 1px 5px; font-size: 12.5px;
}}
.callout-info    {{ background: {PRIMARY_SOFT}; border-left-color: {PRIMARY}; }}
.callout-success {{ background: {GOOD_SOFT}; border-left-color: {GOOD}; }}
.callout-warning {{ background: {WARN_SOFT}; border-left-color: {WARN}; }}
.callout-neutral {{ background: {NEUTRAL_SOFT}; border-left-color: {SLATE}; }}

/* Page header */
.page-title {{
    font-size: 26px; font-weight: 700; color: {INK};
    letter-spacing: -0.02em; margin-bottom: 3px;
}}
.page-subtitle {{ font-size: 14px; color: {MUTED}; margin-bottom: 16px; }}

/* Section header */
.section-title {{
    font-size: 16px; font-weight: 650; color: {INK};
    margin: 6px 0 2px 0; letter-spacing: -0.01em;
}}
.section-caption {{ font-size: 12.5px; color: {MUTED}; margin-bottom: 8px; }}

/* Small badge chip */
.chip {{
    display: inline-block; padding: 2px 9px; border-radius: 999px;
    font-size: 11.5px; font-weight: 600; letter-spacing: 0.02em;
}}
.chip-indigo {{ background: {PRIMARY_SOFT}; color: {PRIMARY}; }}
.chip-green  {{ background: {GOOD_SOFT}; color: {GOOD}; }}
.chip-amber  {{ background: {WARN_SOFT}; color: {WARN}; }}
.chip-rose   {{ background: {BAD_SOFT}; color: {BAD}; }}
.chip-slate  {{ background: #F1F5F9; color: {MUTED}; }}

/* Tables */
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: var(--radius-sm); }}

/* Segmented control / pills: tighten the active state to the brand indigo */
[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-baseweb="pill"][aria-selected="true"] {{ font-weight: 600; }}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
