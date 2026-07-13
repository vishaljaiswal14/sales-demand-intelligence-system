"""Reusable presentation components: KPI cards, headers, callouts, chips.

All markup funnels through these helpers so spacing, typography, and card styling stay
identical on every page.
"""

from __future__ import annotations

import streamlit as st


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="page-title">{title}</div>'
        f'<div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def section(title: str, caption: str | None = None) -> None:
    html = f'<div class="section-title">{title}</div>'
    if caption:
        html += f'<div class="section-caption">{caption}</div>'
    st.markdown(html, unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str | None = None) -> None:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def kpi_row(cards: list[tuple[str, str, str | None]]) -> None:
    """A row of equal-width KPI cards. Each card is (label, value, sub)."""
    for column, (label, value, sub) in zip(st.columns(len(cards)), cards):
        with column:
            kpi_card(label, value, sub)


def callout(text: str, kind: str = "info") -> None:
    """Styled inline callout. ``kind``: info | success | warning | neutral."""
    st.markdown(f'<div class="callout callout-{kind}">{text}</div>', unsafe_allow_html=True)


def chip(text: str, tone: str = "slate") -> str:
    """Return badge HTML (for embedding in markdown). Tones: indigo, green, amber, rose, slate."""
    return f'<span class="chip chip-{tone}">{text}</span>'


def spacer(rem: float = 0.8) -> None:
    st.markdown(f'<div style="height:{rem}rem"></div>', unsafe_allow_html=True)


def fmt_money(value: float, compact: bool = True) -> str:
    if compact:
        if abs(value) >= 1_000_000:
            return f"${value / 1_000_000:.2f}M"
        if abs(value) >= 1_000:
            return f"${value / 1_000:.0f}K"
    return f"${value:,.0f}"


def fmt_pct(value: float, signed: bool = True) -> str:
    sign = "+" if signed and value >= 0 else ""
    return f"{sign}{value:.1f}%"
