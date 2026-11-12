import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, Optional

def apply_institutional_theme():
    """
    Injects custom CSS styling for high-end institutional terminal aesthetics.
    Implements a bespoke palette: deep slate (#0a0e17), cyan (#38bdf8), emerald (#10b981),
    amber (#f59e0b), and rose (#f43f5e) with glassmorphism, overflow clipping, and JetBrains Mono metrics.
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Palette */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e2e8f0;
    }

    code, pre, .mono-text {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1440px;
    }

    /* Top Navigation / Hero Header */
    .quant-header-card {
        background: radial-gradient(circle at 10% 20%, #151d2f 0%, #0d131f 60%, #070a10 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        box-shadow: 0 12px 36px -8px rgba(0, 0, 0, 0.8), 0 0 20px rgba(56, 189, 248, 0.08);
        border-radius: 14px;
        padding: 20px 28px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        overflow: hidden;
    }

    .quant-title {
        font-size: 1.95rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 10px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .quant-subtitle {
        color: #94a3b8;
        font-size: 0.92rem;
        font-weight: 400;
        margin: 0;
    }

    /* Ticker Tape Bar */
    .ticker-bar {
        background: #0d1117;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 10px 18px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        overflow-x: auto;
        gap: 24px;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.5);
    }

    .ticker-item {
        display: flex;
        align-items: center;
        gap: 8px;
        white-space: nowrap;
        font-size: 0.82rem;
    }

    .ticker-symbol {
        font-weight: 700;
        color: #94a3b8;
        font-family: 'JetBrains Mono', monospace;
    }

    .ticker-price {
        font-weight: 600;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }

    .ticker-delta-pos {
        color: #10b981;
        font-weight: 600;
        font-size: 0.78rem;
        background: rgba(16, 185, 129, 0.12);
        padding: 2px 6px;
        border-radius: 4px;
    }

    .ticker-delta-neg {
        color: #f43f5e;
        font-weight: 600;
        font-size: 0.78rem;
        background: rgba(244, 63, 94, 0.12);
        padding: 2px 6px;
        border-radius: 4px;
    }

    /* Modern Glassmorphic Metric Cards with Layout Overflow Protection */
    .macro-metric-card {
        background: rgba(17, 24, 39, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(31, 41, 55, 0.9);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
        transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
        overflow: hidden;
        min-width: 0;
        box-sizing: border-box;
    }

    .macro-metric-card:hover {
        border-color: rgba(56, 189, 248, 0.6);
        box-shadow: 0 6px 24px rgba(56, 189, 248, 0.12);
        transform: translateY(-2px);
    }

    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .metric-value {
        font-size: clamp(1.2rem, 2.2vw, 1.55rem);
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #f8fafc;
        letter-spacing: -0.02em;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        line-height: 1.3;
    }

    .metric-subtext {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 5px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    /* Tooltip Helper */
    .tooltip-icon {
        cursor: help;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: #64748b;
        font-size: 0.72rem;
        margin-left: 4px;
        transition: color 0.15s ease;
    }
    .tooltip-icon:hover {
        color: #38bdf8;
    }

    /* Badges */
    .badge-hawkish {
        background-color: rgba(244, 63, 94, 0.16);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.35);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        display: inline-block;
        white-space: nowrap;
    }

    .badge-dovish {
        background-color: rgba(16, 185, 129, 0.16);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        display: inline-block;
        white-space: nowrap;
    }

    .badge-neutral {
        background-color: rgba(245, 158, 11, 0.16);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.35);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        display: inline-block;
        white-space: nowrap;
    }

    .badge-info {
        background-color: rgba(56, 189, 248, 0.14);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        display: inline-block;
        white-space: nowrap;
    }

    /* Section Cards */
    .section-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 20px;
        overflow: hidden;
    }

    /* Tab Controls */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #0b1120;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #1e293b;
        margin-bottom: 20px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 46px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.92rem;
        color: #94a3b8;
        padding: 0 20px;
        transition: all 0.15s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
        color: #38bdf8 !important;
        border-bottom: 2.5px solid #38bdf8 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }

    /* Status Dot */
    .pulse-dot {
        height: 10px;
        width: 10px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        box-shadow: 0 0 10px #10b981;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.02em;
        transition: all 0.15s ease;
    }
    </style>
    """, unsafe_allow_html=True)

def render_top_header():
    """
    Renders top header banner and real-time institutional market marquee tape.
    """
    st.markdown("""
    <div class="quant-header-card">
        <div>
            <div class="quant-title">⚡ QuantMacro AI</div>
            <div class="quant-subtitle">Institutional Macro Intelligence • Econometric Hypothesis Testing • Multi-Horizon ML • Stochastic Risk Engine</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.82rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
                <span class="pulse-dot"></span>SYSTEM: PRODUCTION ACTIVE
            </div>
            <div style="font-size: 0.74rem; color: #64748b; margin-top: 4px;">FASTAPI & CELERY READY • v1.0.0</div>
        </div>
    </div>

    <div class="ticker-bar">
        <div class="ticker-item">
            <span class="ticker-symbol">^GSPC (S&P 500)</span>
            <span class="ticker-price">5,815.20</span>
            <span class="ticker-delta-pos">+0.42%</span>
        </div>
        <div class="ticker-item">
            <span class="ticker-symbol">DGS10 (10Y Yield)</span>
            <span class="ticker-price">4.08%</span>
            <span class="ticker-delta-neg">-3 bps</span>
        </div>
        <div class="ticker-item">
            <span class="ticker-symbol">T10Y2Y (Yield Curve)</span>
            <span class="ticker-price">+0.12%</span>
            <span class="ticker-delta-pos">Bull Steep</span>
        </div>
        <div class="ticker-item">
            <span class="ticker-symbol">CL=F (Crude Oil)</span>
            <span class="ticker-price">$70.85</span>
            <span class="ticker-delta-neg">-1.15%</span>
        </div>
        <div class="ticker-item">
            <span class="ticker-symbol">GC=F (Gold)</span>
            <span class="ticker-price">$2,660.40</span>
            <span class="ticker-delta-pos">+0.74%</span>
        </div>
        <div class="ticker-item">
            <span class="ticker-symbol">FEDFUNDS</span>
            <span class="ticker-price">4.83%</span>
            <span class="ticker-delta-pos">Neutral</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def get_institutional_chart_layout(
    title: Optional[str] = None,
    height: int = 400,
    show_legend: bool = True
) -> Dict[str, Any]:
    """
    Returns a unified, clean institutional Plotly layout dictionary matching dark Bloomberg/Aladdin UI.
    Guarantees no 'undefined' headers appear in Plotly rendering.
    """
    clean_title = str(title).strip() if (title is not None and str(title).strip() not in ["None", "undefined", ""]) else None
    
    layout = dict(
        height=height,
        margin=dict(l=24, r=24, t=44 if clean_title else 22, b=24),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(11, 17, 32, 0.6)',
        font=dict(family='Inter', color="#cbd5e1", size=12),
        hoverlabel=dict(
            bgcolor="#1e293b",
            font_size=12,
            font_family="JetBrains Mono",
            bordercolor="#38bdf8"
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='#1e293b',
            gridwidth=1,
            zeroline=False,
            color='#94a3b8',
            tickfont=dict(family='JetBrains Mono', size=11)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#1e293b',
            gridwidth=1,
            zeroline=False,
            color='#94a3b8',
            tickfont=dict(family='JetBrains Mono', size=11)
        )
    )
    if clean_title:
        layout["title"] = dict(text=clean_title, font=dict(family='Inter', size=15, color='#f1f5f9'))
    else:
        layout["title"] = None

    if show_legend:
        layout["legend"] = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(15, 23, 42, 0.7)",
            bordercolor="#1e293b",
            borderwidth=1,
            font=dict(size=11)
        )
    return layout
