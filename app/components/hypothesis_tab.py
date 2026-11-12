import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.data_ingestion.fred_loader import FredLoader
from src.data_ingestion.yahoo_loader import YahooFinanceLoader
from src.analytics.hypothesis_engine import HypothesisEngine
from app.components.ui_styles import get_institutional_chart_layout

def get_market_loaders():
    fred = FredLoader()
    yahoo = YahooFinanceLoader()
    return fred, yahoo

@st.cache_data(ttl=3600)
def load_combined_market_dataset():
    fred, yahoo = get_market_loaders()
    # Pull both datasets at daily calendar frequency to ensure fully aligned datetime indices
    df_macro = fred.get_macro_dataset(start_date="2018-01-01", freq="D")
    df_assets = yahoo.get_multi_asset_dataset(start_date="2018-01-01", freq="D")
    
    # Strip timezones and normalize datetimes to eliminate index join mismatch
    df_macro.index = pd.to_datetime(df_macro.index).tz_localize(None).normalize()
    df_assets.index = pd.to_datetime(df_assets.index).tz_localize(None).normalize()

    combined = pd.concat([df_macro, df_assets], axis=1).ffill().bfill()
    return combined

def get_series_unit(series_code: str) -> str:
    """Returns canonical institutional market unit for series."""
    if series_code in ["T10Y2Y", "FEDFUNDS", "DGS10", "DGS2", "UNRATE"]:
        return "%"
    elif series_code in ["CL=F"]:
        return "USD/bbl"
    elif series_code in ["GC=F"]:
        return "USD/oz"
    elif series_code in ["CPIAUCSL"]:
        return "Index"
    elif series_code in ["^GSPC", "^RUT"]:
        return "Points"
    return "Value"

def render_hypothesis_tab():
    st.markdown("### 🧪 Quantitative Hypothesis Testing Lab")
    st.caption("Econometric proof of market narratives: Stationarity (ADF), Granger Causality, Rolling Correlations & Regime Splits.")

    with st.spinner("Aligning multi-asset macroeconomic time-series..."):
        df = load_combined_market_dataset()

    series_display_map = {
        "CL=F": "Crude Oil Futures (CL=F)",
        "CPIAUCSL": "Consumer Price Index (CPIAUCSL)",
        "T10Y2Y": "10Y-2Y Yield Spread (T10Y2Y)",
        "FEDFUNDS": "Federal Funds Rate (FEDFUNDS)",
        "DGS10": "10-Year Treasury Yield (DGS10)",
        "DGS2": "2-Year Treasury Yield (DGS2)",
        "UNRATE": "Unemployment Rate (UNRATE)",
        "^GSPC": "S&P 500 Index (^GSPC)",
        "^RUT": "Russell 2000 (^RUT)",
        "GC=F": "Gold Futures (GC=F)"
    }
    available_cols = [c for c in series_display_map.keys() if c in df.columns]

    # Initialize preset session state defaults
    if "hyp_var_x" not in st.session_state:
        st.session_state["hyp_var_x"] = "CL=F"
    if "hyp_var_y" not in st.session_state:
        st.session_state["hyp_var_y"] = "CPIAUCSL"

    # Presets Row
    st.markdown("##### ⚡ Institutional Hypothesis Presets")
    preset_cols = st.columns(4)
    if preset_cols[0].button("🛢️ Oil -> CPI Inflation", use_container_width=True):
        st.session_state["hyp_var_x"] = "CL=F"
        st.session_state["hyp_var_y"] = "CPIAUCSL"
    if preset_cols[1].button("📉 10Y-2Y Spread -> S&P 500", use_container_width=True):
        st.session_state["hyp_var_x"] = "T10Y2Y"
        st.session_state["hyp_var_y"] = "^GSPC"
    if preset_cols[2].button("🏛️ Fed Funds -> 10Y Yield", use_container_width=True):
        st.session_state["hyp_var_x"] = "FEDFUNDS"
        st.session_state["hyp_var_y"] = "DGS10"
    if preset_cols[3].button("💼 Unemployment -> Russell 2000", use_container_width=True):
        st.session_state["hyp_var_x"] = "UNRATE"
        st.session_state["hyp_var_y"] = "^RUT"

    # Selectors
    c_sel1, c_sel2, c_params = st.columns([1.5, 1.5, 1])
    with c_sel1:
        current_x = st.session_state["hyp_var_x"]
        idx_x = available_cols.index(current_x) if current_x in available_cols else 0
        var_x_code = st.selectbox(
            "Predictor / Driver Variable (X):",
            available_cols,
            index=idx_x,
            key="hyp_sel_x",
            format_func=lambda c: series_display_map.get(c, c),
            help="Select exogenous macro indicator or driver time-series."
        )
        st.session_state["hyp_var_x"] = var_x_code

    with c_sel2:
        current_y = st.session_state["hyp_var_y"]
        idx_y = available_cols.index(current_y) if current_y in available_cols else (1 if len(available_cols) > 1 else 0)
        var_y_code = st.selectbox(
            "Target / Response Variable (Y):",
            available_cols,
            index=idx_y,
            key="hyp_sel_y",
            format_func=lambda c: series_display_map.get(c, c),
            help="Select response asset price or macroeconomic target."
        )
        st.session_state["hyp_var_y"] = var_y_code

    with c_params:
        max_lags = st.slider("Max Granger Lags:", min_value=1, max_value=10, value=5, help="Maximum number of historical trading lags to test for predictive causality.")
        significance_alpha = st.select_slider("Significance (α):", options=[0.01, 0.05, 0.10], value=0.05, help="Type I error significance threshold for rejecting H0.")

    engine = HypothesisEngine(significance_level=significance_alpha)

    x_series = df[var_x_code].dropna()
    y_series = df[var_y_code].dropna()
    x_series.name = series_display_map.get(var_x_code, var_x_code)
    y_series.name = series_display_map.get(var_y_code, var_y_code)

    unit_x = get_series_unit(var_x_code)
    unit_y = get_series_unit(var_y_code)

    # Convert to explicit python date strings and floats to prevent binary buffer (bdata) corruption in Plotly.js
    dates_x = [d.strftime("%Y-%m-%d") for d in x_series.index]
    values_x = [round(float(v), 4) for v in x_series.values]
    
    dates_y = [d.strftime("%Y-%m-%d") for d in y_series.index]
    values_y = [round(float(v), 4) for v in y_series.values]

    # 1. Historical Trajectory Chart with Strict Dual-Axis Isolation
    c_title, c_toggle = st.columns([2, 1])
    with c_title:
        st.markdown("#### 📈 Historical Trajectory & Co-Movement")
    with c_toggle:
        chart_mode = st.radio("Display Mode:", ["Dual-Axis Scales", "Rebased Index (Base=100)"], horizontal=True)

    if chart_mode == "Rebased Index (Base=100)":
        base_x = values_x[0] if len(values_x) > 0 else 1.0
        base_y = values_y[0] if len(values_y) > 0 else 1.0
        
        # Safe normalization guarding against initial near-zero or negative spread denominators
        if abs(base_x) < 0.05:
            norm_x = [round(100.0 + (v - base_x) * 100.0, 2) for v in values_x]
        else:
            norm_x = [round((v / base_x) * 100.0, 2) for v in values_x]

        if abs(base_y) < 0.05:
            norm_y = [round(100.0 + (v - base_y) * 100.0, 2) for v in values_y]
        else:
            norm_y = [round((v / base_y) * 100.0, 2) for v in values_y]

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=dates_x,
            y=norm_x,
            name=f"X: {x_series.name} (Base 100)",
            line=dict(color="#38bdf8", width=2.2),
            hovertemplate="<b>%{x}</b><br>X Index: %{y:.2f}<extra></extra>"
        ))
        fig_ts.add_trace(go.Scatter(
            x=dates_y,
            y=norm_y,
            name=f"Y: {y_series.name} (Base 100)",
            line=dict(color="#f43f5e", width=2.2, dash="dot"),
            hovertemplate="<b>%{x}</b><br>Y Index: %{y:.2f}<extra></extra>"
        ))
        chart_layout = get_institutional_chart_layout(title="Rebased Trajectory Comparison (Base = 100)", height=350)
        chart_layout["yaxis"]["title"] = "Normalized Index (Base = 100)"
        chart_layout["margin"] = dict(l=65, r=65, t=44, b=24)
        fig_ts.update_layout(**chart_layout)
    else:
        # Institutional Dual-Axis: Predictor X strictly on Left Y-Axis, Target Y strictly on Right Y-Axis
        fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Left Y-Axis: Exogenous Predictor X (Cyan / Blue #38bdf8)
        fig_ts.add_trace(
            go.Scatter(
                x=dates_x,
                y=values_x,
                name=f"X: {x_series.name}",
                line=dict(color="#38bdf8", width=2.2),
                hovertemplate=f"<b>%{{x}}</b><br>X ({var_x_code}): %{{y:.2f}} {unit_x}<extra></extra>"
            ),
            secondary_y=False
        )
        
        # Right Y-Axis: Target Response Y (Rose / Red #f43f5e)
        fig_ts.add_trace(
            go.Scatter(
                x=dates_y,
                y=values_y,
                name=f"Y: {y_series.name}",
                line=dict(color="#f43f5e", width=2.2, dash="dot"),
                hovertemplate=f"<b>%{{x}}</b><br>Y ({var_y_code}): %{{y:.2f}} {unit_y}<extra></extra>"
            ),
            secondary_y=True
        )

        # Explicitly configure both left and right y-axes with independent scales and units
        fig_ts.update_yaxes(
            title_text=f"<b>{x_series.name} ({unit_x})</b>",
            title_font=dict(color="#38bdf8", size=12),
            tickfont=dict(family="JetBrains Mono", size=11, color="#38bdf8"),
            showgrid=True,
            gridcolor="#1e293b",
            side="left",
            secondary_y=False
        )
        fig_ts.update_yaxes(
            title_text=f"<b>{y_series.name} ({unit_y})</b>",
            title_font=dict(color="#f43f5e", size=12),
            tickfont=dict(family="JetBrains Mono", size=11, color="#f43f5e"),
            showgrid=False,
            side="right",
            secondary_y=True
        )

        chart_layout = get_institutional_chart_layout(title="Dual-Axis Historical Trajectory", height=350)
        # Strip default yaxis/yaxis2 from layout dict to avoid clobbering subplots configuration
        chart_layout.pop("yaxis", None)
        chart_layout.pop("yaxis2", None)
        chart_layout["margin"] = dict(l=75, r=75, t=44, b=24)
        fig_ts.update_layout(**chart_layout)

    st.plotly_chart(fig_ts, use_container_width=True)

    # 2. Run Comprehensive Econometric Suite via HypothesisEngine
    st.markdown("---")
    st.markdown("#### 🔬 Econometric Test Proofs & Critical Values")

    test_suite = engine.run_full_bivariate_suite(x_series, y_series, max_lags=max_lags)

    # Stationarity Cards (Strict separation of Raw Levels d=0 vs Differenced d=1)
    col_adf1, col_adf2 = st.columns(2)
    with col_adf1:
        adf_x = test_suite.get("stationarity_x", {})
        raw_is_stat_x = bool(adf_x.get("raw_is_stationary", False))
        raw_stat_x = adf_x.get("raw_statistic")
        raw_pval_x = adf_x.get("raw_p_value")
        diff_stat_x = adf_x.get("differenced_statistic")
        diff_pval_x = adf_x.get("differenced_p_value")
        diff_is_stat_x = bool(adf_x.get("differenced_is_stationary", False))
        interp_x = adf_x.get("interpretation", "Stationarity assessment complete.")

        raw_stat_str = f"{raw_stat_x:.4f}" if raw_stat_x is not None else "N/A"
        raw_p_str = f"{raw_pval_x:.4f}" if raw_pval_x is not None else "N/A"
        diff_stat_str = f"{diff_stat_x:.4f}" if diff_stat_x is not None else "N/A"
        diff_p_str = f"{diff_pval_x:.4e}" if (diff_pval_x is not None and diff_pval_x < 0.0001) else (f"{diff_pval_x:.4f}" if diff_pval_x is not None else "N/A")

        if raw_is_stat_x:
            badge_html = '<span class="badge-dovish">STATIONARY (Raw Levels d=0)</span>'
            stats_html = f"Raw ADF: <b>{raw_stat_str}</b> | p-value: <b>{raw_p_str}</b>"
        else:
            diff_badge = '<span class="badge-dovish" style="margin-left:6px;">STATIONARY (Differenced d=1)</span>' if diff_is_stat_x else '<span class="badge-hawkish" style="margin-left:6px;">NON-STATIONARY (d=1)</span>'
            badge_html = f'<span class="badge-hawkish">NON-STATIONARY (Raw Levels d=0)</span> {diff_badge}'
            stats_html = (
                f"Raw ADF: <b>{raw_stat_str}</b> (p={raw_p_str}) | "
                f"Diff ADF: <b>{diff_stat_str}</b> (p={diff_p_str})"
            )

        st.markdown(f"""
        <div class="macro-metric-card">
            <div class="metric-label">
                <span>ADF Unit Root Test: {x_series.name}</span>
                <span class="tooltip-icon" title="Augmented Dickey-Fuller test: H0 assumes unit root (non-stationary); Reject H0 when p < α">ℹ️</span>
            </div>
            <div style="margin: 8px 0;">{badge_html}</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #cbd5e1; margin-bottom: 6px;">
                {stats_html}
            </div>
            <div class="metric-subtext">{interp_x}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_adf2:
        adf_y = test_suite.get("stationarity_y", {})
        raw_is_stat_y = bool(adf_y.get("raw_is_stationary", False))
        raw_stat_y = adf_y.get("raw_statistic")
        raw_pval_y = adf_y.get("raw_p_value")
        diff_stat_y = adf_y.get("differenced_statistic")
        diff_pval_y = adf_y.get("differenced_p_value")
        diff_is_stat_y = bool(adf_y.get("differenced_is_stationary", False))
        interp_y = adf_y.get("interpretation", "Stationarity assessment complete.")

        raw_stat_str_y = f"{raw_stat_y:.4f}" if raw_stat_y is not None else "N/A"
        raw_p_str_y = f"{raw_pval_y:.4f}" if raw_pval_y is not None else "N/A"
        diff_stat_str_y = f"{diff_stat_y:.4f}" if diff_stat_y is not None else "N/A"
        diff_p_str_y = f"{diff_pval_y:.4e}" if (diff_pval_y is not None and diff_pval_y < 0.0001) else (f"{diff_pval_y:.4f}" if diff_pval_y is not None else "N/A")

        if raw_is_stat_y:
            badge_html_y = '<span class="badge-dovish">STATIONARY (Raw Levels d=0)</span>'
            stats_html_y = f"Raw ADF: <b>{raw_stat_str_y}</b> | p-value: <b>{raw_p_str_y}</b>"
        else:
            diff_badge_y = '<span class="badge-dovish" style="margin-left:6px;">STATIONARY (Differenced d=1)</span>' if diff_is_stat_y else '<span class="badge-hawkish" style="margin-left:6px;">NON-STATIONARY (d=1)</span>'
            badge_html_y = f'<span class="badge-hawkish">NON-STATIONARY (Raw Levels d=0)</span> {diff_badge_y}'
            stats_html_y = (
                f"Raw ADF: <b>{raw_stat_str_y}</b> (p={raw_p_str_y}) | "
                f"Diff ADF: <b>{diff_stat_str_y}</b> (p={diff_p_str_y})"
            )

        st.markdown(f"""
        <div class="macro-metric-card">
            <div class="metric-label">
                <span>ADF Unit Root Test: {y_series.name}</span>
                <span class="tooltip-icon" title="Augmented Dickey-Fuller test: H0 assumes unit root (non-stationary); Reject H0 when p < α">ℹ️</span>
            </div>
            <div style="margin: 8px 0;">{badge_html_y}</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #cbd5e1; margin-bottom: 6px;">
                {stats_html_y}
            </div>
            <div class="metric-subtext">{interp_y}</div>
        </div>
        """, unsafe_allow_html=True)

    # 3. Granger Causality & Correlations
    gc_res = test_suite.get("granger_x_causes_y", {})
    corr_res = test_suite.get("correlation", {})

    col_gc, col_corr = st.columns([1.6, 1.4])
    with col_gc:
        st.markdown(f"##### ⏱️ Granger Causality: Does X Precede Y? [Best: Lag {gc_res.get('best_lag', 1)}]")
        st.info(gc_res.get("interpretation", "Granger causality analysis complete."))

        if "lag_details" in gc_res and gc_res["lag_details"]:
            lag_df = pd.DataFrame.from_dict(gc_res["lag_details"], orient="index")
            best_lag = gc_res.get("best_lag", 1)

            fig_lags = go.Figure()
            # F-Statistic bar trace
            bar_colors = [
                "#38bdf8" if l == best_lag and s else ("#10b981" if s else "#475569")
                for l, s in zip(lag_df["lag"], lag_df["is_significant"])
            ]
            fig_lags.add_trace(go.Bar(
                x=[f"Lag {lag}" for lag in lag_df["lag"]],
                y=[float(f) for f in lag_df["f_stat"]],
                name="F-Statistic",
                marker=dict(color=bar_colors),
                hovertemplate="<b>%{x}</b><br>F-Stat: %{y:.3f}<extra></extra>"
            ))
            # p-value line trace
            fig_lags.add_trace(go.Scatter(
                x=[f"Lag {lag}" for lag in lag_df["lag"]],
                y=[float(p) for p in lag_df["p_value"]],
                name="p-value",
                yaxis="y2",
                mode="lines+markers",
                line=dict(color="#f59e0b", width=2.5),
                marker=dict(size=8, color=["#f59e0b" if l != best_lag else "#fbbf24" for l in lag_df["lag"]]),
                hovertemplate="<b>%{x}</b><br>p-value: %{y:.4f}<extra></extra>"
            ))
            # Decision boundary horizontal line
            fig_lags.add_hline(
                y=significance_alpha,
                yref="y2",
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"Decision Boundary α={significance_alpha}",
                annotation_position="top right"
            )

            layout_lags = get_institutional_chart_layout(title="Lagged F-Statistic & Significance Curve", height=270)
            layout_lags["yaxis"]["title"] = "F-Statistic"
            layout_lags["yaxis2"] = dict(title="p-value", overlaying="y", side="right", range=[-0.02, 1.05], showgrid=False, color="#f59e0b")
            fig_lags.update_layout(**layout_lags)
            st.plotly_chart(fig_lags, use_container_width=True)

    with col_corr:
        st.markdown("##### 📐 Correlation & Regime Split t-test")
        
        # Pearson metric card
        p_val_corr = corr_res.get('pearson_p_value', 1.0)
        p_str_corr = f"{p_val_corr:.4e}" if p_val_corr < 0.0001 else f"{p_val_corr:.4f}"
        st.markdown(f"""
        <div class="macro-metric-card">
            <div class="metric-label">
                <span>Pearson Linear Correlation (r)</span>
                <span class="tooltip-icon" title="Parametric linear association metric with Fisher Z 95% confidence interval">ℹ️</span>
            </div>
            <div class="metric-value">{corr_res.get('pearson_r', 'N/A')}</div>
            <div class="metric-subtext">p-value: {p_str_corr} | 95% CI: [{corr_res.get('ci_95', [0,0])[0]}, {corr_res.get('ci_95', [0,0])[1]}]</div>
        </div>
        """, unsafe_allow_html=True)

        # Spearman metric card
        s_val_corr = corr_res.get('spearman_p_value', 1.0)
        s_str_corr = f"{s_val_corr:.4e}" if s_val_corr < 0.0001 else f"{s_val_corr:.4f}"
        st.markdown(f"""
        <div class="macro-metric-card">
            <div class="metric-label">
                <span>Spearman Rank Correlation (ρ)</span>
                <span class="tooltip-icon" title="Non-parametric monotonic rank association unaffected by non-linear scalings">ℹ️</span>
            </div>
            <div class="metric-value">{corr_res.get('spearman_rho', 'N/A')}</div>
            <div class="metric-subtext">p-value: {s_str_corr} (Monotonic rank association)</div>
        </div>
        """, unsafe_allow_html=True)

        # Cross-Regime Welch t-test (Computed canonically by HypothesisEngine)
        t_res = test_suite.get("regime_t_test", {})
        split_thresh = t_res.get("median_split_threshold", 0.0)
        t_val = t_res.get('statistic', 0.0)
        t_p_val = t_res.get('p_value', 1.0)
        t_p_str = f"{t_p_val:.4e}" if t_p_val < 0.0001 else f"{t_p_val:.4f}"

        st.markdown(f"""
        <div class="macro-metric-card">
            <div class="metric-label">
                <span>Two-Sample Regime Welch t-test</span>
                <span class="tooltip-icon" title="Welch t-test assessing if mean target value differs significantly above vs below median predictor">ℹ️</span>
            </div>
            <div style="font-size: 0.92rem; font-weight: 600; color: #f1f5f9; margin-bottom: 4px;">
                Split on Median {x_series.name} ({split_thresh:.2f} {unit_x})
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #cbd5e1;">
                High Mean: <b>{t_res.get('sample_a_mean', 0)} {unit_y}</b> | Low Mean: <b>{t_res.get('sample_b_mean', 0)} {unit_y}</b><br>
                Welch t-stat: <b>{t_val}</b> (p={t_p_str})
            </div>
            <div class="metric-subtext">{t_res.get('interpretation', '')}</div>
        </div>
        """, unsafe_allow_html=True)

    # 4. Rolling 60-Day Correlation Dynamics (Computed canonically by HypothesisEngine with .resample('D').ffill())
    st.markdown("##### 🌊 Rolling 60-Day Correlation Dynamics")
    rolling_corr = test_suite.get("rolling_correlation")

    if rolling_corr is not None and len(rolling_corr) >= 10:
        rc_dates = [d.strftime("%Y-%m-%d") for d in rolling_corr.index]
        rc_values = [round(float(v), 4) for v in rolling_corr.values]

        fig_rc = go.Figure()
        fig_rc.add_trace(go.Scatter(
            x=rc_dates,
            y=rc_values,
            mode="lines",
            line=dict(color="#38bdf8", width=2.2),
            name="60-Day Rolling Correlation",
            hovertemplate="<b>%{x}</b><br>Correlation (r): %{y:.3f}<extra></extra>"
        ))
        fig_rc.add_hline(y=0.0, line_dash="dash", line_color="#64748b")
        fig_rc.add_hline(y=0.5, line_dash="dot", line_color="rgba(16, 185, 129, 0.4)")
        fig_rc.add_hline(y=-0.5, line_dash="dot", line_color="rgba(244, 63, 94, 0.4)")
        
        rc_layout = get_institutional_chart_layout(title="60-Day Rolling Bivariate Correlation Timeline", height=240)
        rc_layout["yaxis"]["range"] = [-1.05, 1.05]
        rc_layout["yaxis"]["title"] = "Correlation (r)"
        rc_layout["margin"] = dict(l=65, r=25, t=38, b=24)
        fig_rc.update_layout(**rc_layout)
        st.plotly_chart(fig_rc, use_container_width=True)
    else:
        st.warning("Insufficient overlapping observations to construct a rolling 60-day correlation timeline.")

    # 5. Bidirectional Granger check
    with st.expander("🔄 Bidirectional Feedback: Does Target Y Granger-cause Predictor X?"):
        gc_rev = test_suite.get("granger_y_causes_x", {})
        st.write(gc_rev.get("interpretation", "No reverse Granger causality analysis available."))
        if gc_rev.get("is_significant", False):
            st.warning("⚠️ Simultaneous feedback detected (endogeneity / bi-directional causality). Both variables influence each other.")
        else:
            st.success("✅ Uni-directional relationship confirmed. No reverse Granger causation from Y to X.")
