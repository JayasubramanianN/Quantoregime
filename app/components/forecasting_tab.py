import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.data_ingestion.yahoo_loader import YahooFinanceLoader
from src.data_ingestion.fred_loader import FredLoader
from src.analytics.forecasting_engine import MacroForecastingEngine
from app.components.ui_styles import get_institutional_chart_layout

@st.cache_resource
def get_forecasting_engine():
    return MacroForecastingEngine()

def render_forecasting_tab():
    st.markdown("### 🔮 Multi-Horizon Forecasting Engine")
    st.caption("Prophet Bayesian trend-seasonality blended with XGBoost recursive autoregression & macro feature engineering.")

    engine = get_forecasting_engine()
    yahoo = YahooFinanceLoader()
    fred = FredLoader()

    # Asset Selection and Hyperparameter Controls
    col_a, col_h, col_w, col_s = st.columns([1.2, 1, 1.2, 1])
    with col_a:
        asset_options = {
            "^GSPC": "S&P 500 (^GSPC)",
            "^RUT": "Russell 2000 (^RUT)",
            "GC=F": "Gold Futures (GC=F)",
            "CL=F": "Crude Oil Futures (CL=F)",
            "BTC-USD": "Bitcoin USD (BTC-USD)"
        }
        selected_ticker = st.selectbox("Asset Benchmark:", list(asset_options.keys()), format_func=lambda k: asset_options[k], help="Select target asset for multi-horizon quantitative forecasting.")

    with col_h:
        horizon_days = st.select_slider(
            "Forecast Horizon:",
            options=[30, 60, 90, 180, 252, 365],
            value=180,
            format_func=lambda d: f"{d} Trading Days (~{d//21}M)",
            help="Number of future business days to forecast forward."
        )

    with col_w:
        prophet_pct = st.slider("Ensemble Blend (Prophet %):", min_value=0, max_value=100, value=40, step=5, help="Weighting allocated to Prophet trend/seasonality vs XGBoost macro feature regressor.")
        xgb_pct = 100 - prophet_pct
        st.caption(f"**Blend:** Prophet `{prophet_pct}%` | XGBoost `{xgb_pct}%`")

    with col_s:
        seasonality = st.selectbox("Seasonality Mode:", ["multiplicative", "additive"], help="Prophet seasonality modeling specification.")

    # Load series & exogenous macro indicators
    with st.spinner("Retrieving historical asset prices and macroeconomic regressors..."):
        asset_series = yahoo.get_asset_series(selected_ticker, start_date="2018-01-01")
        # Strip timezones for clean alignment
        asset_series.index = pd.to_datetime(asset_series.index).tz_localize(None).normalize()

        exog_macro = fred.get_macro_dataset(series_ids=["FEDFUNDS", "DGS10", "DGS2", "CPIAUCSL", "T10Y2Y"], start_date="2018-01-01", freq="B")
        exog_macro.index = pd.to_datetime(exog_macro.index).tz_localize(None).normalize()

    # Run Ensemble Forecasting
    with st.spinner("Fitting Prophet & XGBoost ensemble models..."):
        forecast_res = engine.run_ensemble_forecast(
            series=asset_series,
            exog_df=exog_macro,
            horizon_days=horizon_days,
            prophet_weight=prophet_pct / 100.0,
            xgboost_weight=xgb_pct / 100.0
        )

    # Performance KPI Cards
    col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
    with col_k1:
        st.markdown(f"""
        <div class="macro-metric-card">
            <div class="metric-label">
                <span>Spot Price</span>
                <span class="tooltip-icon" title="Most recent recorded closing price">ℹ️</span>
            </div>
            <div class="metric-value">${forecast_res['current_price']:,.2f}</div>
            <div class="metric-subtext">{selected_ticker} Latest Close</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k2:
        proj_p = forecast_res['projected_price']
        st.markdown(f"""
        <div class="macro-metric-card">
            <div class="metric-label">
                <span>Target ({horizon_days}d)</span>
                <span class="tooltip-icon" title="Projected expected price at the horizon terminal date">ℹ️</span>
            </div>
            <div class="metric-value" style="color: #38bdf8;">${proj_p:,.2f}</div>
            <div class="metric-subtext">Mean Expected Target</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k3:
        roi = forecast_res['expected_change_pct']
        roi_color = "#34d399" if roi >= 0 else "#fb7185"
        st.markdown(f"""
        <div class="macro-metric-card">
            <div class="metric-label">
                <span>Expected Return</span>
                <span class="tooltip-icon" title="Projected percentage capital change from spot to horizon target">ℹ️</span>
            </div>
            <div class="metric-value" style="color: {roi_color};">{roi:+.2f}%</div>
            <div class="metric-subtext">Cumulative Projection</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k4:
        st.markdown(f"""
        <div class="macro-metric-card">
            <div class="metric-label">
                <span>In-Sample RMSE</span>
                <span class="tooltip-icon" title="Root Mean Squared Error of ensemble backtest on historical data">ℹ️</span>
            </div>
            <div class="metric-value">${forecast_res['rmse']:,.2f}</div>
            <div class="metric-subtext">MAE: ${forecast_res['mae']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k5:
        st.markdown(f"""
        <div class="macro-metric-card">
            <div class="metric-label">
                <span>In-Sample MAPE</span>
                <span class="tooltip-icon" title="Mean Absolute Percentage Error on in-sample evaluations">ℹ️</span>
            </div>
            <div class="metric-value">{forecast_res['mape']:.2f}%</div>
            <div class="metric-subtext">Mean Abs % Error</div>
        </div>
        """, unsafe_allow_html=True)

    # Interactive Forecast Plotly Chart
    st.markdown("#### 📈 Multi-Horizon Ensemble Trajectory & 95% Confidence Corridor")

    c_ch1, c_ch2, _ = st.columns([1.2, 1.2, 1.6])
    with c_ch1:
        show_components = st.checkbox("Show Individual Sub-Models", value=True)
    with c_ch2:
        show_ribbon = st.checkbox("Show 95% Confidence Ribbon", value=True)

    fc_table = forecast_res["forecast_table"]
    # Slice historical to ~350 trading days
    hist_slice = fc_table[fc_table["actual"].notna()].iloc[-350:]
    last_hist_dt = hist_slice.index[-1]
    last_hist_val = float(hist_slice["actual"].iloc[-1])
    future_slice = fc_table[fc_table.index > last_hist_dt]

    fig = go.Figure()

    # 1. Historical Actual Price (Primary Y-Axis)
    fig.add_trace(go.Scatter(
        x=hist_slice.index,
        y=hist_slice["actual"],
        name="Historical Price",
        mode="lines",
        line=dict(color="#38bdf8", width=2.4),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Historical Price: $%{y:,.2f}<extra></extra>"
    ))

    # Seamless connection points for future curves
    connect_dates = [last_hist_dt] + future_slice.index.tolist()
    connect_ensemble = [last_hist_val] + future_slice["ensemble_forecast"].tolist()
    connect_lower = [last_hist_val] + future_slice["ensemble_lower_95"].tolist()
    connect_upper = [last_hist_val] + future_slice["ensemble_upper_95"].tolist()

    # 2. Shaded Confidence Interval Ribbon (Seamlessly anchored at last historical point)
    if show_ribbon and not future_slice.empty:
        fig.add_trace(go.Scatter(
            x=connect_dates + connect_dates[::-1],
            y=connect_upper + connect_lower[::-1],
            fill='toself',
            fillcolor='rgba(16, 185, 129, 0.16)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name="95% Confidence Corridor"
        ))

    # 3. Ensemble Forecast Path
    fig.add_trace(go.Scatter(
        x=connect_dates,
        y=connect_ensemble,
        name=f"Ensemble Blend ({prophet_pct}% P + {xgb_pct}% XGB)",
        mode="lines",
        line=dict(color="#10b981", width=3.2),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Ensemble: $%{y:,.2f}<extra></extra>"
    ))

    # 4. Standalone Sub-Models
    if show_components and not future_slice.empty:
        connect_prophet = [last_hist_val] + future_slice["prophet_hat"].tolist()
        fig.add_trace(go.Scatter(
            x=connect_dates,
            y=connect_prophet,
            name="Prophet Baseline",
            mode="lines",
            line=dict(color="#f59e0b", width=1.8, dash="dash"),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Prophet: $%{y:,.2f}<extra></extra>"
        ))

        connect_xgb = [last_hist_val] + future_slice["xgb_hat"].tolist()
        fig.add_trace(go.Scatter(
            x=connect_dates,
            y=connect_xgb,
            name="XGBoost Regressor",
            mode="lines",
            line=dict(color="#c084fc", width=1.8, dash="dot"),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>XGBoost: $%{y:,.2f}<extra></extra>"
        ))

    # Current Price Reference Line
    fig.add_hline(
        y=forecast_res["current_price"],
        line_dash="dot",
        line_color="#64748b",
        annotation_text=f"Current: ${forecast_res['current_price']:.2f}",
        annotation_position="bottom right"
    )

    # Dynamic Y-Axis Bounds (Guarantees historical line is never flattened or compressed)
    all_prices = pd.concat([
        hist_slice["actual"],
        future_slice["ensemble_forecast"],
        future_slice["ensemble_lower_95"],
        future_slice["ensemble_upper_95"]
    ]).dropna()
    min_p = float(all_prices.min())
    max_p = float(all_prices.max())

    chart_layout = get_institutional_chart_layout(title=f"{selected_ticker} Multi-Horizon Price Trajectory", height=450)
    chart_layout["yaxis"]["title"] = f"{selected_ticker} Price (USD)"
    chart_layout["yaxis"]["range"] = [min_p * 0.95, max_p * 1.05]
    chart_layout["yaxis"]["autorange"] = False
    fig.update_layout(**chart_layout)

    st.plotly_chart(fig, use_container_width=True)

    # Forecast Data Table & Export
    with st.expander("📋 Forward Projection Data Table & Export"):
        export_df = pd.DataFrame({
            "Date": forecast_res["future_dates"],
            "Ensemble_Mean": forecast_res["future_ensemble"],
            "Ensemble_Lower_95": forecast_res["future_lower"],
            "Ensemble_Upper_95": forecast_res["future_upper"],
            "Prophet_Baseline": forecast_res["future_prophet"],
            "XGBoost_Estimate": forecast_res["future_xgboost"]
        })
        st.dataframe(export_df, use_container_width=True, height=220)
        csv_data = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Forecast Projections (CSV)",
            data=csv_data,
            file_name=f"{selected_ticker}_forecast_{horizon_days}d.csv",
            mime="text/csv"
        )
