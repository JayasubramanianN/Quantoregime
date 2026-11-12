import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.analytics.simulation_engine import MonteCarloSimulationEngine
from app.components.ui_styles import get_institutional_chart_layout

@st.cache_resource
def get_simulation_engine():
    return MonteCarloSimulationEngine(seed=42)

def render_simulation_tab():
    st.markdown("### 🎲 Stochastic Monte Carlo & Macro Shock Simulator")
    st.caption("Vectorized Geometric Brownian Motion (10,000+ paths) with macro stress-testing and tail-risk VaR/CVaR metrics.")

    engine = get_simulation_engine()

    # Simulation Parameter Configuration
    c_base, c_shock = st.columns([1.3, 1.7])

    with c_base:
        st.markdown("##### ⚙️ Baseline Asset Dynamics")
        col_s0, col_days = st.columns(2)
        with col_s0:
            s0 = st.number_input("Spot Price (S0):", min_value=1.0, max_value=200000.0, value=5815.0, step=50.0, help="Initial spot market price for asset.")
        with col_days:
            sim_days = st.slider("Trading Days Horizon:", min_value=21, max_value=504, value=252, step=21, help="Simulation time step horizon in trading business days.")

        col_mu, col_sig = st.columns(2)
        with col_mu:
            mu_pct = st.number_input("Expected Drift (μ %):", min_value=-50.0, max_value=80.0, value=9.5, step=0.5, help="Annualized expected price return.") / 100.0
        with col_sig:
            sigma_pct = st.number_input("Annual Volatility (σ %):", min_value=1.0, max_value=150.0, value=17.5, step=0.5, help="Annualized volatility standard deviation.") / 100.0

        num_paths = st.select_slider("Simulation Paths (N):", options=[2500, 5000, 10000, 20000], value=10000, help="Number of vectorized stochastic Monte Carlo trajectories.")

    with c_shock:
        st.markdown("##### ⚡ Macro Shock & Stress Testing Overlay")
        scenario_preset = st.selectbox(
            "Institutional Stress Scenario:",
            [
                "Baseline Neutral (No Shock)",
                "Hawkish Rate Shock (+150 bps Tightening)",
                "Geopolitical Stagflation Shock (Energy Spike)",
                "Liquidity Squeeze / Credit Contraction (VIX 40+)",
                "Soft Landing & Accommodative Monetary Pivot",
                "Custom Dynamic Shock Override"
            ],
            help="Select institutional macroeconomic stress scenario to apply parameter overrides."
        )

        preset_params = {
            "Baseline Neutral (No Shock)": {"d_mu": 0.0, "d_sigma": 0.0, "jump": 0.0},
            "Hawkish Rate Shock (+150 bps Tightening)": {"d_mu": -0.04, "d_sigma": 0.08, "jump": -0.02},
            "Geopolitical Stagflation Shock (Energy Spike)": {"d_mu": -0.09, "d_sigma": 0.16, "jump": -0.06},
            "Liquidity Squeeze / Credit Contraction (VIX 40+)": {"d_mu": -0.03, "d_sigma": 0.22, "jump": -0.05},
            "Soft Landing & Accommodative Monetary Pivot": {"d_mu": 0.05, "d_sigma": -0.04, "jump": 0.02},
        }

        if scenario_preset in preset_params:
            p = preset_params[scenario_preset]
            shock_mu_val = p["d_mu"]
            shock_sigma_val = p["d_sigma"]
            jump_pct_val = p["jump"]
            st.info(f"Active Scenario: **{scenario_preset}** (Drift: `{shock_mu_val*100:+.1f}%`, Vol: `{shock_sigma_val*100:+.1f}%`, Instant Jump: `{jump_pct_val*100:+.1f}%`)")
        else:
            col_sm, col_ss, col_sj = st.columns(3)
            with col_sm:
                shock_mu_val = st.slider("Drift Shock (Δμ %):", min_value=-25.0, max_value=25.0, value=-5.0, step=0.5) / 100.0
            with col_ss:
                shock_sigma_val = st.slider("Vol Shock (Δσ %):", min_value=-10.0, max_value=40.0, value=12.0, step=1.0) / 100.0
            with col_sj:
                jump_pct_val = st.slider("Immediate Jump (ΔS %):", min_value=-25.0, max_value=15.0, value=-3.0, step=0.5) / 100.0

    # Run Vectorized Monte Carlo Simulation
    with st.spinner(f"Simulating {num_paths:,} vectorized stochastic paths across {sim_days} days..."):
        time_steps, paths = engine.simulate_gbm_paths(
            s0=s0,
            mu=mu_pct,
            sigma=sigma_pct,
            days=sim_days,
            num_paths=num_paths,
            shock_mu=shock_mu_val,
            shock_sigma=shock_sigma_val,
            immediate_shock_pct=jump_pct_val
        )
        risk = engine.calculate_risk_metrics(s0, paths)

    # Risk KPI Cards with Guaranteed Consistent Negative Loss Formatting
    st.markdown("---")
    st.markdown("#### 🛡️ Value at Risk (VaR) & Tail Risk Scorecard")

    var_95_formatted = f"-${abs(risk['var_95_dollar']):,.2f}"
    var_95_pct_formatted = f"-{abs(risk['var_95_pct']):.1f}%"
    cvar_95_formatted = f"-${abs(risk['cvar_95_dollar']):,.2f}"
    cvar_95_pct_formatted = f"-{abs(risk['cvar_95_pct']):.1f}%"
    var_99_formatted = f"-${abs(risk['var_99_dollar']):,.2f}"
    var_99_pct_formatted = f"-{abs(risk['var_99_pct']):.1f}%"

    col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
    with col_r1:
        st.markdown(f"""
        <div class="macro-metric-card">
            <div class="metric-label">
                <span>VaR (95% 1-Year)</span>
                <span class="tooltip-icon" title="Value at Risk: Maximum expected loss at 95% confidence interval over the horizon">ℹ️</span>
            </div>
            <div class="metric-value" style="color: #fb7185;">{var_95_formatted}</div>
            <div class="metric-subtext">Capital Loss: {var_95_pct_formatted}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_r2:
        st.markdown(f"""
        <div class="macro-metric-card">
            <div class="metric-label">
                <span>CVaR / Tail Risk (95%)</span>
                <span class="tooltip-icon" title="Conditional VaR (Expected Shortfall): Mean loss magnitude given that loss breaches the 95% VaR threshold">ℹ️</span>
            </div>
            <div class="metric-value" style="color: #f43f5e;">{cvar_95_formatted}</div>
            <div class="metric-subtext">Expected Shortfall: {cvar_95_pct_formatted}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_r3:
        st.markdown(f"""
        <div class="macro-metric-card">
            <div class="metric-label">
                <span>VaR (99% Extreme)</span>
                <span class="tooltip-icon" title="Extreme tail loss cutoff at the 99% confidence boundary">ℹ️</span>
            </div>
            <div class="metric-value" style="color: #e11d48;">{var_99_formatted}</div>
            <div class="metric-subtext">Capital Loss: {var_99_pct_formatted}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_r4:
        st.markdown(f"""
        <div class="macro-metric-card">
            <div class="metric-label">
                <span>P(Capital Loss)</span>
                <span class="tooltip-icon" title="Probability of final terminal price falling below the initial spot price S0">ℹ️</span>
            </div>
            <div class="metric-value" style="color: #fbbf24;">{risk['prob_loss_pct']:.1f}%</div>
            <div class="metric-subtext">Prob ending &lt; ${s0:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_r5:
        st.markdown(f"""
        <div class="macro-metric-card">
            <div class="metric-label">
                <span>P(Severe Loss &gt; 20%)</span>
                <span class="tooltip-icon" title="Probability of terminal drawdown exceeding 20% of capital">ℹ️</span>
            </div>
            <div class="metric-value" style="color: #fb7185;">{risk['prob_loss_20_pct']:.1f}%</div>
            <div class="metric-subtext">Severe Drawdown Likelihood</div>
        </div>
        """, unsafe_allow_html=True)

    # Visualizations: Cone Trajectory & Terminal Distribution
    col_chart1, col_chart2 = st.columns([1.6, 1.4])

    with col_chart1:
        st.markdown("##### 🌪️ Vectorized Monte Carlo Path Percentile Cone")
        cones = risk["cone_trajectories"]
        days_arr = list(range(sim_days + 1))

        fig_cone = go.Figure()

        # 1% to 99%
        fig_cone.add_trace(go.Scatter(
            x=days_arr + days_arr[::-1],
            y=cones["p99"] + cones["p1"][::-1],
            fill='toself',
            fillcolor='rgba(56, 189, 248, 0.08)',
            line=dict(color='rgba(255,255,255,0)'),
            name="1% - 99% Tail Range",
            hoverinfo="skip"
        ))
        # 5% to 95%
        fig_cone.add_trace(go.Scatter(
            x=days_arr + days_arr[::-1],
            y=cones["p95"] + cones["p5"][::-1],
            fill='toself',
            fillcolor='rgba(56, 189, 248, 0.16)',
            line=dict(color='rgba(255,255,255,0)'),
            name="5% - 95% Confidence Band",
            hoverinfo="skip"
        ))
        # 25% to 75%
        fig_cone.add_trace(go.Scatter(
            x=days_arr + days_arr[::-1],
            y=cones["p75"] + cones["p25"][::-1],
            fill='toself',
            fillcolor='rgba(56, 189, 248, 0.28)',
            line=dict(color='rgba(255,255,255,0)'),
            name="25% - 75% Interquartile",
            hoverinfo="skip"
        ))
        # Median (50%)
        fig_cone.add_trace(go.Scatter(
            x=days_arr,
            y=cones["p50"],
            line=dict(color="#38bdf8", width=2.6),
            name="Median Trajectory (p50)",
            hovertemplate="<b>Day %{x}</b><br>Median: $%{y:,.2f}<extra></extra>"
        ))
        # Reference Price
        fig_cone.add_hline(y=s0, line_dash="dash", line_color="#94a3b8", annotation_text=f"S0 = ${s0:,.0f}")

        cone_layout = get_institutional_chart_layout(title="Stochastic GBM Path Distribution Cone", height=390)
        cone_layout["xaxis"]["title"] = "Trading Days Forward"
        cone_layout["yaxis"]["title"] = "Simulated Price (USD)"
        fig_cone.update_layout(**cone_layout)
        st.plotly_chart(fig_cone, use_container_width=True)

    with col_chart2:
        st.markdown("##### 📊 Terminal Price Distribution & Tail Cutoffs")
        final_samples = risk["final_prices_sample"]

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=final_samples,
            nbinsx=40,
            marker=dict(color="#818cf8", line=dict(color="#1e1b4b", width=1)),
            name="Terminal Outcomes",
            hovertemplate="Price Bucket: $%{x:,.0f}<br>Count: %{y}<extra></extra>"
        ))

        # VaR 95%
        p5_val = risk["p5_terminal_price"]
        fig_hist.add_vline(
            x=p5_val,
            line_dash="dash",
            line_color="#fb7185",
            annotation_text=f"VaR 95%: ${p5_val:,.0f}",
            annotation_position="top left"
        )

        # VaR 99%
        p1_val = risk["p1_terminal_price"]
        fig_hist.add_vline(
            x=p1_val,
            line_dash="dot",
            line_color="#f43f5e",
            annotation_text=f"VaR 99%: ${p1_val:,.0f}",
            annotation_position="bottom left"
        )

        # Baseline S0
        fig_hist.add_vline(
            x=s0,
            line_color="#38bdf8",
            annotation_text=f"Spot: ${s0:,.0f}",
            annotation_position="top right"
        )

        hist_layout = get_institutional_chart_layout(title="Terminal Horizon Price Density", height=390, show_legend=False)
        hist_layout["xaxis"]["title"] = "Terminal Price (USD)"
        hist_layout["yaxis"]["title"] = "Density"
        fig_hist.update_layout(**hist_layout)
        st.plotly_chart(fig_hist, use_container_width=True)

    # Multi-Scenario Stress Testing Matrix & Grouped Risk vs Return Bar Chart
    st.markdown("---")
    st.markdown("#### 📋 Institutional Macro Stress Testing Matrix (All Regimes)")

    with st.spinner("Computing full stress testing scenario matrix..."):
        matrix_results = engine.run_stress_test_matrix(s0=s0, mu=mu_pct, sigma=sigma_pct, days=sim_days, num_paths=5000)

        # Cross-Scenario Grouped Bar Chart (Clean non-overlapping side-by-side bars for Expected ROI and 95% VaR Capital Downside)
        scen_names = list(matrix_results.keys())
        scen_rois = [matrix_results[k]["expected_roi_pct"] for k in scen_names]
        scen_vars_neg = [-abs(matrix_results[k]["var_95_pct"]) for k in scen_names]

        fig_compare = go.Figure()
        # Bar 1: Expected ROI (%)
        fig_compare.add_trace(go.Bar(
            x=scen_names,
            y=scen_rois,
            name="Expected ROI (%)",
            marker=dict(color=["#10b981" if r >= 0 else "#fb7185" for r in scen_rois]),
            hovertemplate="<b>%{x}</b><br>Expected ROI: %{y:+.1f}%<extra></extra>"
        ))
        # Bar 2: 95% VaR Capital Downside (%) (Cleanly plotted as downside negative bar)
        fig_compare.add_trace(go.Bar(
            x=scen_names,
            y=scen_vars_neg,
            name="VaR 95% Max Capital Downside (%)",
            marker=dict(color="#f43f5e"),
            hovertemplate="<b>%{x}</b><br>95% VaR Downside: %{y:.1f}%<extra></extra>"
        ))

        fig_compare.add_hline(y=0.0, line_dash="solid", line_color="#64748b", line_width=1)

        comp_layout = get_institutional_chart_layout(title="Scenario Stress Matrix: Expected ROI vs 95% VaR Downside Risk", height=310)
        comp_layout["barmode"] = "group"
        comp_layout["yaxis"]["title"] = "Percentage (% of Capital)"
        fig_compare.update_layout(**comp_layout)
        st.plotly_chart(fig_compare, use_container_width=True)

        # Matrix Data Table with Guaranteed Format f"-${abs(val):,.2f}"
        matrix_rows = []
        for s_name, m_data in matrix_results.items():
            var_dollar_clean = f"-${abs(m_data['var_95_dollar']):,.2f}"
            var_pct_clean = f"-{abs(m_data['var_95_pct']):.1f}%"
            cvar_dollar_clean = f"-${abs(m_data['cvar_95_dollar']):,.2f}"

            matrix_rows.append({
                "Scenario": s_name,
                "Eff. Drift (μ)": f"{m_data['effective_mu']}%",
                "Eff. Vol (σ)": f"{m_data['effective_sigma']}%",
                "Expected ROI": f"{m_data['expected_roi_pct']:+.1f}%",
                "VaR 95% ($)": var_dollar_clean,
                "VaR 95% (%)": var_pct_clean,
                "CVaR 95% ($)": cvar_dollar_clean,
                "P(Loss)": f"{m_data['prob_loss_pct']:.1f}%",
                "Description": m_data["description"]
            })

        matrix_df = pd.DataFrame(matrix_rows)
        st.dataframe(matrix_df, use_container_width=True, hide_index=True)

        # One-Click PDF Tearsheet Export for Active Simulation
        st.markdown("---")
        c_exp1, c_exp2 = st.columns([1.8, 1.2])
        with c_exp1:
            st.markdown("##### 📄 Export Executive Risk Tearsheet (PDF)")
            st.caption("Generate a publication-grade 2-page executive tearsheet with active spot price, simulation horizon, and stress testing matrix.")
        with c_exp2:
            export_key = f"pdf_{s0}_{sim_days}_{mu_pct}_{sigma_pct}"
            if st.button("📊 Generate Executive Tearsheet", key="btn_gen_pdf_tab", use_container_width=True, type="primary"):
                with st.spinner("Compiling institutional PDF tearsheet..."):
                    from src.reporting.tearsheet_generator import generate_executive_briefing_pdf
                    pdf_bytes = generate_executive_briefing_pdf(
                        spot_price=float(s0),
                        expected_drift=float(mu_pct),
                        annual_volatility=float(sigma_pct),
                        target_horizon_days=int(sim_days),
                        asset_label="S&P 500 (^GSPC)"
                    )
                    st.session_state[export_key] = pdf_bytes
                    st.success("Tearsheet ready!")

            if export_key in st.session_state:
                st.download_button(
                    label="📥 Download Executive Tearsheet (PDF)",
                    data=st.session_state[export_key],
                    file_name=f"QuantMacro_Risk_Tearsheet_{int(s0)}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

