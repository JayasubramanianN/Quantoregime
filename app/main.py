import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.components.ui_styles import apply_institutional_theme, render_top_header
from app.components.macro_tab import render_macro_tab
from app.components.hypothesis_tab import render_hypothesis_tab
from app.components.forecasting_tab import render_forecasting_tab
from app.components.simulation_tab import render_simulation_tab
from config.settings import settings

def main():
    st.set_page_config(
        page_title="QuantMacro AI | Institutional Macroeconomic Intelligence",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply CSS styling
    apply_institutional_theme()

    # Render Hero Top Header & Ticker Tape
    render_top_header()

    # Sidebar Navigation & System Monitor
    with st.sidebar:
        st.markdown("### 🎛️ Terminal Controls")
        st.caption("QuantMacro AI Core Architecture")

        st.markdown("---")
        st.markdown("#### 📄 Executive Briefing Export")
        st.caption("One-click 2-page institutional PDF tearsheet.")
        if st.button("📊 Compile Executive Tearsheet", key="sidebar_compile_btn", use_container_width=True, type="primary"):
            with st.spinner("Generating institutional PDF tearsheet..."):
                from src.reporting.tearsheet_generator import generate_executive_briefing_pdf
                pdf_data = generate_executive_briefing_pdf(
                    spot_price=5815.0,
                    expected_drift=0.095,
                    annual_volatility=0.175,
                    target_horizon_days=252,
                    asset_label="S&P 500 (^GSPC)"
                )
                st.session_state["sidebar_executive_pdf"] = pdf_data
                st.success("Tearsheet compiled!")

        if "sidebar_executive_pdf" in st.session_state:
            from datetime import datetime
            st.download_button(
                label="📥 Download Tearsheet (PDF)",
                data=st.session_state["sidebar_executive_pdf"],
                file_name=f"QuantMacro_Executive_Tearsheet_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="sidebar_download_btn"
            )

        st.markdown("---")
        st.markdown("#### ⚡ Active Environment")
        st.markdown(f"""
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; line-height: 1.8; color: #cbd5e1;">
            <b>Status:</b> <span style="color:#10b981;">Online (FastAPI + Streamlit)</span><br>
            <b>Environment:</b> <span style="color:#38bdf8;">{settings.ENVIRONMENT.upper()}</span><br>
            <b>Vector Store:</b> <span style="color:#c084fc;">FAISS {settings.VECTOR_DIMENSION}d (Cosine)</span><br>
            <b>Simulation:</b> <span style="color:#fbbf24;">Vectorized GBM (Numpy)</span><br>
            <b>Seed:</b> <span style="color:#94a3b8;">{settings.RANDOM_SEED} (Deterministic)</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 🔗 API Endpoints")
        st.markdown("""
        - [Swagger / OpenAPI Docs](http://localhost:8000/docs)
        - [Health Check (`/health`)](http://localhost:8000/health)
        - [Monte Carlo API (`/api/simulation/monte-carlo`)](http://localhost:8000/docs#/default/run_monte_carlo_api_simulation_monte_carlo_post)
        - [PDF Tearsheet API (`/api/reports/executive-tearsheet`)](http://localhost:8000/api/reports/executive-tearsheet)
        """)

        st.markdown("---")
        if st.button("🧹 Clear Analytical Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared! Reloading fresh market feeds.")

        st.markdown("---")
        st.caption("QuantMacro AI v1.0.0 — Institutional Edition")

    # Quick Top Action Bar right below Ticker Tape
    c_bar1, c_bar2 = st.columns([2.6, 1.4])
    with c_bar1:
        st.markdown("""
        <div style="padding: 6px 0 10px 2px;">
            <span style="font-size: 0.82rem; font-weight: 700; color: #38bdf8; font-family: 'JetBrains Mono', monospace;">
                ⚡ INSTITUTIONAL INTELLIGENCE TERMINAL
            </span>
            <span style="font-size: 0.80rem; color: #94a3b8; margin-left: 8px;">
                • Live Macro Regime • Econometric Causality • 10,000-Path Monte Carlo VaR
            </span>
        </div>
        """, unsafe_allow_html=True)
    with c_bar2:
        if st.button("📄 Export Executive Tearsheet (PDF)", key="top_bar_export_btn", use_container_width=True, type="primary"):
            with st.spinner("Compiling institutional 2-page PDF tearsheet..."):
                from src.reporting.tearsheet_generator import generate_executive_briefing_pdf
                pdf_top = generate_executive_briefing_pdf(
                    spot_price=5815.0,
                    expected_drift=0.095,
                    annual_volatility=0.175,
                    target_horizon_days=252,
                    asset_label="S&P 500 (^GSPC)"
                )
                st.session_state["top_bar_pdf"] = pdf_top
                st.success("Tearsheet compiled!")

        if "top_bar_pdf" in st.session_state:
            from datetime import datetime
            st.download_button(
                label="📥 Download Tearsheet (PDF)",
                data=st.session_state["top_bar_pdf"],
                file_name=f"QuantMacro_Executive_Tearsheet_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="top_bar_download_btn"
            )

    # 4 Executive Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏛️ Macro Intelligence & Qualitative RAG",
        "🧪 Hypothesis Lab",
        "🔮 Trend Forecasting",
        "🎲 Risk & Monte Carlo Simulator"
    ])

    with tab1:
        render_macro_tab()

    with tab2:
        render_hypothesis_tab()

    with tab3:
        render_forecasting_tab()

    with tab4:
        render_simulation_tab()

if __name__ == "__main__":
    main()
