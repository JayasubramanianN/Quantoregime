from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from config.settings import settings
from src.analytics.hypothesis_engine import HypothesisEngine
from src.analytics.forecasting_engine import MacroForecastingEngine
from src.analytics.simulation_engine import MonteCarloSimulationEngine
from src.llm_rag.macro_summarizer import MacroRAGSummarizer
from src.data_ingestion.fred_loader import FredLoader
from src.data_ingestion.yahoo_loader import YahooFinanceLoader
from src.reporting.tearsheet_generator import generate_executive_briefing_pdf

app = FastAPI(
    title="QuantMacro AI Institutional API",
    description="REST API for Macroeconomic Intelligence, Hypothesis Testing, Forecasting & Stochastic Risk Simulation",
    version=settings.APP_VERSION
)

hypothesis_engine = HypothesisEngine()
simulation_engine = MonteCarloSimulationEngine()
forecasting_engine = MacroForecastingEngine()
rag_summarizer = MacroRAGSummarizer()
fred_loader = FredLoader()
yahoo_loader = YahooFinanceLoader()

class SimulationRequest(BaseModel):
    s0: float = Field(default=5000.0, description="Initial spot asset price")
    mu: float = Field(default=0.09, description="Expected annual drift")
    sigma: float = Field(default=0.18, description="Annual volatility")
    days: int = Field(default=252, description="Trading days horizon")
    num_paths: int = Field(default=10000, description="Number of Monte Carlo paths")
    shock_mu: float = Field(default=0.0, description="Exogenous drift shock override")
    shock_sigma: float = Field(default=0.0, description="Exogenous volatility shock override")
    immediate_shock_pct: float = Field(default=0.0, description="Immediate jump shock pct")

class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="Macro query or analytical question")
    top_k: int = Field(default=3, description="Number of context passages")
    source_filter: Optional[str] = Field(default=None, description="Optional document source filter")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }

@app.post("/api/simulation/monte-carlo")
def run_monte_carlo(req: SimulationRequest):
    _, paths = simulation_engine.simulate_gbm_paths(
        s0=req.s0,
        mu=req.mu,
        sigma=req.sigma,
        days=req.days,
        num_paths=req.num_paths,
        shock_mu=req.shock_mu,
        shock_sigma=req.shock_sigma,
        immediate_shock_pct=req.immediate_shock_pct
    )
    metrics = simulation_engine.calculate_risk_metrics(req.s0, paths)
    # Remove large trajectory arrays from lightweight API response
    metrics_summary = {k: v for k, v in metrics.items() if k not in ["cone_trajectories", "final_prices_sample"]}
    return {"status": "success", "parameters": req.model_dump(), "risk_metrics": metrics_summary}

@app.post("/api/simulation/stress-test")
def run_stress_test(s0: float = 5000.0, mu: float = 0.09, sigma: float = 0.18, days: int = 252):
    matrix = simulation_engine.run_stress_test_matrix(s0=s0, mu=mu, sigma=sigma, days=days, num_paths=5000)
    for m in matrix.values():
        m.pop("cone_trajectories", None)
        m.pop("final_prices_sample", None)
    return {"status": "success", "scenarios": matrix}

@app.post("/api/rag/query")
def rag_query(req: RAGQueryRequest):
    result = rag_summarizer.query_and_summarize(
        query=req.query,
        top_k=req.top_k,
        source_filter=req.source_filter
    )
    return result

@app.get("/api/macro/indicators")
def get_macro_indicators():
    df = fred_loader.get_macro_dataset(freq="B")
    latest = df.iloc[-1].to_dict()
    return {"status": "success", "latest_readings": latest}

@app.get("/api/reports/executive-tearsheet", summary="Generate Executive Briefing & Risk Tearsheet PDF")
def get_executive_tearsheet(
    spot_price: float = Query(default=5815.0, description="Spot asset price"),
    drift: float = Query(default=0.095, description="Expected annual drift"),
    volatility: float = Query(default=0.175, description="Annual volatility"),
    horizon_days: int = Query(default=252, description="Trading days horizon"),
    asset_label: str = Query(default="S&P 500 (^GSPC)", description="Target asset label")
):
    """
    Renders and streams a high-resolution 2-page institutional PDF tearsheet
    synthesizing macroeconomic regime data, econometric causality,
    multi-horizon forecasts, and 10,000-path stochastic Monte Carlo stress testing.
    """
    pdf_bytes = generate_executive_briefing_pdf(
        spot_price=spot_price,
        expected_drift=drift,
        annual_volatility=volatility,
        target_horizon_days=horizon_days,
        asset_label=asset_label
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=QuantMacro_Executive_Tearsheet.pdf"
        }
    )

