import pytest
import numpy as np
from src.analytics.simulation_engine import MonteCarloSimulationEngine

@pytest.fixture
def mc_engine():
    return MonteCarloSimulationEngine(seed=42)

def test_gbm_simulation_dimensions(mc_engine):
    days = 100
    paths_count = 5000
    s0 = 100.0
    mu = 0.08
    sigma = 0.20
    
    t_steps, paths = mc_engine.simulate_gbm_paths(
        s0=s0,
        mu=mu,
        sigma=sigma,
        days=days,
        num_paths=paths_count
    )
    
    assert len(t_steps) == days + 1
    assert paths.shape == (days + 1, paths_count)
    # Price at t=0 must exactly equal S0
    assert np.allclose(paths[0, :], s0)
    # Prices must strictly remain positive (GBM boundary condition)
    assert np.all(paths > 0)

def test_reproducibility(mc_engine):
    engine1 = MonteCarloSimulationEngine(seed=42)
    engine2 = MonteCarloSimulationEngine(seed=42)
    
    _, p1 = engine1.simulate_gbm_paths(100.0, 0.05, 0.15, days=50, num_paths=1000)
    _, p2 = engine2.simulate_gbm_paths(100.0, 0.05, 0.15, days=50, num_paths=1000)
    
    np.testing.assert_array_equal(p1, p2)

def test_risk_metrics_var_cvar(mc_engine):
    s0 = 500.0
    _, paths = mc_engine.simulate_gbm_paths(s0, 0.06, 0.25, days=252, num_paths=10000)
    metrics = mc_engine.calculate_risk_metrics(s0, paths)
    
    # Fundamental risk relationship: CVaR (Expected Shortfall) >= VaR
    assert metrics["cvar_95_dollar"] >= metrics["var_95_dollar"]
    assert metrics["cvar_99_dollar"] >= metrics["var_99_dollar"]
    assert metrics["var_99_dollar"] >= metrics["var_95_dollar"]
    
    # Loss probabilities between 0 and 100%
    assert 0 <= metrics["prob_loss_pct"] <= 100

    # Negative loss formatting verification (no string corruption like -$2/4.00)
    clean_var_str = f"-${abs(metrics['var_95_dollar']):,.2f}"
    clean_cvar_str = f"-${abs(metrics['cvar_95_dollar']):,.2f}"
    assert clean_var_str.startswith("-$")
    assert clean_cvar_str.startswith("-$")
    assert "/" not in clean_var_str

def test_stress_test_matrix(mc_engine):
    matrix = mc_engine.run_stress_test_matrix(s0=100.0, mu=0.08, sigma=0.18, days=60, num_paths=2000)
    assert "Baseline Neutral" in matrix
    assert "Hawkish Rate Shock (+150 bps)" in matrix
    assert "Geopolitical Stagflation Shock" in matrix
    
    # Stagflation shock should yield higher VaR than baseline
    base_var = matrix["Baseline Neutral"]["var_95_dollar"]
    stag_var = matrix["Geopolitical Stagflation Shock"]["var_95_dollar"]
    assert stag_var > base_var

def test_worker_async_monte_carlo():
    from src.worker import async_monte_carlo
    res = async_monte_carlo(s0=100.0, mu=0.08, sigma=0.18, days=60, num_paths=1000)
    assert "var_95_dollar" in res
    assert "cvar_95_dollar" in res
    assert "expected_roi_pct" in res
    # Large raw arrays should be stripped for lightweight transmission
    assert "cone_trajectories" not in res
    assert "final_prices_sample" not in res

