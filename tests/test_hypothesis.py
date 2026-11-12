import pytest
import numpy as np
import pandas as pd
from src.analytics.hypothesis_engine import HypothesisEngine

@pytest.fixture
def hypothesis_engine():
    return HypothesisEngine(significance_level=0.05)

def test_adf_stationary_series(hypothesis_engine):
    # White noise is stationary (d=0) in raw levels
    rng = np.random.default_rng(42)
    s = pd.Series(rng.normal(0, 1, 300), name="WhiteNoise")
    res = hypothesis_engine.adf_stationarity_test(s)
    
    assert res["raw_is_stationary"] is True
    assert res["is_stationary"] is True
    assert res["differencing_order"] == 0
    assert res["raw_p_value"] < 0.05
    assert "STATIONARY (Raw Levels)" in res["status_label"]

def test_adf_random_walk_differencing(hypothesis_engine):
    # Random walk with unit root: raw levels are non-stationary, d=1 differencing achieves stationarity
    rng = np.random.default_rng(42)
    rw = pd.Series(np.cumsum(rng.normal(0, 1, 300)), name="RandomWalk")
    res = hypothesis_engine.adf_stationarity_test(rw, max_diff=1)
    
    assert res["raw_is_stationary"] is False
    assert res["differenced_is_stationary"] is True
    assert res["differencing_order"] == 1
    assert "STATIONARY (Differenced d=1)" in res["status_label"]

def test_granger_causality_with_true_lagged_driver(hypothesis_engine):
    # Construct X and Y where Y = 0.8 * X_{t-1} + noise
    rng = np.random.default_rng(42)
    n = 250
    x_innov = rng.normal(0, 1, n)
    x = pd.Series(x_innov, name="OilPrices")
    
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.75 * x.iloc[t - 1] + rng.normal(0, 0.2)
    y_series = pd.Series(y, name="Inflation")

    res = hypothesis_engine.granger_causality_test(x, y_series, max_lags=3, ensure_stationarity=False)
    assert res["is_significant"] is True
    assert res["best_lag"] == 1
    assert f"at Lag {res['best_lag']}" in res["interpretation"]
    assert res["min_p_value"] < 0.05

def test_parametric_t_test(hypothesis_engine):
    rng = np.random.default_rng(42)
    sample_a = pd.Series(rng.normal(10.0, 1.0, 100))
    sample_b = pd.Series(rng.normal(12.0, 1.0, 100))
    res = hypothesis_engine.parametric_t_test(sample_a, sample_b)
    
    assert res["is_significant"]
    assert res["p_value"] < 0.01
    assert res["mean_difference"] < 0

def test_correlation_analysis(hypothesis_engine):
    rng = np.random.default_rng(42)
    x = pd.Series(rng.normal(0, 1, 150))
    y = x * 2.0 + rng.normal(0, 0.5, 150)
    res = hypothesis_engine.correlation_analysis(x, y)
    
    assert res["is_significant"]
    assert res["pearson_r"] > 0.85
    assert res["spearman_rho"] > 0.80
