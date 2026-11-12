import pytest
import numpy as np
import pandas as pd
from src.analytics.forecasting_engine import MacroForecastingEngine

@pytest.fixture
def forecast_engine():
    return MacroForecastingEngine(prophet_weight=0.40, xgboost_weight=0.60)

def test_ensemble_forecasting_pipeline(forecast_engine):
    # Create trending macro time-series
    dates = pd.date_range("2021-01-01", periods=400, freq="B")
    trend = np.linspace(100, 160, 400)
    noise = np.random.default_rng(42).normal(0, 1.5, 400)
    series = pd.Series(trend + noise, index=dates, name="BenchmarkAsset")

    horizon = 30
    res = forecast_engine.run_ensemble_forecast(series, horizon_days=horizon)

    assert res["horizon_days"] == horizon
    assert len(res["future_ensemble"]) == horizon
    assert len(res["future_lower"]) == horizon
    assert len(res["future_upper"]) == horizon

    # Validate that current_price matches the last historical close within 2 decimals
    assert np.isclose(res["current_price"], series.iloc[-1], atol=0.01)

    # Validate confidence intervals: lower <= mean <= upper
    lower_arr = np.array(res["future_lower"])
    upper_arr = np.array(res["future_upper"])
    ensemble_arr = np.array(res["future_ensemble"])
    assert np.all(lower_arr <= upper_arr)
    assert np.all(lower_arr <= ensemble_arr + 1e-5)
    assert np.all(ensemble_arr <= upper_arr + 1e-5)

    # Verify dates formatting
    assert len(res["future_dates"]) == horizon
    assert pd.to_datetime(res["future_dates"][0]) > series.index[-1]
