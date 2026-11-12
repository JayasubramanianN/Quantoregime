import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from config.settings import settings
from config.logging_config import logger

class MonteCarloSimulationEngine:
    """
    Vectorized Stochastic Monte Carlo Simulation & Risk Engine.
    Simulates multi-thousand Geometric Brownian Motion paths with stress testing shock overlays,
    computing parametric & empirical VaR, CVaR (Expected Shortfall), and drawdown probabilities.
    """

    def __init__(self, seed: int = settings.RANDOM_SEED):
        self.seed = seed

    def simulate_gbm_paths(
        self,
        s0: float,
        mu: float,
        sigma: float,
        days: int = 252,
        num_paths: int = 10000,
        shock_mu: float = 0.0,
        shock_sigma: float = 0.0,
        immediate_shock_pct: float = 0.0,
        dt: float = 1.0 / 252.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ultra-fast vectorized GBM simulator:
        S_t = S_{t-1} * exp( (mu_eff - 0.5 * sigma_eff^2)*dt + sigma_eff * sqrt(dt) * Z_t )
        
        Returns:
            time_steps: array of day indices (0 to days)
            paths: 2D array of shape (days + 1, num_paths)
        """
        # Apply random seed for exact reproducibility
        rng = np.random.default_rng(self.seed)

        # Incorporate macro stress shocks
        mu_eff = mu + shock_mu
        sigma_eff = max(0.001, sigma + shock_sigma)

        # Apply immediate jump if specified (e.g. -5% immediate repricing)
        s_initial = s0 * (1.0 + immediate_shock_pct)

        # Vectorized normal random variables Z ~ N(0, 1) of shape (days, num_paths)
        z = rng.standard_normal((days, num_paths))

        # Vectorized incremental log returns
        drift_term = (mu_eff - 0.5 * (sigma_eff ** 2)) * dt
        diffusion_term = sigma_eff * np.sqrt(dt) * z
        log_increments = drift_term + diffusion_term

        # Cumulative log returns
        cum_log_returns = np.zeros((days + 1, num_paths))
        cum_log_returns[1:, :] = np.cumsum(log_increments, axis=0)

        # Final asset paths S_t
        paths = s_initial * np.exp(cum_log_returns)
        time_steps = np.arange(days + 1)

        return time_steps, paths

    def calculate_risk_metrics(
        self,
        s0: float,
        paths: np.ndarray,
        confidence_levels: Tuple[float, float] = (0.95, 0.99)
    ) -> Dict[str, Any]:
        """
        Calculates institutional risk metrics from simulated paths:
        - Value at Risk (VaR 95% & 99%)
        - Conditional VaR / Expected Shortfall (CVaR 95% & 99%)
        - Probability of capital loss P(S_final < S0)
        - Mean, median, max drawdown, skewness, kurtosis
        """
        final_prices = paths[-1, :]
        losses = s0 - final_prices  # Positive loss represents capital destruction
        loss_pcts = (s0 - final_prices) / s0

        # Percentile cutoffs for losses
        # VaR_95 corresponds to the 95th percentile of loss (or 5th percentile of price)
        var_95_dollar = float(np.percentile(losses, 95))
        var_99_dollar = float(np.percentile(losses, 99))
        var_95_pct = float(np.percentile(loss_pcts, 95))
        var_99_pct = float(np.percentile(loss_pcts, 99))

        # Price percentiles
        p5_price = float(np.percentile(final_prices, 5))
        p1_price = float(np.percentile(final_prices, 1))

        # CVaR (Expected Shortfall): E[Loss | Loss >= VaR]
        tail_95 = losses[losses >= var_95_dollar]
        cvar_95_dollar = float(np.mean(tail_95)) if len(tail_95) > 0 else var_95_dollar
        cvar_95_pct = cvar_95_dollar / s0

        tail_99 = losses[losses >= var_99_dollar]
        cvar_99_dollar = float(np.mean(tail_99)) if len(tail_99) > 0 else var_99_dollar
        cvar_99_pct = cvar_99_dollar / s0

        # Drawdown probability
        prob_drawdown = float(np.mean(final_prices < s0))
        prob_severe_loss_20 = float(np.mean(final_prices < (s0 * 0.80)))

        # Trajectory statistics
        expected_final = float(np.mean(final_prices))
        median_final = float(np.median(final_prices))
        std_final = float(np.std(final_prices))
        expected_roi = float((expected_final - s0) / s0)

        # Percentile trajectories across the entire time horizon for cone plots
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        cone_trajectories = {
            f"p{p}": np.percentile(paths, p, axis=1).tolist()
            for p in percentiles
        }

        return {
            "s0": round(s0, 2),
            "expected_final_price": round(expected_final, 2),
            "median_final_price": round(median_final, 2),
            "std_final_price": round(std_final, 2),
            "expected_roi_pct": round(expected_roi * 100, 2),
            "var_95_dollar": round(var_95_dollar, 2),
            "var_95_pct": round(var_95_pct * 100, 2),
            "var_99_dollar": round(var_99_dollar, 2),
            "var_99_pct": round(var_99_pct * 100, 2),
            "cvar_95_dollar": round(cvar_95_dollar, 2),
            "cvar_95_pct": round(cvar_95_pct * 100, 2),
            "cvar_99_dollar": round(cvar_99_dollar, 2),
            "cvar_99_pct": round(cvar_99_pct * 100, 2),
            "prob_loss_pct": round(prob_drawdown * 100, 2),
            "prob_loss_20_pct": round(prob_severe_loss_20 * 100, 2),
            "p5_terminal_price": round(p5_price, 2),
            "p1_terminal_price": round(p1_price, 2),
            "cone_trajectories": cone_trajectories,
            "final_prices_sample": final_prices[::20].tolist() # 500 points for UI histogram
        }

    def run_stress_test_matrix(
        self,
        s0: float,
        mu: float,
        sigma: float,
        days: int = 252,
        num_paths: int = 10000
    ) -> Dict[str, Dict[str, Any]]:
        """
        Runs standardized institutional macro stress testing scenarios:
        1. Baseline: Standard drift and volatility.
        2. Hawkish Fed Rate Shock (+150 bps rate hike drag): -4% drift, +8% volatility.
        3. Geopolitical Oil Crisis / Stagflation: -8% drift, +18% volatility, -6% instant jump.
        4. Volatility Spike (VIX > 40 style liquidity squeeze): -2% drift, +25% volatility.
        5. Dovish Easing & Economic Soft Landing: +5% drift, -3% volatility.
        """
        scenarios = {
            "Baseline Neutral": {
                "shock_mu": 0.0,
                "shock_sigma": 0.0,
                "immediate_shock_pct": 0.0,
                "description": "Baseline macroeconomic regime without exogenous shocks."
            },
            "Hawkish Rate Shock (+150 bps)": {
                "shock_mu": -0.04,
                "shock_sigma": 0.08,
                "immediate_shock_pct": -0.02,
                "description": "Aggressive monetary tightening increases discount rates and equity volatility."
            },
            "Geopolitical Stagflation Shock": {
                "shock_mu": -0.09,
                "shock_sigma": 0.16,
                "immediate_shock_pct": -0.06,
                "description": "Energy supply disruption induces cost-push inflation and compression of margins."
            },
            "Liquidity Squeeze (VIX 40+)": {
                "shock_mu": -0.03,
                "shock_sigma": 0.22,
                "immediate_shock_pct": -0.05,
                "description": "Sharp credit spread widening and sudden wholesale funding drawdown."
            },
            "Soft Landing & Monetary Easing": {
                "shock_mu": 0.05,
                "shock_sigma": -0.04,
                "immediate_shock_pct": 0.02,
                "description": "Disinflation enables central bank rate cuts while corporate earnings expand."
            }
        }

        results = {}
        for name, params in scenarios.items():
            _, paths = self.simulate_gbm_paths(
                s0=s0,
                mu=mu,
                sigma=sigma,
                days=days,
                num_paths=num_paths,
                shock_mu=params["shock_mu"],
                shock_sigma=params["shock_sigma"],
                immediate_shock_pct=params["immediate_shock_pct"]
            )
            metrics = self.calculate_risk_metrics(s0, paths)
            metrics["description"] = params["description"]
            metrics["effective_mu"] = round((mu + params["shock_mu"]) * 100, 2)
            metrics["effective_sigma"] = round(max(0.01, sigma + params["shock_sigma"]) * 100, 2)
            results[name] = metrics

        return results
