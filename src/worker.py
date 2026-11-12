import os
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analytics.simulation_engine import MonteCarloSimulationEngine
from config.settings import settings
from config.logging_config import logger

def _run_monte_carlo(s0: float, mu: float, sigma: float, days: int = 252, num_paths: int = 10000) -> Dict[str, Any]:
    """
    Execute vectorized Monte Carlo path simulation and calculate risk metrics.
    """
    engine = MonteCarloSimulationEngine()
    _, paths = engine.simulate_gbm_paths(s0=s0, mu=mu, sigma=sigma, days=days, num_paths=num_paths)
    metrics = engine.calculate_risk_metrics(s0, paths)
    # Strip heavy trajectory arrays for serialized JSON return
    metrics.pop("cone_trajectories", None)
    metrics.pop("final_prices_sample", None)
    return metrics

try:
    from celery import Celery  # type: ignore[import-untyped, missing-import]

    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    celery_app = Celery("quantmacro_tasks", broker=REDIS_URL, backend=REDIS_URL)

    @celery_app.task(name="tasks.async_monte_carlo")
    def async_monte_carlo(s0: float, mu: float, sigma: float, days: int = 252, num_paths: int = 10000) -> Dict[str, Any]:
        """
        Asynchronous Celery task for distributed large-scale Monte Carlo batches (e.g. 100,000+ paths).
        """
        return _run_monte_carlo(s0=s0, mu=mu, sigma=sigma, days=days, num_paths=num_paths)

except (ImportError, ModuleNotFoundError):
    logger.info("Celery not installed; falling back to synchronous vectorized execution.")
    celery_app = None

    class _FallbackTask:
        """
        Synchronous fallback task wrapper supporting direct call and .delay(...) invocation.
        """
        def __init__(self, fn):
            self._fn = fn

        def __call__(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

        def delay(self, *args, **kwargs):
            class _Result:
                def __init__(self, val):
                    self._val = val
                def get(self, *a, **k):
                    return self._val
            return _Result(self._fn(*args, **kwargs))

    async_monte_carlo = _FallbackTask(_run_monte_carlo)

if __name__ == "__main__":
    if celery_app is not None:
        # On Windows, Celery requires 'solo' or 'threads' pool for reliable local execution
        default_pool = "solo" if sys.platform.startswith("win") else "prefork"
        user_args = sys.argv[1:]
        if not user_args:
            cli_argv = ["worker", "-l", "INFO", "-P", default_pool]
        elif user_args[0] == "worker":
            cli_argv = user_args
        else:
            cli_argv = ["worker"] + user_args
        celery_app.worker_main(argv=cli_argv)
    else:
        logger.info("Testing synchronous fallback Monte Carlo task...")
        result = async_monte_carlo(s0=5000.0, mu=0.08, sigma=0.18, days=252, num_paths=1000)
        logger.info(f"Simulation completed. VaR (95%): ${result.get('var_95_dollar')}")

