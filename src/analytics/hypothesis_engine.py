import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from config.settings import settings
from config.logging_config import logger

class HypothesisEngine:
    """
    Institutional-Grade Quantitative Hypothesis Testing Engine.
    Executes Augmented Dickey-Fuller (ADF) stationarity tests, automated differencing,
    bivariate rolling correlations, Granger causality, and regime split t-tests
    with rigorous econometric interpretation.
    """

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def adf_stationarity_test(
        self,
        series: pd.Series,
        max_diff: int = 1
    ) -> Dict[str, Any]:
        """
        Augmented Dickey-Fuller (ADF) Unit Root Test.
        Tests raw levels first (d=0). If non-stationary (p >= alpha), evaluates
        order-1 first differencing (d=1) to confirm integration order I(1).

        Null Hypothesis (H0): Unit root present (series is non-stationary).
        Alternative (H1): Series is stationary with time-invariant mean and variance.
        """
        clean_series = series.dropna()
        if len(clean_series) < 20:
            return {
                "test_name": "Augmented Dickey-Fuller (ADF)",
                "series_name": series.name or "Series",
                "raw_statistic": None,
                "raw_p_value": None,
                "raw_is_stationary": False,
                "differencing_order": 0,
                "statistic": None,
                "p_value": None,
                "is_significant": False,
                "is_stationary": False,
                "status_label": "INSUFFICIENT_DATA",
                "interpretation": "Insufficient data points (< 20) for reliable ADF test."
            }

        # 1. Test Raw Levels (d=0)
        raw_res = adfuller(clean_series, autolag="AIC")
        raw_stat = float(raw_res[0])
        raw_pval = float(raw_res[1])
        raw_crit = {k: float(v) for k, v in raw_res[4].items()}
        raw_is_stat = bool(raw_pval < self.significance_level)

        # 2. Test First Differences (d=1)
        diff_series = clean_series.diff().dropna()
        diff_stat = None
        diff_pval = None
        diff_crit = {}
        diff_is_stat = False

        if len(diff_series) >= 20:
            diff_res = adfuller(diff_series, autolag="AIC")
            diff_stat = float(diff_res[0])
            diff_pval = float(diff_res[1])
            diff_crit = {k: float(v) for k, v in diff_res[4].items()}
            diff_is_stat = bool(diff_pval < self.significance_level)

        if raw_is_stat:
            order_d = 0
            status_label = "STATIONARY (Raw Levels)"
            interpretation = (
                f"Series is stationary in raw levels (ADF={raw_stat:.4f}, p={raw_pval:.4f} < {self.significance_level}). "
                "Reject H0: No unit root present (integrated of order I(0)). Mean and variance are time-invariant."
            )
        else:
            order_d = 1
            if diff_is_stat:
                status_label = "STATIONARY (Differenced d=1)"
                diff_p_formatted = f"{diff_pval:.4e}" if diff_pval < 0.0001 else f"{diff_pval:.4f}"
                interpretation = (
                    f"Raw series is NON-STATIONARY (ADF={raw_stat:.4f}, p={raw_pval:.4f} >= {self.significance_level}). "
                    f"Achieved stationarity after order-1 differencing (ADF={diff_stat:.4f}, p={diff_p_formatted} < {self.significance_level}). "
                    "Series is integrated of order I(1). Differenced increments (e.g. daily changes or returns) must be used in econometric modeling."
                )
            else:
                status_label = "NON-STATIONARY"
                interpretation = (
                    f"Series is non-stationary in raw levels (ADF={raw_stat:.4f}, p={raw_pval:.4f}) and remained non-stationary "
                    f"after order-1 differencing (ADF={diff_stat:.4f if diff_stat is not None else 'N/A'}, p={diff_pval:.4f if diff_pval is not None else 'N/A'})."
                )

        return {
            "test_name": "Augmented Dickey-Fuller (ADF)",
            "series_name": series.name or "Series",
            "differencing_order": order_d,
            "raw_statistic": round(raw_stat, 4),
            "raw_p_value": float(raw_pval),
            "raw_is_stationary": raw_is_stat,
            "differenced_statistic": round(diff_stat, 4) if diff_stat is not None else None,
            "differenced_p_value": float(diff_pval) if diff_pval is not None else None,
            "differenced_is_stationary": diff_is_stat,
            "statistic": round(raw_stat, 4),
            "p_value": float(raw_pval),
            "is_significant": raw_is_stat,
            "is_stationary": raw_is_stat,
            "status_label": status_label,
            "critical_values": raw_crit,
            "differenced_critical_values": diff_crit,
            "interpretation": interpretation
        }

    def align_series(
        self,
        x: pd.Series,
        y: pd.Series,
        freq: str = "D"
    ) -> pd.DataFrame:
        """
        Institutional-grade temporal alignment of bivariate time-series.
        If inputs possess DatetimeIndex, strips timezone offsets, normalizes timestamps,
        resamples to target frequency (default daily 'D'), and forward-fills gaps to eliminate
        monthly-to-daily frequency mismatches (e.g. FRED vs Yahoo Finance).
        If inputs possess standard non-datetime indices, performs clean common-index alignment.
        """
        s_x = x.dropna().copy()
        s_y = y.dropna().copy()

        # Check if index is datetime-like
        is_dt_x = isinstance(s_x.index, pd.DatetimeIndex) or (len(s_x) > 0 and isinstance(s_x.index[0], (pd.Timestamp, str)) and pd.to_datetime(s_x.index[:3], errors='coerce').notna().all())
        is_dt_y = isinstance(s_y.index, pd.DatetimeIndex) or (len(s_y) > 0 and isinstance(s_y.index[0], (pd.Timestamp, str)) and pd.to_datetime(s_y.index[:3], errors='coerce').notna().all())

        if is_dt_x and is_dt_y:
            s_x.index = pd.to_datetime(s_x.index).tz_localize(None).normalize()
            s_y.index = pd.to_datetime(s_y.index).tz_localize(None).normalize()

            # Resample both series to target frequency and forward-fill
            s_x_daily = s_x.resample(freq).last().ffill()
            s_y_daily = s_y.resample(freq).last().ffill()

            df_aligned = pd.concat([s_x_daily.rename("X"), s_y_daily.rename("Y")], axis=1).dropna()
        else:
            df_aligned = pd.concat([s_x.rename("X"), s_y.rename("Y")], axis=1).dropna()

        return df_aligned

    def compute_rolling_correlation(
        self,
        x: pd.Series,
        y: pd.Series,
        window: int = 60,
        min_periods: Optional[int] = None
    ) -> pd.Series:
        """
        Computes rolling Pearson bivariate correlation timeline.
        Guarantees non-empty output by automatically aligning daily and monthly frequencies via .resample('D').ffill().
        """
        df_aligned = self.align_series(x, y, freq="D")
        if len(df_aligned) < 15:
            return pd.Series(dtype=float, name="rolling_corr")

        eff_window = min(window, max(15, len(df_aligned) // 2))
        if min_periods is None:
            min_periods = max(10, eff_window // 3)

        rolling_corr = df_aligned["X"].rolling(window=eff_window, min_periods=min_periods).corr(df_aligned["Y"])
        # Eliminate any numerical infs caused by zero-variance sub-windows (e.g. constant Fed Funds or CPI over short windows)
        rolling_corr = rolling_corr.replace([np.inf, -np.inf], np.nan).ffill().bfill().clip(-1.0, 1.0).dropna()
        rolling_corr.name = f"{eff_window}d_rolling_corr"
        return rolling_corr

    def granger_causality_test(
        self,
        x: pd.Series,
        y: pd.Series,
        max_lags: int = 5,
        ensure_stationarity: bool = True
    ) -> Dict[str, Any]:
        """
        Tests whether variable X Granger-causes variable Y.
        Null Hypothesis (H0): Lagged values of X do NOT improve prediction of Y.
        Alternative (H1): Lagged values of X provide statistically significant predictive information for Y.
        """
        df = self.align_series(x, y, freq="D")
        if len(df) < max_lags * 4 + 10:
            return {
                "test_name": "Granger Causality",
                "x_name": x.name or "X",
                "y_name": y.name or "Y",
                "error": "Insufficient overlapping observations for Granger causality test.",
                "is_significant": False
            }

        # Auto-difference non-stationary raw data to prevent spurious regressions
        if ensure_stationarity:
            adf_y = self.adf_stationarity_test(df["Y"])
            adf_x = self.adf_stationarity_test(df["X"])
            
            # Explicitly force first-order differencing if either series is non-stationary
            if not adf_y["raw_is_stationary"] or not adf_x["raw_is_stationary"]:
                df = df.diff().dropna()

        lag_results: Dict[int, Dict[str, Any]] = {}
        min_p_value = 1.0
        best_lag = 1
        significant_lags = []

        try:
            gc_res = grangercausalitytests(df[["Y", "X"]], maxlag=max_lags, verbose=False)
            
            for lag, data in gc_res.items():
                f_test = data[0]["ssr_ftest"]
                f_stat = float(f_test[0])
                p_val = float(f_test[1])
                
                chi2_test = data[0]["ssr_chi2test"]
                chi2_stat = float(chi2_test[0])
                chi2_pval = float(chi2_test[1])

                is_sig = bool(p_val < self.significance_level)
                if is_sig:
                    significant_lags.append(lag)
                if p_val < min_p_value:
                    min_p_value = p_val
                    best_lag = lag

                lag_results[lag] = {
                    "lag": lag,
                    "f_stat": round(f_stat, 4),
                    "p_value": float(p_val),
                    "chi2_stat": round(chi2_stat, 4),
                    "chi2_p_value": float(chi2_pval),
                    "is_significant": is_sig
                }

        except Exception as e:
            logger.error(f"Granger causality execution error: {e}")
            return {
                "test_name": "Granger Causality",
                "x_name": x.name or "X",
                "y_name": y.name or "Y",
                "error": str(e),
                "is_significant": False
            }

        overall_sig = len(significant_lags) > 0
        x_name = x.name or "Variable X"
        y_name = y.name or "Variable Y"
        best_f = lag_results[best_lag]["f_stat"] if best_lag in lag_results else 0.0

        if overall_sig:
            interpretation = (
                f"Statistically significant causality detected: '{x_name}' Granger-causes '{y_name}' "
                f"at Lag {best_lag} (F-stat={best_f:.2f}, p={min_p_value:.4f} < {self.significance_level}). "
                f"Reject H0: Historical movements in {x_name} contain incremental predictive power for future {y_name}."
            )
        else:
            interpretation = (
                f"No significant Granger causality detected from '{x_name}' to '{y_name}' across 1 to {max_lags} lags "
                f"(Lowest p-value: {min_p_value:.4f} >= {self.significance_level} at Lag {best_lag}). "
                f"Fail to reject H0: '{x_name}' does not statistically precede '{y_name}' over these horizons."
            )

        return {
            "test_name": "Granger Causality",
            "x_name": x_name,
            "y_name": y_name,
            "best_lag": best_lag,
            "min_p_value": float(min_p_value),
            "is_significant": overall_sig,
            "significant_lags": significant_lags,
            "lag_details": lag_results,
            "interpretation": interpretation
        }

    def parametric_t_test(
        self,
        sample_a: pd.Series,
        sample_b: pd.Series,
        equal_var: bool = False
    ) -> Dict[str, Any]:
        """Student's / Welch's Two-Sample t-test."""
        s_a = sample_a.dropna()
        s_b = sample_b.dropna()

        if len(s_a) < 5 or len(s_b) < 5:
            return {
                "test_name": "Two-Sample t-test",
                "statistic": None,
                "p_value": None,
                "is_significant": False,
                "interpretation": "Insufficient observations in one or both samples."
            }

        t_stat, p_val = stats.ttest_ind(s_a, s_b, equal_var=equal_var)
        diff_mean = float(s_a.mean() - s_b.mean())

        se_diff = np.sqrt(s_a.var(ddof=1)/len(s_a) + s_b.var(ddof=1)/len(s_b))
        df_welch = ((s_a.var(ddof=1)/len(s_a) + s_b.var(ddof=1)/len(s_b))**2) / (
            (s_a.var(ddof=1)/len(s_a))**2 / (len(s_a) - 1) + 
            (s_b.var(ddof=1)/len(s_b))**2 / (len(s_b) - 1)
        ) if not equal_var else (len(s_a) + len(s_b) - 2)
        
        t_crit = stats.t.ppf(1 - self.significance_level / 2, df=df_welch)
        ci_lower = diff_mean - t_crit * se_diff
        ci_upper = diff_mean + t_crit * se_diff

        is_sig = bool(p_val < self.significance_level)
        p_str = f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}"
        
        interpretation = (
            f"Statistically significant difference in means (t={t_stat:.4f}, p={p_str} < {self.significance_level}). "
            f"Mean difference: {diff_mean:.4f} [95% CI: {ci_lower:.4f}, {ci_upper:.4f}]. "
            "Reject H0: The two regimes/samples exhibit distinct mean performance."
            if is_sig else
            f"No statistically significant difference in means (t={t_stat:.4f}, p={p_str} >= {self.significance_level}). "
            f"Mean difference: {diff_mean:.4f} [95% CI: {ci_lower:.4f}, {ci_upper:.4f}]. "
            "Fail to reject H0."
        )

        return {
            "test_name": "Two-Sample Welch t-test" if not equal_var else "Two-Sample Student t-test",
            "sample_a_mean": round(float(s_a.mean()), 4),
            "sample_b_mean": round(float(s_b.mean()), 4),
            "mean_difference": round(diff_mean, 4),
            "statistic": round(float(t_stat), 4),
            "p_value": float(p_val),
            "ci_95": (round(ci_lower, 4), round(ci_upper, 4)),
            "is_significant": is_sig,
            "interpretation": interpretation
        }

    def regime_split_t_test(
        self,
        x: pd.Series,
        y: pd.Series
    ) -> Dict[str, Any]:
        """
        Splits response series Y across the median of driver series X,
        then evaluates two-sample Welch t-test for difference in regime means.
        """
        df_aligned = self.align_series(x, y, freq="D")
        if len(df_aligned) < 10:
            return {
                "test_name": "Two-Sample Welch t-test",
                "statistic": None,
                "p_value": None,
                "is_significant": False,
                "interpretation": "Insufficient overlapping observations for regime split."
            }

        median_x = float(df_aligned["X"].median())
        high_group = df_aligned.loc[df_aligned["X"] >= median_x, "Y"]
        low_group = df_aligned.loc[df_aligned["X"] < median_x, "Y"]

        t_res = self.parametric_t_test(high_group, low_group, equal_var=False)
        t_res["median_split_threshold"] = round(median_x, 4)
        t_res["predictor_name"] = x.name or "Predictor X"
        t_res["target_name"] = y.name or "Target Y"
        return t_res

    def correlation_analysis(
        self,
        x: pd.Series,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Computes Pearson and Spearman correlations with Fisher Z confidence interval."""
        df = self.align_series(x, y, freq="D")
        n = len(df)
        if n < 10:
            return {
                "test_name": "Correlation Analysis",
                "error": "Insufficient overlapping data points.",
                "is_significant": False
            }

        r_p, p_p = stats.pearsonr(df["X"], df["Y"])
        r_s, p_s = stats.spearmanr(df["X"], df["Y"])

        # Fisher z-transformation for Pearson 95% CI
        z = np.arctanh(np.clip(r_p, -0.9999, 0.9999))
        se_z = 1.0 / np.sqrt(n - 3) if n > 3 else 1.0
        z_crit = stats.norm.ppf(1 - self.significance_level / 2)
        ci_lower = np.tanh(z - z_crit * se_z)
        ci_upper = np.tanh(z + z_crit * se_z)

        is_sig = bool(p_p < self.significance_level)
        p_str = f"{p_p:.4e}" if p_p < 0.0001 else f"{p_p:.4f}"

        interpretation = (
            f"Significant linear association (Pearson r={r_p:.3f}, p={p_str}; Spearman rho={r_s:.3f}). "
            f"95% CI for Pearson r: [{ci_lower:.3f}, {ci_upper:.3f}]."
            if is_sig else
            f"No statistically significant correlation (Pearson r={r_p:.3f}, p={p_str} >= {self.significance_level})."
        )

        return {
            "test_name": "Correlation Analysis",
            "pearson_r": round(float(r_p), 4),
            "pearson_p_value": float(p_p),
            "spearman_rho": round(float(r_s), 4),
            "spearman_p_value": float(p_s),
            "ci_95": (round(float(ci_lower), 4), round(float(ci_upper), 4)),
            "is_significant": is_sig,
            "sample_size": n,
            "interpretation": interpretation
        }

    def run_full_bivariate_suite(
        self,
        series_x: pd.Series,
        series_y: pd.Series,
        max_lags: int = 5
    ) -> Dict[str, Any]:
        """
        Runs comprehensive institutional test pipeline:
        1. ADF stationarity testing for both series (raw levels & differenced)
        2. Bivariate correlation analysis (Pearson, Spearman, Fisher CI)
        3. Bidirectional Granger causality tests
        4. Rolling 60-day correlation series (daily aligned)
        5. Regime split Welch t-test
        """
        adf_x = self.adf_stationarity_test(series_x)
        adf_y = self.adf_stationarity_test(series_y)
        corr = self.correlation_analysis(series_x, series_y)
        gc_x_to_y = self.granger_causality_test(series_x, series_y, max_lags=max_lags)
        gc_y_to_x = self.granger_causality_test(series_y, series_x, max_lags=max_lags)
        rolling_corr = self.compute_rolling_correlation(series_x, series_y, window=60)
        regime_t = self.regime_split_t_test(series_x, series_y)

        return {
            "variable_x": series_x.name or "Variable X",
            "variable_y": series_y.name or "Variable Y",
            "stationarity_x": adf_x,
            "stationarity_y": adf_y,
            "correlation": corr,
            "granger_x_causes_y": gc_x_to_y,
            "granger_y_causes_x": gc_y_to_x,
            "rolling_correlation": rolling_corr,
            "regime_t_test": regime_t
        }