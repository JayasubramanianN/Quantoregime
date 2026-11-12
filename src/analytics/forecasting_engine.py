import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from config.settings import settings
from config.logging_config import logger

class MacroForecastingEngine:
    """
    Multi-Horizon Macroeconomic Forecasting Engine.
    Combines:
    1. Prophet Bayesian Time-Series (trend changepoints, yearly/quarterly seasonality)
    2. XGBoost Regressor (lagged macroeconomic features: 1M, 3M, 12M, rolling volatility)
    3. Weighted Ensemble Blend (custom weighting, e.g. 40% Prophet + 60% XGBoost)
    with 95% upper and lower confidence intervals.
    """

    def __init__(
        self,
        prophet_weight: float = 0.40,
        xgboost_weight: float = 0.60
    ):
        self.prophet_weight = prophet_weight
        self.xgboost_weight = xgboost_weight

    def _engineer_features(
        self,
        series: pd.Series,
        exog_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Engineers lagged values (1M = 21d, 3M = 63d, 12M = 252d), rolling statistics,
        momentum indicators, and exogenous macro features.
        """
        df = pd.DataFrame({"y": series.dropna()})
        
        # Target lags
        df["lag_1"] = df["y"].shift(1)
        df["lag_5"] = df["y"].shift(5)
        df["lag_21"] = df["y"].shift(21)  # ~1 month
        df["lag_63"] = df["y"].shift(63)  # ~3 months
        df["lag_126"] = df["y"].shift(126) # ~6 months
        df["lag_252"] = df["y"].shift(252) # ~12 months

        # Rolling statistics
        df["rolling_mean_21"] = df["y"].rolling(21).mean()
        df["rolling_std_21"] = df["y"].rolling(21).std()
        df["rolling_mean_63"] = df["y"].rolling(63).mean()
        df["rolling_std_63"] = df["y"].rolling(63).std()
        
        # Momentum & returns
        df["return_21d"] = df["y"].pct_change(21)
        df["return_63d"] = df["y"].pct_change(63)

        # Merge exogenous variables if provided
        if exog_df is not None and not exog_df.empty:
            aligned_exog = exog_df.reindex(df.index).ffill().bfill()
            for col in aligned_exog.columns:
                if col != series.name:
                    df[f"exog_{col}"] = aligned_exog[col]
                    df[f"exog_{col}_lag21"] = aligned_exog[col].shift(21)

        df = df.dropna()
        X = df.drop(columns=["y"])
        y = df["y"]
        return X, y

    def fit_predict_prophet(
        self,
        series: pd.Series,
        horizon_days: int = 180,
        seasonality_mode: str = "multiplicative"
    ) -> pd.DataFrame:
        """
        Fits Prophet model on the series and produces forward trajectory with uncertainty intervals.
        """
        try:
            from prophet import Prophet
            clean_s = series.dropna()
            df_prophet = pd.DataFrame({
                "ds": clean_s.index,
                "y": clean_s.values
            })

            # Instantiate with institutional time-series params
            m = Prophet(
                growth="linear",
                seasonality_mode=seasonality_mode,
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                interval_width=0.95
            )
            # Suppress prophet diagnostic logs
            import logging
            logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
            logging.getLogger("prophet").setLevel(logging.WARNING)

            m.fit(df_prophet)
            future = m.make_future_dataframe(periods=horizon_days, freq="B")
            forecast = m.predict(future)
            
            # Extract test/forward slice
            forecast.set_index("ds", inplace=True)
            return forecast[["yhat", "yhat_lower", "yhat_upper"]]
        except Exception as e:
            logger.error(f"Prophet forecast fitting error: {e}")
            # Fallback exponential smoothing trend
            return self._fallback_prophet(series, horizon_days)

    def _fallback_prophet(self, series: pd.Series, horizon_days: int) -> pd.DataFrame:
        """
        Graceful mathematical baseline if Prophet environment raises external C++ exception.
        """
        clean_s = series.dropna()
        last_date = clean_s.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days, freq="B")
        
        # Linear trend extrapolation from past 252 days
        recent = clean_s.iloc[-252:] if len(clean_s) >= 252 else clean_s
        x_idx = np.arange(len(recent))
        poly = np.polyfit(x_idx, recent.values, 1)
        slope, intercept = poly[0], poly[1]
        
        hist_fitted = intercept + slope * x_idx
        resids = recent.values - hist_fitted
        std_err = np.std(resids)

        future_idx = np.arange(len(recent), len(recent) + horizon_days)
        future_y = intercept + slope * future_idx
        
        df_future = pd.DataFrame({
            "yhat": future_y,
            "yhat_lower": future_y - 1.96 * std_err * np.sqrt(np.linspace(1, 3, horizon_days)),
            "yhat_upper": future_y + 1.96 * std_err * np.sqrt(np.linspace(1, 3, horizon_days)),
        }, index=future_dates)
        
        df_hist = pd.DataFrame({
            "yhat": recent.values,
            "yhat_lower": recent.values - 1.96 * std_err,
            "yhat_upper": recent.values + 1.96 * std_err
        }, index=recent.index)

        return pd.concat([df_hist, df_future])

    def fit_predict_xgboost(
        self,
        series: pd.Series,
        exog_df: Optional[pd.DataFrame] = None,
        horizon_days: int = 180
    ) -> pd.DataFrame:
        """
        Fits XGBoost regressor with recursive multi-step autoregression and feature engineering.
        """
        try:
            import xgboost as xgb
            X, y = self._engineer_features(series, exog_df)
            if len(X) < 60:
                raise ValueError("Insufficient feature observations for XGBoost.")

            model = xgb.XGBRegressor(
                n_estimators=120,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=settings.RANDOM_SEED,
                verbosity=0
            )
            model.fit(X, y)

            # In-sample fitted values
            in_sample_preds = model.predict(X)
            residuals = y.values - in_sample_preds
            sigma_res = np.std(residuals)

            # Recursive forward forecasting
            last_date = series.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days, freq="B")
            
            # Start state
            hist_series = series.copy()
            forward_preds = []
            
            for i, f_date in enumerate(future_dates):
                # Engineer latest feature vector
                temp_x, _ = self._engineer_features(hist_series, exog_df)
                latest_features = temp_x.iloc[[-1]]
                next_pred = float(model.predict(latest_features)[0])
                forward_preds.append(next_pred)
                # Append predicted point for recursive lag update
                hist_series.loc[f_date] = next_pred

            future_df = pd.DataFrame({
                "xgb_pred": forward_preds,
                "xgb_lower": np.array(forward_preds) - 1.96 * sigma_res * np.sqrt(np.linspace(1, 2.5, horizon_days)),
                "xgb_upper": np.array(forward_preds) + 1.96 * sigma_res * np.sqrt(np.linspace(1, 2.5, horizon_days))
            }, index=future_dates)

            hist_df = pd.DataFrame({
                "xgb_pred": in_sample_preds,
                "xgb_lower": in_sample_preds - 1.96 * sigma_res,
                "xgb_upper": in_sample_preds + 1.96 * sigma_res
            }, index=X.index)

            return pd.concat([hist_df, future_df])

        except Exception as e:
            logger.error(f"XGBoost forecasting error: {e}")
            return self._fallback_xgboost(series, horizon_days)

    def _fallback_xgboost(self, series: pd.Series, horizon_days: int) -> pd.DataFrame:
        """
        Autoregressive fallback if XGBoost raises unexpected dimension mismatch.
        """
        clean_s = series.dropna()
        last_date = clean_s.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days, freq="B")
        
        # Exponential moving average extrapolation
        ema = clean_s.ewm(span=30).mean().iloc[-1]
        returns = clean_s.pct_change().dropna()
        mean_ret = returns.iloc[-60:].mean()
        std_ret = returns.iloc[-60:].std()

        projections = []
        curr = clean_s.iloc[-1]
        for step in range(1, horizon_days + 1):
            curr = curr * (1 + mean_ret)
            projections.append(curr)

        proj_arr = np.array(projections)
        df_future = pd.DataFrame({
            "xgb_pred": proj_arr,
            "xgb_lower": proj_arr * (1 - 1.96 * std_ret * np.sqrt(np.arange(1, horizon_days + 1))),
            "xgb_upper": proj_arr * (1 + 1.96 * std_ret * np.sqrt(np.arange(1, horizon_days + 1)))
        }, index=future_dates)
        
        df_hist = pd.DataFrame({
            "xgb_pred": clean_s.values,
            "xgb_lower": clean_s.values * (1 - 0.05),
            "xgb_upper": clean_s.values * (1 + 0.05)
        }, index=clean_s.index)

        return pd.concat([df_hist, df_future])

    def run_ensemble_forecast(
        self,
        series: pd.Series,
        exog_df: Optional[pd.DataFrame] = None,
        horizon_days: int = 180,
        prophet_weight: Optional[float] = None,
        xgboost_weight: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Blends Prophet and XGBoost predictions into a unified institutional ensemble.
        Calculates combined 95% upper and lower confidence bounds.
        """
        w_p = prophet_weight if prophet_weight is not None else self.prophet_weight
        w_x = xgboost_weight if xgboost_weight is not None else self.xgboost_weight
        # Normalize weights
        total_w = w_p + w_x
        w_p /= total_w
        w_x /= total_w

        clean_s = series.dropna()
        clean_s.index = pd.to_datetime(clean_s.index).tz_localize(None).normalize()
        
        prophet_df = self.fit_predict_prophet(clean_s, horizon_days=horizon_days)
        xgb_df = self.fit_predict_xgboost(clean_s, exog_df=exog_df, horizon_days=horizon_days)
        
        prophet_df.index = pd.to_datetime(prophet_df.index).tz_localize(None).normalize()
        xgb_df.index = pd.to_datetime(xgb_df.index).tz_localize(None).normalize()

        # Merge on overlapping dates
        merged = pd.DataFrame({
            "actual": clean_s,
            "prophet_hat": prophet_df["yhat"],
            "prophet_lower": prophet_df["yhat_lower"],
            "prophet_upper": prophet_df["yhat_upper"],
            "xgb_hat": xgb_df["xgb_pred"],
            "xgb_lower": xgb_df["xgb_lower"],
            "xgb_upper": xgb_df["xgb_upper"]
        })

        # Ensemble blend
        merged["ensemble_forecast"] = w_p * merged["prophet_hat"] + w_x * merged["xgb_hat"]
        merged["ensemble_lower_95"] = w_p * merged["prophet_lower"] + w_x * merged["xgb_lower"]
        merged["ensemble_upper_95"] = w_p * merged["prophet_upper"] + w_x * merged["xgb_upper"]

        # Future only partition
        last_hist_date = clean_s.index[-1]
        future_forecast = merged[merged.index > last_hist_date]
        historical_eval = merged[merged.index <= last_hist_date]

        # Calculate In-Sample Accuracy Metrics (MAE, RMSE, MAPE)
        if not historical_eval.empty and "actual" in historical_eval.columns:
            valid_hist = historical_eval.dropna(subset=["actual", "ensemble_forecast"])
            mae = float(np.mean(np.abs(valid_hist["actual"] - valid_hist["ensemble_forecast"])))
            rmse = float(np.sqrt(np.mean((valid_hist["actual"] - valid_hist["ensemble_forecast"]) ** 2)))
            mape = float(np.mean(np.abs((valid_hist["actual"] - valid_hist["ensemble_forecast"]) / valid_hist["actual"]))) * 100
        else:
            mae, rmse, mape = 0.0, 0.0, 0.0

        current_price = float(clean_s.iloc[-1])
        projected_mean = float(future_forecast["ensemble_forecast"].iloc[-1]) if not future_forecast.empty else current_price
        expected_change_pct = float((projected_mean - current_price) / current_price * 100)

        return {
            "ticker": series.name or "Asset",
            "horizon_days": horizon_days,
            "weights": {"prophet": round(w_p, 2), "xgboost": round(w_x, 2)},
            "current_price": round(current_price, 2),
            "projected_price": round(projected_mean, 2),
            "expected_change_pct": round(expected_change_pct, 2),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 2),
            "forecast_table": merged,
            "future_dates": [d.strftime("%Y-%m-%d") for d in future_forecast.index],
            "future_ensemble": future_forecast["ensemble_forecast"].round(2).tolist(),
            "future_lower": future_forecast["ensemble_lower_95"].round(2).tolist(),
            "future_upper": future_forecast["ensemble_upper_95"].round(2).tolist(),
            "future_prophet": future_forecast["prophet_hat"].round(2).tolist(),
            "future_xgboost": future_forecast["xgb_hat"].round(2).tolist()
        }
