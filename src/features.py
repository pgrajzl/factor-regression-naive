"""
features.py
Transforms raw price/macro data into regression-ready features:
computing returns/volatility targets, aligning frequencies, applying
lags. Kept generic so any target/predictor combination can flow
through the same functions.
"""

import pandas as pd
import numpy as np


def compute_returns(prices, price_col="Close"):
    """Simple daily returns from a price DataFrame/Series."""
    if isinstance(prices, pd.DataFrame):
        return prices[price_col].pct_change()
    return prices.pct_change()

def compute_forward_return(prices, horizon=1, price_col="Close"):
    """
    Forward return over `horizon` days, aligned so the value on date T
    represents the return from T to T+horizon — i.e., what you'd earn
    holding a position entered on date T. Use this as the regression
    target to test predicting returns over different horizons, rather
    than just the next single day.
    """
    if isinstance(prices, pd.DataFrame):
        prices = prices[price_col]
    return prices.pct_change(horizon).shift(-horizon)


def compute_rolling_volatility(returns, window=20):
    """Rolling realized volatility (std of returns) as an alternative target."""
    return returns.rolling(window).std()


def apply_lag(series, lag=1):
    """
    Shifts a series forward by `lag` periods, so that today's row
    contains the value from `lag` periods ago. Use this on predictors
    to avoid using same-day information that wouldn't actually have
    been available yet, or to test whether a predictor's effect shows
    up with a delay.
    """
    return series.shift(lag)


def build_feature_matrix(target, predictors_df, lags=None, dropna=True):
    """
    Combines a target Series with one or more predictor Series into a
    single aligned DataFrame, with optional per-predictor lags.

    target: Series (e.g. SPY daily returns)
    predictors_df: DataFrame of predictor columns (e.g. VIX, Yield_Slope)
    lags: dict of {column_name: lag_periods}, e.g. {"VIX": 1, "Yield_Slope": 0}.
        Any predictor not in this dict defaults to lag=0 (contemporaneous).
        Pass None to use lag=0 for everything.
    dropna: if True, drops any row with a missing value in target or
        any predictor column (standard for regression-ready data)
    """
    lags = lags or {}

    feature_df = pd.DataFrame(index=target.index)
    feature_df["target"] = target

    for col in predictors_df.columns:
        lag = lags.get(col, 0)
        feature_df[col] = apply_lag(predictors_df[col], lag=lag).reindex(target.index).ffill()

    if dropna:
        feature_df = feature_df.dropna()

    return feature_df


def standardize_features(feature_df, exclude_cols=("target",)):
    """
    Z-scores all columns except those in exclude_cols (typically the
    target). Useful before regularized regression (Ridge/Lasso), which
    is sensitive to feature scale.
    """
    scaled = feature_df.copy()
    predictor_cols = [c for c in feature_df.columns if c not in exclude_cols]

    for col in predictor_cols:
        mean = scaled[col].mean()
        std = scaled[col].std()
        scaled[col] = (scaled[col] - mean) / std

    return scaled

def build_lag_grid(predictor_names, lag_range):
    """
    Builds all combinations of lags across an arbitrary list of
    predictors. Returns a list of lag-config dicts suitable for the
    'lags' parameter in a grid search.
    """
    import itertools
    lag_options = [lag_range] * len(predictor_names)
    return [dict(zip(predictor_names, combo)) for combo in itertools.product(*lag_options)]


def split_lag_columns(grid_results, predictor_names):
    """
    Breaks the 'lags' dict column in grid search results into separate
    '{predictor}_lag' columns for easier filtering/plotting.
    """
    df = grid_results.copy()
    for name in predictor_names:
        df[f"{name}_lag"] = df["lags"].apply(lambda d: d[name])
    return df.drop(columns=["lags"])