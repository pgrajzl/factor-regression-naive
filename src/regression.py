"""
regression.py
Regression fitting: simple/multiple OLS via statsmodels (for full
statistical output), rolling-window regression (time-varying
coefficients), and Ridge/Lasso via scikit-learn (for regularization).
Kept generic — takes a feature_df with a "target" column and any
number of predictor columns.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split


def fit_ols(feature_df, predictor_cols=None):
    """
    Fits an OLS regression via statsmodels, returning the fitted
    model object (which has .summary(), .params, .pvalues, .resid,
    .rsquared, etc.).

    predictor_cols: list of column names to use as predictors. If
        None, uses every column in feature_df except "target".
    """
    predictor_cols = predictor_cols or [c for c in feature_df.columns if c != "target"]

    y = feature_df["target"]
    X = sm.add_constant(feature_df[predictor_cols])

    model = sm.OLS(y, X).fit()
    return model


def fit_rolling_ols(feature_df, predictor_cols=None, window=252):
    """
    Fits OLS on a rolling window, refitting at each date to produce
    time-varying coefficients. Returns a DataFrame of coefficients
    (including the intercept) indexed by date, one column per
    predictor plus "const".

    Note: this refits a full regression at every single date in the
    sample, which is meaningfully slower than a single fit — expect
    this to take a bit longer on large datasets or wide windows.
    """
    predictor_cols = predictor_cols or [c for c in feature_df.columns if c != "target"]

    dates = feature_df.index[window:]
    coef_records = []

    for date in dates:
        window_data = feature_df.loc[:date].tail(window)
        y = window_data["target"]
        X = sm.add_constant(window_data[predictor_cols])

        try:
            model = sm.OLS(y, X).fit()
            coef_records.append({"date": date, **model.params.to_dict(), "r_squared": model.rsquared})
        except Exception:
            continue

    coef_df = pd.DataFrame(coef_records).set_index("date")
    return coef_df


def fit_regularized(feature_df, predictor_cols=None, method="ridge", alpha=1.0):
    """
    Fits a regularized regression (Ridge or Lasso) via scikit-learn.
    Assumes features are already standardized if scale matters (see
    features.standardize_features).

    method: "ridge" or "lasso"
    alpha: regularization strength
    """
    predictor_cols = predictor_cols or [c for c in feature_df.columns if c != "target"]

    y = feature_df["target"]
    X = feature_df[predictor_cols]

    if method == "ridge":
        model = Ridge(alpha=alpha)
    elif method == "lasso":
        model = Lasso(alpha=alpha)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ridge' or 'lasso'.")

    model.fit(X, y)
    return model


def train_test_split_ts(feature_df, test_size=0.2):
    """
    Splits time series data into train/test WITHOUT shuffling (regular
    sklearn train_test_split shuffles by default, which is invalid for
    time series — this preserves chronological order, train always
    before test).
    """
    split_idx = int(len(feature_df) * (1 - test_size))
    train = feature_df.iloc[:split_idx]
    test = feature_df.iloc[split_idx:]
    return train, test

def find_best_regression(price_series, macro_df, predictor_cols, horizons, lag_range):
    """
    Searches across horizon x lag combinations for a single price
    series (e.g. one sector's prices), returning the fitted model with
    the highest R-squared, along with its parameters.
    """
    from src.features import compute_forward_return, build_feature_matrix, build_lag_grid

    lag_configs = build_lag_grid(predictor_cols, lag_range)

    best_r2 = -float("inf")
    best_model = None
    best_params = None

    for horizon in horizons:
        target = compute_forward_return(price_series, horizon=horizon)
        for lags in lag_configs:
            feature_df = build_feature_matrix(target, macro_df, lags=lags)
            if len(feature_df) < 30:
                continue
            try:
                model = fit_ols(feature_df, predictor_cols=predictor_cols)
            except Exception:
                continue

            if model.rsquared > best_r2:
                best_r2 = model.rsquared
                best_model = model
                best_params = {"horizon": horizon, "lags": lags}

    return best_model, best_params, best_r2