"""
evaluation.py
Regression diagnostics: R-squared, residual analysis, out-of-sample
error, coefficient significance summaries.
"""

import pandas as pd
import numpy as np


def summarize_ols(model):
    """Pulls the key stats out of a fitted statsmodels OLS result into a clean DataFrame."""
    summary = pd.DataFrame({
        "coefficient": model.params,
        "std_err": model.bse,
        "t_stat": model.tvalues,
        "p_value": model.pvalues,
    })
    summary["significant"] = summary["p_value"] < 0.05
    return summary, model.rsquared, model.rsquared_adj


def compute_residual_diagnostics(model):
    """Returns residuals and fitted values for diagnostic plotting."""
    return pd.DataFrame({
        "fitted": model.fittedvalues,
        "residuals": model.resid,
    })


def evaluate_out_of_sample(model, test_df, predictor_cols, target_col="target"):
    """
    Evaluates a fitted model's out-of-sample performance: predicts on
    test_df and computes R-squared and RMSE against actual values.
    Works with either a statsmodels or scikit-learn fitted model.
    """
    import statsmodels.api as sm

    X_test = test_df[predictor_cols]
    y_test = test_df[target_col]

    if hasattr(model, "predict") and "statsmodels" in str(type(model)):
        X_test_sm = sm.add_constant(X_test, has_constant="add")
        predictions = model.predict(X_test_sm)
    else:
        predictions = model.predict(X_test)

    residuals = y_test - predictions
    ss_res = (residuals ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum()
    r_squared_oos = 1 - (ss_res / ss_tot)
    rmse = np.sqrt((residuals ** 2).mean())

    return {"OOS R-squared": r_squared_oos, "OOS RMSE": rmse}

def evaluate_regression_combo(horizon, lags, predictor_cols, spy_prices, macro_df, method="ols"):
    """
    Fits a regression for one specific combination of forward-return
    horizon and predictor lags, returning key fit statistics. This
    function must live in a real module (not defined inline in a
    notebook) since multiprocessing on macOS uses spawn, which
    requires functions to be importable by module path.

    horizon: forward return horizon for the target
    lags: dict of {predictor_col: lag_periods}
    predictor_cols: list of predictor column names to use
    spy_prices, macro_df: raw data, passed through so each worker
        process has what it needs (spawn doesn't inherit notebook
        variables)
    """
    from src.features import compute_forward_return, build_feature_matrix
    from src.regression import fit_ols

    target = compute_forward_return(spy_prices, horizon=horizon)
    feature_df = build_feature_matrix(target, macro_df, lags=lags)

    if len(feature_df) < 30:
        return {"r_squared": float("nan"), "r_squared_adj": float("nan"), "n_obs": len(feature_df)}

    try:
        model = fit_ols(feature_df, predictor_cols=predictor_cols)
    except Exception:
        return {"r_squared": float("nan"), "r_squared_adj": float("nan"), "n_obs": len(feature_df)}

    return {
        "r_squared": model.rsquared,
        "r_squared_adj": model.rsquared_adj,
        "n_obs": len(feature_df),
    }

def find_best_regression_by_sector(sector_prices, macro_df, predictor_cols, horizons, lag_range):
    """
    Runs find_best_regression independently for each sector's price
    series, returning a dict of {sector: (model, params, r_squared)}.
    """
    from src.regression import find_best_regression

    results = {}
    for sector, price_df in sector_prices.items():
        print(f"Finding best regression for {sector}...")
        model, params, r2 = find_best_regression(
            price_df, macro_df, predictor_cols, horizons, lag_range
        )
        results[sector] = {"model": model, "params": params, "r_squared": r2}

    return results


def summarize_sector_results(sector_results):
    """
    Turns the dict from find_best_regression_by_sector into a clean
    comparison DataFrame: sector, best horizon, best lags, r_squared,
    and each predictor's coefficient/p-value.
    """
    rows = []
    for sector, result in sector_results.items():
        model = result["model"]
        row = {
            "Sector": sector,
            "R_squared": result["r_squared"],
            "Horizon": result["params"]["horizon"],
        }
        for predictor, lag in result["params"]["lags"].items():
            row[f"{predictor}_lag"] = lag
        if model is not None:
            for predictor in model.params.index:
                if predictor != "const":
                    row[f"{predictor}_coef"] = model.params[predictor]
                    row[f"{predictor}_pvalue"] = model.pvalues[predictor]
        rows.append(row)

    return pd.DataFrame(rows).sort_values("R_squared", ascending=False)