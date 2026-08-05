"""
visualize.py
Regression diagnostic plots: actual vs. predicted, residuals, rolling
coefficients over time.
"""

import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


def plot_actual_vs_fitted(model):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(model.fittedvalues, model.model.endog, alpha=0.4, s=15, color="steelblue")
    lims = [min(model.fittedvalues.min(), model.model.endog.min()),
            max(model.fittedvalues.max(), model.model.endog.max())]
    ax.plot(lims, lims, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Actual Values")
    ax.set_title("Actual vs. Fitted")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_residuals(residual_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].scatter(residual_df["fitted"], residual_df["residuals"], alpha=0.4, s=15, color="darkred")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Fitted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs. Fitted")
    axes[0].grid(alpha=0.3)

    axes[1].hist(residual_df["residuals"], bins=40, color="darkred", alpha=0.7)
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")

    plt.tight_layout()
    plt.show()


def plot_rolling_coefficients(coef_df, predictor_cols):
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in predictor_cols:
        if col in coef_df.columns:
            ax.plot(coef_df.index, coef_df[col], linewidth=1.3, label=col)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Rolling Regression Coefficients Over Time")
    ax.set_ylabel("Coefficient")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

import matplotlib.colors as mcolors

PINK_BLUE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "pink_blue", ["#f4a6c6", "#ffffff", "#a6cdf4"]
)


def plot_lag_heatmap(grid_results, horizon, predictor_x, predictor_y, metric="r_squared"):
    """
    2D heatmap of one predictor's lag vs. another's, for a single
    horizon, showing the given metric. Column names must match
    "{predictor}_lag" as produced by the grid search results.
    """
    import seaborn as sns

    x_col = f"{predictor_x}_lag"
    y_col = f"{predictor_y}_lag"

    subset = grid_results[grid_results["horizon"] == horizon]
    heatmap_data = subset.pivot(index=y_col, columns=x_col, values=metric)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(heatmap_data, cmap=PINK_BLUE_CMAP, annot=False, linewidths=0.1,
                cbar_kws={"label": metric}, ax=ax)
    ax.set_title(f"{metric} — {predictor_y} Lag x {predictor_x} Lag (horizon={horizon})", fontsize=11)
    ax.set_xlabel(f"{predictor_x} Lag (days)", fontsize=9)
    ax.set_ylabel(f"{predictor_y} Lag (days)", fontsize=9)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    plt.show()


def plot_lag_surface_3d(grid_results, horizon, predictor_x, predictor_y, metric="r_squared"):
    """
    True 3D surface plot of one predictor's lag vs. another's vs. the
    metric, for a single horizon.
    """
    import numpy as np

    x_col = f"{predictor_x}_lag"
    y_col = f"{predictor_y}_lag"

    subset = grid_results[grid_results["horizon"] == horizon]
    pivot = subset.pivot(index=y_col, columns=x_col, values=metric)

    X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
    Z = pivot.values

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap=PINK_BLUE_CMAP, edgecolor="none", alpha=0.9)

    ax.set_xlabel(f"{predictor_x} Lag (days)", fontsize=9)
    ax.set_ylabel(f"{predictor_y} Lag (days)", fontsize=9)
    ax.set_zlabel(metric, fontsize=9)
    ax.set_title(f"{metric} Surface — horizon={horizon}", fontsize=11)
    fig.colorbar(surf, shrink=0.5, aspect=10, label=metric)
    plt.tight_layout()
    plt.show()