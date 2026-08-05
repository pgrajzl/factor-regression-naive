"""
optimization.py
Parallel grid search for regression parameter combinations (forward
return horizon, predictor lags), using multiprocessing to spread work
across CPU cores.
"""

import itertools
from multiprocessing import Pool, cpu_count
import pandas as pd


def _evaluate_combo(args):
    """
    Internal worker function. evaluate_fn must be picklable (a plain
    module-level function) since multiprocessing needs to send it to
    worker processes via spawn.
    """
    evaluate_fn, params = args
    result = evaluate_fn(**params)
    return {**params, **result}


def run_grid_search_parallel(evaluate_fn, param_grid, fixed_args=None, n_jobs=None):
    """
    Runs evaluate_fn across every combination in param_grid, in
    parallel across CPU cores.

    evaluate_fn: a function (must be importable from a real module)
        that takes the grid parameters plus fixed_args as keyword
        arguments, and returns a dict of result metrics
    param_grid: dict of {param_name: list_of_values}
    fixed_args: dict of additional keyword arguments passed to every
        call, unchanged across the grid (e.g. shared price/macro data)
    n_jobs: number of parallel worker processes (defaults to all
        available cores minus one)
    """
    n_jobs = n_jobs or max(1, cpu_count() - 1)
    fixed_args = fixed_args or {}

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    print(f"Running {len(combos)} combinations across {n_jobs} processes...")

    args = [(evaluate_fn, {**combo, **fixed_args}) for combo in combos]

    with Pool(processes=n_jobs) as pool:
        results = pool.map(_evaluate_combo, args)

    # Drop the large fixed_args from result rows before building the DataFrame
    for r in results:
        for key in fixed_args:
            r.pop(key, None)

    results_df = pd.DataFrame(results)
    return results_df