# examples/j_investment_eval.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import math
import time

import numpy as np
import pandas as pd


DATA_FILE_NAME = "j_raw_data_2022-01-01_2025-08-29.csv"
ROLLING_WINDOW = 100
TRAIN_RATIO = 0.8


def _load_close_series() -> np.ndarray:
    """
    Load the CSV and return the 'close' column as a 1D numpy array of floats.
    Assumes the file has a 'close' column and an extra first column (timestamp/index).
    """
    data_path = Path(__file__).resolve().parent / DATA_FILE_NAME
    df = pd.read_csv(data_path)

    if "close" not in df.columns:
        raise ValueError(f"'close' column not found in {DATA_FILE_NAME}")

    # Ensure float dtype
    closes = df["close"].astype(float).to_numpy()
    if closes.ndim != 1 or closes.size == 0:
        raise ValueError("Close price series is empty or has wrong shape")

    return closes


def _get_predict_function(program_module, task_def):
    """
    Retrieve the predict function from the candidate module,
    using the function name specified in the task definition.
    """
    # Support either `function_name` or `function_name_to_evolve`
    func_name = getattr(task_def, "function_name", None) or getattr(
        task_def, "function_name_to_evolve", None
    )
    if not func_name:
        raise AttributeError("TaskDefinition does not specify a function name")

    if not hasattr(program_module, func_name):
        raise AttributeError(
            f"Candidate module does not define required function '{func_name}'"
        )

    return getattr(program_module, func_name)


def _safe_predict(predict_fn, history_closes, required_len: int) -> np.ndarray:
    """
    Call predict_fn(history_closes) and perform sanity checks.
    Ensure we get at least `required_len` numeric predictions.
    """
    output = predict_fn(history_closes)

    # Convert to list first
    if not isinstance(output, list):
        # Try to coerce to list if it's some iterable
        try:
            output = list(output)
        except Exception as e:  # noqa: BLE001
            raise TypeError(f"Prediction is not a list-like object: {e}") from e

    if len(output) < required_len:
        raise ValueError(
            f"Prediction length {len(output)} is smaller than required {required_len}"
        )

    # Take the first required_len elements and convert to float array
    try:
        preds = np.array(output[:required_len], dtype=float)
    except Exception as e:  # noqa: BLE001
        raise TypeError(f"Prediction elements are not numeric: {e}") from e

    return preds


def evaluate_candidate(program_module, task_def) -> Dict[str, Any]:
    """
    Main entry point for OpenAlpha_Evolve's metrics-based evaluation.

    Returns a dict of scalar metrics, including:
      - total_mse: sum of rolling-window MSEs on the test set
      - neg_total_mse: negative of total_mse (for maximization)
      - num_windows: how many 100-step windows were evaluated
      - num_points: how many ground-truth points were used in total
      - elapsed: evaluation wall-clock time in seconds
    """
    t0 = time.time()

    try:
        closes = _load_close_series()
    except Exception as e:  # noqa: BLE001
        # Fatal data error: return a huge penalty so the candidate is strongly disfavored
        total_mse = float("inf")
        elapsed = time.time() - t0
        return {
            "total_mse": total_mse,
            "neg_total_mse": -1e12,
            "num_windows": 0,
            "num_points": 0,
            "elapsed": elapsed,
            "error": f"Data loading error: {e}",
        }

    n = closes.shape[0]
    if n < 2 * ROLLING_WINDOW:
        # Not enough data to meaningfully split and roll
        total_mse = float("inf")
        elapsed = time.time() - t0
        return {
            "total_mse": total_mse,
            "neg_total_mse": -1e12,
            "num_windows": 0,
            "num_points": 0,
            "elapsed": elapsed,
            "error": "Not enough data for rolling evaluation",
        }

    split_idx = int(math.floor(TRAIN_RATIO * n))
    train = closes[:split_idx]
    test = closes[split_idx:]

    try:
        predict_fn = _get_predict_function(program_module, task_def)
    except Exception as e:  # noqa: BLE001
        total_mse = float("inf")
        elapsed = time.time() - t0
        return {
            "total_mse": total_mse,
            "neg_total_mse": -1e12,
            "num_windows": 0,
            "num_points": 0,
            "elapsed": elapsed,
            "error": f"Function lookup error: {e}",
        }

    history = list(float(x) for x in train)
    pos = 0
    total_mse = 0.0
    num_windows = 0
    num_points = 0

    # Rolling evaluation over the test set, 100-step windows
    while pos < len(test):
        true_window = test[pos : pos + ROLLING_WINDOW]
        L = len(true_window)
        if L == 0:
            break

        try:
            preds = _safe_predict(predict_fn, history, required_len=L)
        except Exception as e:  # noqa: BLE001
            # If the candidate misbehaves, assign a huge penalty and stop
            total_mse = float("inf")
            elapsed = time.time() - t0
            return {
                "total_mse": total_mse,
                "neg_total_mse": -1e12,
                "num_windows": num_windows,
                "num_points": num_points,
                "elapsed": elapsed,
                "error": f"Prediction error: {e}",
            }

        # Compute MSE for this window
        diff = preds - true_window
        mse = float(np.mean(diff * diff))

        total_mse += mse
        num_windows += 1
        num_points += L

        # Append the true values (not predictions) to history
        history.extend(float(x) for x in true_window)

        pos += ROLLING_WINDOW

    elapsed = time.time() - t0

    if math.isinf(total_mse) or math.isnan(total_mse):
        neg_total_mse = -1e12
    else:
        neg_total_mse = -float(total_mse)

    metrics: Dict[str, Any] = {
        "total_mse": float(total_mse),
        "neg_total_mse": float(neg_total_mse),
        "num_windows": int(num_windows),
        "num_points": int(num_points),
        "elapsed": float(elapsed),
    }

    return metrics
