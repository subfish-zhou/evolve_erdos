# test_j_investment_eval.py

from __future__ import annotations

from pathlib import Path
import math
import types

import numpy as np

from j_investment_eval import evaluate_candidate, DATA_FILE_NAME


def create_dummy_csv():
    """
    Create a dummy CSV file with at least 200 rows of synthetic price data.
    This ensures the evaluator has enough data for the 0.8/0.2 split and
    rolling-window evaluation.
    """
    this_dir = Path(__file__).resolve().parent
    csv_path = this_dir / DATA_FILE_NAME

    # Simple synthetic data: linearly increasing close price
    num_rows = 200
    with csv_path.open("w", encoding="utf-8") as f:
        # Header: first empty column name, then open, high, low, close, volume
        f.write(",open,high,low,close,volume\n")
        for i in range(num_rows):
            # Timestamp or index in the first column (not used by evaluator)
            # Here we just use a simple index i as string.
            ts = str(i)
            base_price = 100.0 + 0.1 * i
            open_p = base_price
            high_p = base_price + 1.0
            low_p = base_price - 1.0
            close_p = base_price + 0.5  # simple trend
            volume = 1000 + i
            f.write(f"{ts},{open_p},{high_p},{low_p},{close_p},{volume}\n")

    print(f"Dummy CSV created at: {csv_path}")


def dummy_predict_next_100(history_closes):
    """
    A very simple baseline predictor:
    - Always predicts the last observed close price for all 100 future steps.
    This is just for testing that the evaluation pipeline runs end-to-end.
    """
    if not history_closes:
        # If somehow called with empty history, just return zeros.
        last = 0.0
    else:
        last = float(history_closes[-1])

    return [last] * 100


class DummyTaskDef:
    """
    Minimal stand-in for the real TaskDefinition.
    We only need to provide the function name attribute used in _get_predict_function.
    """

    # If your real TaskDefinition uses "function_name_to_evolve",
    # you can change this attribute name accordingly.
    function_name = "predict_next_100"

    # Optional: if you want to mimic metrics config (not strictly required here)
    metrics_primary_key = "neg_total_mse"
    metrics_scalarization = {"neg_total_mse": 1.0}


def main():
    # 1) Ensure dummy CSV exists
    create_dummy_csv()

    # 2) Build a fake "program module" that has predict_next_100 as an attribute.
    program_module = types.SimpleNamespace(predict_next_100=dummy_predict_next_100)

    # 3) Create a minimal task definition object
    task_def = DummyTaskDef()

    # 4) Run evaluation
    metrics = evaluate_candidate(program_module, task_def)

    # 5) Print results
    print("\nEvaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 6) Basic sanity checks
    assert "total_mse" in metrics, "total_mse is missing from metrics"
    assert "neg_total_mse" in metrics, "neg_total_mse is missing from metrics"
    assert not math.isnan(metrics["total_mse"]), "total_mse is NaN"

    print("\nBasic sanity checks passed.")


if __name__ == "__main__":
    main()
