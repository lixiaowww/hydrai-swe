from neuralhydrology.evaluation import get_tester
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import compute_all

def _extract_metrics_from_results(results) -> dict:
    """
    Best-effort extraction of regression metrics from a tester.evaluate() result.
    Returns an empty dict if nothing usable found.
    """
    try:
        if isinstance(results, dict):
            for _k, v in results.items():
                if hasattr(v, 'index') and hasattr(v, 'columns'):
                    pred_col = next((c for c in v.columns if str(c).lower() in ("qsim", "prediction", "pred")), None)
                    obs_col = next((c for c in v.columns if str(c).lower() in ("qobs", "observed", "obs")), None)
                    if pred_col and obs_col:
                        df = v[[obs_col, pred_col]].dropna()
                        y_true = df[obs_col].to_numpy(dtype=float)
                        y_pred = df[pred_col].to_numpy(dtype=float)
                        return compute_all(y_true, y_pred)
    except Exception:
        pass
    return {}


def evaluate_run_dir(run_dir: str) -> dict:
    tester = get_tester(run_dir=Path(run_dir), period="test")
    results = tester.evaluate()
    metrics = _extract_metrics_from_results(results)
    return metrics


def predict_with_neuralhydrology(run_dir):
    """
    Evaluates a trained NeuralHydrology model on the test set.

    Args:
        run_dir (str): The path to the NeuralHydrology run directory, 
                       which contains the trained model and config file.
    """
    print(f"Evaluating model from {run_dir}...")

    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        print(f"Error: Run directory not found at {run_dir}")
        print("Please train the model first. The run directory is created during training.")
        return

    # The get_tester function will load the model and data from the run directory
    # and evaluate the model on the test set specified in the config file.
    tester = get_tester(run_dir=run_dir, period="test")
    results = tester.evaluate()
    metrics = _extract_metrics_from_results(results)
    if metrics:
        print("Regression metrics:")
        for k, val in metrics.items():
            print(f"  {k}: {val:.4f}")
    
    print("\nPrediction finished.")
    print(f"You can find the detailed prediction results in the run directory: {run_dir}")


if __name__ == "__main__":
    # --- IMPORTANT ---
    # You need to replace this with the actual path to your run directory.
    # The run directory is created automatically by NeuralHydrology during training.
    # It will be something like: 'runs/hydrai_swe_experiment_xxxxxxxx_xxxxxx'
    # Please check the 'runs' directory after training to find the correct path.
    
    # As a placeholder, we can't run this directly without a trained model.
    # We will just print a message.
    print("To run the prediction, please train a model first and then update")
    print("the `run_dir_path` variable in this script with the correct path to your run directory.")
    
    # Example of how to run the prediction once you have a run directory:
    # run_dir_path = "runs/hydrai_swe_experiment_xxxxxxxx_xxxxxx" 
    # predict_with_neuralhydrology(run_dir=run_dir_path)