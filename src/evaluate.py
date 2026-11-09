"""Shared evaluation and result-saving helpers."""
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config import RESULTS_DIR


def evaluate(model_name: str, y_true, y_pred, split: str = "test") -> dict:
    """Compute the standard metrics for one model on one split."""
    return {
        "model": model_name,
        "split": split,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def save_results(results: list[dict], member: str) -> pd.DataFrame:
    """Save a list of evaluate(...) dicts to results/metrics_<member>.csv."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / f"metrics_{member}.csv", index=False)
    return df
