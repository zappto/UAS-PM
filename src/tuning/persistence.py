"""Tuning artifact persistence — save best models, params, CV results.

Supports joblib (models), JSON (parameters), and CSV (results).
"""

import json
import os

import joblib
import numpy as np
import pandas as pd


def save_best_model(
    model,
    filepath: str,
    overwrite: bool = False,
) -> str:
    """Save best tuned model using joblib.

    Args:
        model: Fitted best estimator from GridSearchCV.
        filepath: Destination .joblib file path.
        overwrite: Allow overwriting existing files.

    Returns:
        The filepath where the model was saved.

    Raises:
        FileExistsError: If file exists and overwrite is False.
    """
    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(
            f"Model file already exists: {filepath}. "
            f"Set overwrite=True to replace it."
        )

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

    file_size = os.path.getsize(filepath)
    size_str = _format_size(file_size)
    print(f"  ✓ Best model saved: {filepath} ({size_str})")

    return filepath


def save_best_parameters(
    all_best_params: dict[str, dict],
    filepath: str,
    overwrite: bool = False,
) -> str:
    """Save best parameters for all models as JSON.

    Args:
        all_best_params: Dictionary mapping model names to
            their best parameter dictionaries.
        filepath: Destination .json file path.
        overwrite: Allow overwriting existing files.

    Returns:
        The filepath where the parameters were saved.

    Raises:
        FileExistsError: If file exists and overwrite is False.
    """
    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(
            f"Parameters file already exists: {filepath}. "
            f"Set overwrite=True to replace it."
        )

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convert numpy types to Python native for JSON serialization
    serializable = {}
    for model_name, params in all_best_params.items():
        serializable[model_name] = {
            k: _to_native(v) for k, v in params.items()
        }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Best parameters saved: {filepath}")
    return filepath


def save_cv_results(
    results: list[dict],
    filepath: str,
    overwrite: bool = False,
) -> str:
    """Save combined CV results for all models as CSV.

    Args:
        results: List of tuning result dictionaries.
        filepath: Destination .csv file path.
        overwrite: Allow overwriting existing files.

    Returns:
        The filepath where the results were saved.

    Raises:
        FileExistsError: If file exists and overwrite is False.
    """
    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(
            f"CV results file already exists: {filepath}. "
            f"Set overwrite=True to replace it."
        )

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    all_rows = []
    for result in results:
        cv_res = result["cv_results"]
        df = pd.DataFrame(cv_res)
        df.insert(0, "model", result["model_name"])
        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(filepath, index=False)

    print(f"  ✓ CV results saved: {filepath} ({len(combined)} rows)")
    return filepath


def _to_native(value):
    """Convert numpy types to Python native types for JSON.

    Args:
        value: Value to convert.

    Returns:
        Python native type.
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    elif isinstance(value, (np.floating,)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: File size in bytes.

    Returns:
        Formatted string (e.g., '1.2 MB').
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
