"""Input validation for the error analysis stage.

Validates prediction files, test features, labels, and datasets.
"""

import os
import pandas as pd
from typing import Any


def validate_prediction_file(filepath: str) -> None:
    """Validate that a prediction file exists and is not empty.

    Args:
        filepath: Path to the prediction CSV file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or invalid.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prediction file missing: {filepath}")
    
    if os.path.getsize(filepath) == 0:
        raise ValueError(f"Prediction file is empty: {filepath}")


def validate_dataset_alignment(
    df: pd.DataFrame,
    y_test: pd.Series | pd.DataFrame,
    predictions: dict[str, pd.Series],
) -> None:
    """Validate that the original text, true labels, and predictions align.

    Args:
        df: DataFrame containing the original text.
        y_test: True test labels.
        predictions: Dictionary of model predictions.

    Raises:
        ValueError: If shape mismatch occurs.
    """
    n_samples = len(y_test)
    
    # Check predictions
    for model_name, pred_series in predictions.items():
        if len(pred_series) != n_samples:
            raise ValueError(
                f"Shape mismatch: {model_name} predictions ({len(pred_series)}) "
                f"do not match true labels ({n_samples})."
            )
            
    # Note: 'df' might be the full dataset, so we just ensure it's not empty
    # The actual alignment relies on indices or sequential extraction in error_loader
    if len(df) == 0:
        raise ValueError("Original text dataset is empty.")


def validate_labels(y_true: Any, valid_labels: set[str]) -> None:
    """Validate that true labels only contain expected values.
    
    Args:
        y_true: True test labels.
        valid_labels: Set of expected class names.
        
    Raises:
        ValueError: If unexpected labels are found.
    """
    unique_labels = set(y_true.unique())
    unknown = unique_labels - valid_labels
    
    if unknown:
        raise ValueError(f"Found invalid labels in dataset: {unknown}")
