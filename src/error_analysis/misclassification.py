"""Misclassification identification.

Splits data into correct and incorrect predictions,
and calculates basic error distribution.
"""

import pandas as pd


def separate_errors(
    df: pd.DataFrame,
    model_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate dataset into correct and incorrect predictions.

    Args:
        df: DataFrame containing 'true_label' and the model's prediction column.
        model_col: Name of the column containing model predictions.

    Returns:
        Tuple of (correct_df, incorrect_df).
    """
    correct_mask = df["true_label"] == df[model_col]
    
    correct_df = df[correct_mask].copy()
    incorrect_df = df[~correct_mask].copy()
    
    return correct_df, incorrect_df


def analyze_misclassifications(df: pd.DataFrame) -> dict:
    """Analyze basic error statistics for all models in the DataFrame.

    Args:
        df: Main error analysis DataFrame.

    Returns:
        Dictionary containing error counts and DataFrames for each model.
    """
    models = [c for c in df.columns if c.startswith("pred_")]
    results = {}
    
    for m in models:
        model_name = m.replace("pred_", "")
        correct, incorrect = separate_errors(df, m)
        
        results[model_name] = {
            "correct_count": len(correct),
            "error_count": len(incorrect),
            "error_rate": len(incorrect) / len(df),
            "misclassified_df": incorrect,
            "correct_df": correct
        }
        
    return results
