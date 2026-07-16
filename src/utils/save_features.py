"""Safe artifact saving utilities for feature engineering.

Provides overwrite-protected saving for joblib objects,
scipy sparse matrices, and pandas Series/DataFrames.
"""

import os

import joblib
import pandas as pd
from scipy import sparse


def save_artifact(
    obj: object,
    filepath: str,
    overwrite: bool = False,
) -> str:
    """Save any Python object using joblib with overwrite protection.

    Args:
        obj: Object to save (model, vectorizer, etc.).
        filepath: Destination file path.
        overwrite: Allow overwriting existing files.

    Returns:
        The filepath where the artifact was saved.

    Raises:
        FileExistsError: If file exists and overwrite is False.
    """
    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(
            f"File already exists: {filepath}. "
            f"Set overwrite=True to replace it."
        )

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath)
    print(f"✓ Artifact saved: {filepath}")
    return filepath


def save_sparse_matrix(
    matrix,
    filepath: str,
    overwrite: bool = False,
) -> str:
    """Save scipy sparse matrix as .npz with overwrite protection.

    Args:
        matrix: Scipy sparse matrix to save.
        filepath: Destination .npz file path.
        overwrite: Allow overwriting existing files.

    Returns:
        The filepath where the matrix was saved.

    Raises:
        FileExistsError: If file exists and overwrite is False.
    """
    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(
            f"File already exists: {filepath}. "
            f"Set overwrite=True to replace it."
        )

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sparse.save_npz(filepath, matrix)
    print(f"✓ Sparse matrix saved: {filepath} (shape={matrix.shape})")
    return filepath


def save_labels(
    series: pd.Series,
    filepath: str,
    overwrite: bool = False,
) -> str:
    """Save labels Series as CSV with overwrite protection.

    Args:
        series: Pandas Series to save.
        filepath: Destination CSV file path.
        overwrite: Allow overwriting existing files.

    Returns:
        The filepath where the labels were saved.

    Raises:
        FileExistsError: If file exists and overwrite is False.
    """
    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(
            f"File already exists: {filepath}. "
            f"Set overwrite=True to replace it."
        )

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    series.to_csv(filepath, index=False, header=True)
    print(f"✓ Labels saved: {filepath} ({len(series):,} rows)")
    return filepath
