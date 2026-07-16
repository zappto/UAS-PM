"""Safe dataset saving utility.

Provides a safe way to save DataFrames to CSV with overwrite
protection and automatic directory creation.
"""

import os

import pandas as pd


def save_dataset(
    df: pd.DataFrame,
    filepath: str,
    overwrite: bool = False,
) -> str:
    """Save a DataFrame to CSV with safety checks.

    Creates parent directories automatically. Prevents accidental
    overwriting unless explicitly requested.

    Args:
        df: DataFrame to save.
        filepath: Destination CSV file path.
        overwrite: If True, overwrite existing file. If False
            and the file exists, raises FileExistsError.

    Returns:
        The filepath where the dataset was saved.

    Raises:
        FileExistsError: If file exists and overwrite is False.
        ValueError: If DataFrame is empty.
    """
    if len(df) == 0:
        raise ValueError("Cannot save empty DataFrame.")

    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(
            f"File already exists: {filepath}. "
            f"Set overwrite=True to replace it."
        )

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False, encoding="utf-8")

    print(f"✓ Dataset saved: {filepath} ({len(df):,} rows)")
    return filepath
