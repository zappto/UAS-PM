"""Dataset loading and schema validation utilities."""

import os

import pandas as pd

from src.config.settings import DATASET_PATH, REQUIRED_COLUMNS


def load_dataset(filepath: str | None = None) -> pd.DataFrame:
    """Load the dataset from CSV and perform basic validation.

    Args:
        filepath: Path to the CSV file. Defaults to the project's
            processed dataset path defined in settings.

    Returns:
        Validated pandas DataFrame with 'text' and 'label' columns.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If schema validation fails.
    """
    path = filepath or DATASET_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    validation = validate_schema(df)
    if not validation["is_valid"]:
        raise ValueError(
            f"Schema validation failed: {validation['errors']}"
        )

    print(f"✓ Dataset loaded successfully: {len(df):,} rows, "
          f"{len(df.columns)} columns")
    return df


def validate_schema(df: pd.DataFrame) -> dict[str, bool | list[str]]:
    """Validate that the DataFrame has the required schema.

    Args:
        df: DataFrame to validate.

    Returns:
        Dictionary with validation results:
            - is_valid (bool): Whether all checks passed.
            - has_required_columns (bool): Required columns present.
            - errors (list[str]): List of validation error messages.
    """
    errors: list[str] = []

    # Check required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    has_required = len(missing_cols) == 0
    if not has_required:
        errors.append(f"Missing required columns: {missing_cols}")

    # Check DataFrame is not empty
    if len(df) == 0:
        errors.append("Dataset is empty (0 rows)")

    return {
        "is_valid": len(errors) == 0,
        "has_required_columns": has_required,
        "errors": errors,
    }
