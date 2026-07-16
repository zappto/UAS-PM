"""Stratified train/test split for feature engineering.

Splits the dataset following TRD/DRD requirements:
- Train: 80%, Test: 20%
- Stratified by label
- Random state: 42
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.tfidf_config import TfidfConfig


def split_dataset(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    label_col: str = "label",
    config: TfidfConfig | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split dataset into train and test sets with stratification.

    Uses the preprocessed text column as features and the label
    column as the target. Stratification maintains the original
    class distribution in both splits.

    Args:
        df: Input DataFrame with text and label columns.
        text_col: Name of the text column to use as X.
        label_col: Name of the label column to use as y.
        config: Configuration with split parameters.
            Uses defaults (80/20, random_state=42) if None.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) where
        X values are pd.Series of text strings and
        y values are pd.Series of label strings.

    Raises:
        ValueError: If required columns are missing.
        ValueError: If dataset is empty.
    """
    if config is None:
        config = TfidfConfig()

    # Validate columns
    for col in [text_col, label_col]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

    if len(df) == 0:
        raise ValueError("Cannot split an empty DataFrame.")

    X = df[text_col].fillna("")
    y = df[label_col]

    split_params = config.get_split_params()
    stratify_col = y if config.stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=stratify_col,
        **split_params,
    )

    # Reset indices
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print(f"✓ Dataset split: Train={len(X_train):,}, Test={len(X_test):,}")
    print(f"  Split ratio: {len(X_train)/len(df)*100:.1f}% / "
          f"{len(X_test)/len(df)*100:.1f}%")

    return X_train, X_test, y_train, y_test
