"""Safe feature loading utility for model training.

Provides a validated wrapper around the feature engineering
persistence module to load train/test features and labels.
"""

import pandas as pd
from scipy.sparse import issparse

from src.feature_engineering.persistence import load_tfidf_artifacts


def load_features() -> tuple:
    """Load train/test features and labels with validation.

    Wraps load_tfidf_artifacts() from the feature engineering
    stage and performs additional schema validation.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test):
            - X_train: Sparse training feature matrix.
            - X_test: Sparse testing feature matrix.
            - y_train: Training labels (pd.Series).
            - y_test: Testing labels (pd.Series).

    Raises:
        FileNotFoundError: If any feature file is missing.
        ValueError: If loaded data fails validation.
    """
    vectorizer, X_train, X_test, y_train, y_test = load_tfidf_artifacts()

    # Validate sparse matrices
    if not issparse(X_train):
        raise ValueError(
            f"X_train is not a sparse matrix: {type(X_train)}. "
            f"Expected scipy sparse matrix."
        )
    if not issparse(X_test):
        raise ValueError(
            f"X_test is not a sparse matrix: {type(X_test)}. "
            f"Expected scipy sparse matrix."
        )

    # Validate dimensions match
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: "
            f"X_train has {X_train.shape[1]} features, "
            f"X_test has {X_test.shape[1]} features."
        )

    # Validate label counts match sample counts
    if len(y_train) != X_train.shape[0]:
        raise ValueError(
            f"Train label/feature mismatch: "
            f"y_train={len(y_train)}, X_train={X_train.shape[0]}"
        )
    if len(y_test) != X_test.shape[0]:
        raise ValueError(
            f"Test label/feature mismatch: "
            f"y_test={len(y_test)}, X_test={X_test.shape[0]}"
        )

    # Validate no missing labels
    if y_train.isnull().any():
        raise ValueError(
            f"y_train contains {y_train.isnull().sum()} missing values."
        )
    if y_test.isnull().any():
        raise ValueError(
            f"y_test contains {y_test.isnull().sum()} missing values."
        )

    print(f"✓ Features validated: "
          f"X_train={X_train.shape}, X_test={X_test.shape}, "
          f"labels={y_train.nunique()} classes")

    return X_train, X_test, y_train, y_test
