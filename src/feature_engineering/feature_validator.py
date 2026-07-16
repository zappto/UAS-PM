"""Feature validation for TF-IDF outputs.

Validates feature matrices, vocabulary, labels, and
consistency between training and testing sets to catch
issues before model training.
"""

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer


def validate_features(
    X_train,
    X_test,
    y_train: pd.Series,
    y_test: pd.Series,
    vectorizer: TfidfVectorizer,
) -> dict:
    """Validate TF-IDF feature matrices and labels.

    Performs comprehensive validation checks to ensure the
    feature engineering output is ready for model training.

    Args:
        X_train: Training feature matrix (sparse).
        X_test: Testing feature matrix (sparse).
        y_train: Training labels.
        y_test: Testing labels.
        vectorizer: Fitted TF-IDF vectorizer.

    Returns:
        Dictionary with validation results:
            - is_valid (bool): All checks passed.
            - checks (dict): Individual check results.
            - errors (list[str]): Error messages for failed checks.
            - summary (str): Human-readable summary.
    """
    errors: list[str] = []
    checks: dict[str, bool] = {}

    # 1. Vocabulary is not empty
    vocab_size = len(vectorizer.vocabulary_)
    checks["vocabulary_not_empty"] = vocab_size > 0
    if not checks["vocabulary_not_empty"]:
        errors.append(f"Vocabulary is empty (size=0)")

    # 2. Feature matrices are not empty
    checks["X_train_not_empty"] = X_train.shape[0] > 0 and X_train.shape[1] > 0
    checks["X_test_not_empty"] = X_test.shape[0] > 0 and X_test.shape[1] > 0
    if not checks["X_train_not_empty"]:
        errors.append(f"X_train is empty: shape={X_train.shape}")
    if not checks["X_test_not_empty"]:
        errors.append(f"X_test is empty: shape={X_test.shape}")

    # 3. Feature dimensions match
    checks["feature_dims_match"] = X_train.shape[1] == X_test.shape[1]
    if not checks["feature_dims_match"]:
        errors.append(
            f"Feature dimensions mismatch: "
            f"X_train={X_train.shape[1]}, X_test={X_test.shape[1]}"
        )

    # 4. Label counts match sample counts
    checks["train_labels_match"] = len(y_train) == X_train.shape[0]
    checks["test_labels_match"] = len(y_test) == X_test.shape[0]
    if not checks["train_labels_match"]:
        errors.append(
            f"Train label/feature mismatch: "
            f"y_train={len(y_train)}, X_train={X_train.shape[0]}"
        )
    if not checks["test_labels_match"]:
        errors.append(
            f"Test label/feature mismatch: "
            f"y_test={len(y_test)}, X_test={X_test.shape[0]}"
        )

    # 5. Label sets are consistent
    train_labels = set(y_train.unique())
    test_labels = set(y_test.unique())
    checks["label_sets_consistent"] = test_labels.issubset(train_labels)
    if not checks["label_sets_consistent"]:
        missing = test_labels - train_labels
        errors.append(f"Test set has labels not in train set: {missing}")

    # 6. No NaN/Inf in matrices
    if issparse(X_train):
        has_nan_train = np.any(np.isnan(X_train.data)) if len(X_train.data) > 0 else False
        has_inf_train = np.any(np.isinf(X_train.data)) if len(X_train.data) > 0 else False
    else:
        has_nan_train = np.any(np.isnan(X_train))
        has_inf_train = np.any(np.isinf(X_train))
    checks["no_nan_values"] = not has_nan_train
    checks["no_inf_values"] = not has_inf_train
    if has_nan_train:
        errors.append("X_train contains NaN values")
    if has_inf_train:
        errors.append("X_train contains Inf values")

    # 7. No missing labels
    checks["no_missing_labels"] = (
        y_train.isnull().sum() == 0 and y_test.isnull().sum() == 0
    )
    if not checks["no_missing_labels"]:
        errors.append(
            f"Missing labels: y_train={y_train.isnull().sum()}, "
            f"y_test={y_test.isnull().sum()}"
        )

    is_valid = len(errors) == 0

    # Build summary
    lines = ["Feature Validation Report"]
    lines.append("-" * 50)
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        lines.append(f"  {status} {check_name}")
    lines.append("-" * 50)
    lines.append(f"  Status: {'PASS' if is_valid else 'FAILED'}")
    if errors:
        lines.append(f"  Errors: {len(errors)}")
        for err in errors:
            lines.append(f"    - {err}")

    return {
        "is_valid": is_valid,
        "checks": checks,
        "errors": errors,
        "summary": "\n".join(lines),
    }
