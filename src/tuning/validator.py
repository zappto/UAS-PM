"""Input validation for hyperparameter tuning.

Validates feature matrices, labels, and parameter grids
before running GridSearchCV to catch issues early.
"""

import numpy as np
from scipy.sparse import issparse


def validate_tuning_inputs(
    X_train,
    y_train,
    param_grid: dict,
) -> dict:
    """Validate features, labels, and parameter grid.

    Performs comprehensive checks before running GridSearchCV
    to provide informative error messages.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        param_grid: Parameter grid dictionary.

    Returns:
        Dictionary with validation results:
            - is_valid (bool): All checks passed.
            - checks (dict): Individual check results.
            - errors (list[str]): Error messages.

    Raises:
        ValueError: If critical validation fails.
    """
    errors: list[str] = []
    checks: dict[str, bool] = {}

    # 1. Feature matrix is not empty
    checks["features_not_empty"] = X_train.shape[0] > 0 and X_train.shape[1] > 0
    if not checks["features_not_empty"]:
        errors.append(f"X_train is empty: shape={X_train.shape}")

    # 2. Feature matrix is sparse
    checks["features_sparse"] = issparse(X_train)
    if not checks["features_sparse"]:
        # Warning, not error — GridSearchCV works with dense too
        pass

    # 3. Labels are not empty
    checks["labels_not_empty"] = len(y_train) > 0
    if not checks["labels_not_empty"]:
        errors.append("y_train is empty.")

    # 4. Labels match features
    checks["labels_match_features"] = len(y_train) == X_train.shape[0]
    if not checks["labels_match_features"]:
        errors.append(
            f"Label/feature mismatch: "
            f"y_train={len(y_train)}, X_train={X_train.shape[0]}"
        )

    # 5. No missing labels
    has_null = hasattr(y_train, "isnull") and y_train.isnull().any()
    checks["no_missing_labels"] = not has_null
    if has_null:
        errors.append("y_train contains missing values.")

    # 6. Parameter grid is not empty
    checks["param_grid_not_empty"] = len(param_grid) > 0
    if not checks["param_grid_not_empty"]:
        errors.append("Parameter grid is empty.")

    # 7. Parameter grid values are lists
    all_lists = all(isinstance(v, list) for v in param_grid.values())
    checks["param_values_are_lists"] = all_lists
    if not all_lists:
        bad = [k for k, v in param_grid.items() if not isinstance(v, list)]
        errors.append(f"Parameter values must be lists: {bad}")

    # 8. Parameter grid values are non-empty
    all_nonempty = all(len(v) > 0 for v in param_grid.values() if isinstance(v, list))
    checks["param_values_nonempty"] = all_nonempty
    if not all_nonempty:
        empty = [k for k, v in param_grid.items()
                 if isinstance(v, list) and len(v) == 0]
        errors.append(f"Empty parameter lists: {empty}")

    is_valid = len(errors) == 0

    if not is_valid:
        error_msg = "; ".join(errors)
        raise ValueError(f"Tuning input validation failed: {error_msg}")

    return {
        "is_valid": is_valid,
        "checks": checks,
        "errors": errors,
    }
