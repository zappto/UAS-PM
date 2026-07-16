"""Input validation for the model evaluation stage.

Validates models, test features, labels, and predictions.
"""

from scipy.sparse import issparse


def validate_evaluation_inputs(model, X_test, y_test) -> None:
    """Validate model and test dataset inputs.

    Args:
        model: Trained model (sklearn estimator).
        X_test: Testing feature matrix.
        y_test: Testing labels.

    Raises:
        ValueError: If validation checks fail.
    """
    errors = []

    # Model validation
    if not hasattr(model, "predict"):
        errors.append("Model does not have a predict() method.")

    # Features validation
    if not issparse(X_test):
        # We expect sparse matrices for text data (TF-IDF)
        pass

    if X_test.shape[0] == 0:
        errors.append("X_test is empty.")

    # Labels validation
    if len(y_test) == 0:
        errors.append("y_test is empty.")

    if len(y_test) != X_test.shape[0]:
        errors.append(
            f"Dimension mismatch: X_test has {X_test.shape[0]} samples, "
            f"but y_test has {len(y_test)} labels."
        )

    if hasattr(y_test, "isnull") and y_test.isnull().any():
        errors.append("y_test contains missing values.")

    if errors:
        raise ValueError(f"Evaluation input validation failed: {'; '.join(errors)}")


def validate_predictions(y_test, y_pred) -> None:
    """Validate shape of generated predictions.

    Args:
        y_test: True testing labels.
        y_pred: Predicted labels.

    Raises:
        ValueError: If shapes mismatch.
    """
    if len(y_test) != len(y_pred):
        raise ValueError(
            f"Prediction mismatch: {len(y_test)} true labels vs "
            f"{len(y_pred)} predicted labels."
        )
