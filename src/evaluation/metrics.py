"""Metric calculation for model evaluation.

Calculates Accuracy, Precision, Recall, and F1 Score
overall (macro/weighted averages) and per-class.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.config.evaluation_config import EvaluationConfig


def calculate_metrics(
    y_true,
    y_pred,
    config: EvaluationConfig | None = None,
) -> dict:
    """Calculate overall evaluation metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        config: Evaluation configuration (for average setting).

    Returns:
        Dictionary of calculated metrics.
    """
    if config is None:
        config = EvaluationConfig()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def calculate_per_class_metrics(
    y_true,
    y_pred,
    labels: list[str] | None = None,
) -> dict:
    """Calculate evaluation metrics for each individual class.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: List of class labels to evaluate.

    Returns:
        Dictionary mapping class names to their metrics.
    """
    per_class_precision = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    # If labels is None, infer from unique values
    class_labels = labels if labels is not None else sorted(list(set(y_true)))

    results = {}
    for i, label in enumerate(class_labels):
        results[label] = {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i]),
        }

    return results
