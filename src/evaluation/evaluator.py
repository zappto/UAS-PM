"""Evaluation orchestrator for a single model.

Brings together metrics, classification reports, confusion matrices,
and ROC curves into a single comprehensive evaluation pass.
"""

import time
import pandas as pd

from src.config.evaluation_config import EvaluationConfig
from src.evaluation.classification_report import generate_classification_report
from src.evaluation.confusion_matrix import generate_confusion_matrix
from src.evaluation.metrics import calculate_metrics, calculate_per_class_metrics
from src.evaluation.roc_curve import calculate_roc_auc
from src.evaluation.validator import validate_evaluation_inputs, validate_predictions


def evaluate_model(
    model,
    model_name: str,
    X_test,
    y_test: pd.Series,
    config: EvaluationConfig | None = None,
) -> dict:
    """Run full evaluation pipeline on a single model.

    Args:
        model: Trained scikit-learn estimator.
        model_name: Human-readable name for the model.
        X_test: Testing feature matrix.
        y_test: True testing labels.
        config: Evaluation configuration.

    Returns:
        Comprehensive dictionary containing all evaluation results.
    """
    if config is None:
        config = EvaluationConfig()

    print(f"Evaluating {model_name}...")
    validate_evaluation_inputs(model, X_test, y_test)

    start_time = time.time()

    # 1. Generate predictions
    print(f"  Generating predictions...")
    y_pred = model.predict(X_test)
    validate_predictions(y_test, y_pred)

    # 2. Extract class labels in the order the model knows them
    # This is critical for matching probabilities to string labels
    model_classes = list(model.classes_) if hasattr(model, "classes_") else config.class_labels

    # 3. Overall and Per-Class Metrics
    print(f"  Calculating metrics...")
    overall_metrics = calculate_metrics(y_test, y_pred, config)
    per_class_metrics = calculate_per_class_metrics(y_test, y_pred, model_classes)

    # 4. Classification Report
    class_report = generate_classification_report(y_test, y_pred, model_classes)

    # 5. Confusion Matrix
    cm_array, cm_df = generate_confusion_matrix(y_test, y_pred, model_classes)

    # 6. ROC & AUC (requires predict_proba)
    roc_data = None
    if hasattr(model, "predict_proba"):
        print(f"  Calculating ROC curves (OvR)...")
        try:
            y_prob = model.predict_proba(X_test)
            roc_data = calculate_roc_auc(y_test, y_prob, model_classes)
            overall_metrics["roc_auc_macro"] = roc_data["macro"]["auc"]
        except Exception as e:
            print(f"  ⚠️ Failed to calculate ROC/AUC: {e}")
    else:
        print(f"  ⚠️ {model_name} does not support predict_proba. Skipping ROC.")

    duration = time.time() - start_time
    print(f"  ✓ Evaluation complete ({duration:.2f}s)")
    print(f"    F1 (Macro): {overall_metrics['f1_macro']:.4f}")
    print(f"    Accuracy:   {overall_metrics['accuracy']:.4f}")

    return {
        "model_name": model_name,
        "overall_metrics": overall_metrics,
        "per_class_metrics": per_class_metrics,
        "classification_report": class_report,
        "confusion_matrix_array": cm_array,
        "confusion_matrix_df": cm_df,
        "roc_data": roc_data,
        "labels": model_classes,
        "duration": duration,
    }
