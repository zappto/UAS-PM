"""SHAP explainer module for linear models.

This module provides explanation capabilities for Logistic Regression
and Linear SVM using SHAP LinearExplainer.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import shap

from src.explainability.validator import validate_model


class ShapExplainer:
    """Wrapper for SHAP LinearExplainer."""

    def __init__(self, model: Any, x_train: Any = None):
        """Initialize the SHAP explainer.

        Args:
            model: Trained LogisticRegression or LinearSVC model.
            x_train: Training data used as background (optional for LinearExplainer).
        """
        validate_model(model)
        self.model = model
        
        # LinearExplainer works well for LogisticRegression and LinearSVC.
        # Background dataset is optional for LinearExplainer but can improve results.
        self.explainer = shap.LinearExplainer(self.model, x_train) if x_train is not None else shap.LinearExplainer(self.model, shap.maskers.Independent(np.zeros((1, model.coef_.shape[1]))))

    def get_shap_values(self, x_data: Any) -> np.ndarray:
        """Calculate SHAP values for the given data.

        Args:
            x_data: Input features (sparse matrix or numpy array).

        Returns:
            np.ndarray: SHAP values of shape (num_samples, num_features, num_classes)
                        or (num_samples, num_features) for binary.
        """
        # LinearExplainer returns a list of arrays for multiclass or a single array.
        shap_values = self.explainer.shap_values(x_data)
        
        # Convert list of arrays (multiclass) to a single 3D array: (samples, features, classes)
        if isinstance(shap_values, list):
            shap_values = np.stack(shap_values, axis=-1)
            
        return shap_values

    def explain_instance(self, x_instance: Any) -> np.ndarray:
        """Get SHAP values for a single instance.
        
        Args:
            x_instance: A single input sample (1, num_features).
            
        Returns:
            np.ndarray: SHAP values for the instance.
        """
        return self.get_shap_values(x_instance)
