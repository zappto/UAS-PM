"""Local explanation module.

Provides functionalities to extract feature contributions for
a single prediction instance.
"""

from typing import Any, Dict, List, Tuple
import numpy as np


class LocalExplanation:
    """Extracts local feature importance for a single instance."""

    def __init__(self, feature_names: np.ndarray):
        """Initialize with feature names.

        Args:
            feature_names: Array of feature names from vectorizer.
        """
        self.feature_names = feature_names

    def get_explanation(
        self,
        instance_shap_values: np.ndarray,
        predicted_class_index: int,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Extract top positive and negative features for the predicted class.

        Args:
            instance_shap_values: SHAP values for the instance. 
                                  Shape: (num_features, num_classes) or (num_features,)
            predicted_class_index: Index of the predicted class.
            top_k: Number of top features to return.

        Returns:
            Dictionary containing positive and negative feature contributions.
        """
        # Determine if binary or multiclass
        if instance_shap_values.ndim == 2:
            class_shap_values = instance_shap_values[:, predicted_class_index]
        else:
            class_shap_values = instance_shap_values

        # Sort features by contribution
        sorted_indices = np.argsort(class_shap_values)
        
        # Top negative contributions (smallest values)
        neg_indices = sorted_indices[:top_k]
        neg_features = [
            {"feature": self.feature_names[i], "score": float(class_shap_values[i])}
            for i in neg_indices if class_shap_values[i] < 0
        ]
        
        # Top positive contributions (largest values)
        pos_indices = sorted_indices[-top_k:][::-1]
        pos_features = [
            {"feature": self.feature_names[i], "score": float(class_shap_values[i])}
            for i in pos_indices if class_shap_values[i] > 0
        ]

        return {
            "predicted_class_index": predicted_class_index,
            "positive_features": pos_features,
            "negative_features": neg_features
        }
