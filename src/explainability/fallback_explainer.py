"""Fallback explainer module for models not supported by SHAP easily.

Specifically designed for Multinomial Naive Bayes using feature_log_prob_.
"""

from typing import Any
import numpy as np

from src.explainability.validator import validate_model


class FallbackExplainer:
    """Explainer for models using native properties (e.g., Naive Bayes)."""

    def __init__(self, model: Any):
        """Initialize the fallback explainer.

        Args:
            model: Trained ML model (e.g., MultinomialNB).
        """
        validate_model(model)
        self.model = model

    def get_feature_contributions(self, x_data: Any) -> np.ndarray:
        """Calculate feature contributions as pseudo-SHAP values.

        For MultinomialNB, we use the feature log probabilities multiplied by
        the input term frequencies to approximate feature importance per instance.

        Args:
            x_data: Input features (sparse matrix or numpy array).

        Returns:
            np.ndarray: Pseudo-SHAP values of shape (num_samples, num_features, num_classes).
        """
        if not hasattr(self.model, "feature_log_prob_"):
            raise ValueError("Model does not have 'feature_log_prob_' attribute.")

        # feature_log_prob_ shape is (n_classes, n_features)
        # We want to map this to (n_samples, n_features, n_classes)
        log_prob = self.model.feature_log_prob_
        n_classes, n_features = log_prob.shape
        n_samples = x_data.shape[0]

        # Ensure x_data is a dense array for easier broadcasting (or process efficiently)
        # Assuming x_data can fit in memory as dense, else process in batches.
        # But for this scope, let's convert to dense or use sparse matrix multiplication.
        
        # Contribution per feature = x_data[sample, feature] * log_prob[class, feature]
        # x_data is (n_samples, n_features)
        # We need (n_samples, n_features, n_classes)
        
        # Using sparse operations if x_data is sparse
        contributions = np.zeros((n_samples, n_features, n_classes))
        
        # This can be memory intensive for very large datasets,
        # but typical TF-IDF + classical ML on small text dataset is fine.
        x_dense = x_data.toarray() if hasattr(x_data, "toarray") else x_data
        
        for c in range(n_classes):
            # Element-wise multiplication
            contributions[:, :, c] = x_dense * log_prob[c, :]
            
        return contributions

    def get_shap_values(self, x_data: Any) -> np.ndarray:
        """Alias for compatibility with ShapExplainer."""
        return self.get_feature_contributions(x_data)

    def explain_instance(self, x_instance: Any) -> np.ndarray:
        """Get contributions for a single instance.
        
        Args:
            x_instance: A single input sample (1, num_features).
            
        Returns:
            np.ndarray: Contributions for the instance.
        """
        return self.get_feature_contributions(x_instance)
