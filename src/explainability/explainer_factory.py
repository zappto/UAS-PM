"""Explainer factory module.

Provides a factory to automatically select the appropriate explainer
based on the provided machine learning model type.
"""

from typing import Any
from src.explainability.shap_explainer import ShapExplainer
from src.explainability.fallback_explainer import FallbackExplainer


class ExplainerFactory:
    """Factory to instantiate the correct explainer."""

    @staticmethod
    def get_explainer(model: Any, x_train: Any = None) -> Any:
        """Get the appropriate explainer for the given model.

        Args:
            model: The trained ML model.
            x_train: Background dataset for SHAP (optional).

        Returns:
            An instantiated explainer (ShapExplainer or FallbackExplainer).
        """
        model_type = type(model).__name__

        if model_type in ["LogisticRegression", "LinearSVC", "SVC"]:
            return ShapExplainer(model, x_train)
        elif model_type in ["MultinomialNB"]:
            return FallbackExplainer(model)
        else:
            raise ValueError(f"No suitable explainer found for model type: {model_type}")
