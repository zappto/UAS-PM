"""Inference module for explainability.

Provides a single reusable interface for generating predictions 
and explanations for downstream applications.
"""

import time
from typing import Dict, Any, List
import joblib

from src.config.settings import TFIDF_VECTORIZER_PATH, LR_BEST_MODEL_PATH
from src.explainability.explainer_factory import ExplainerFactory
from src.explainability.local_explanation import LocalExplanation
from src.explainability.text_highlighter import TextHighlighter
from src.explainability.prediction_schema import (
    PredictionResult,
    ProbabilityItem,
    WordImportance
)


class InferencePipeline:
    """End-to-end pipeline for prediction and explainability."""

    def __init__(
        self,
        model_path: str = LR_BEST_MODEL_PATH,
        tfidf_path: str = TFIDF_VECTORIZER_PATH,
        top_k_words: int = 10
    ):
        """Initialize the pipeline and load artifacts.

        Args:
            model_path: Path to the trained ML model.
            tfidf_path: Path to the fitted TF-IDF vectorizer.
            top_k_words: Number of top words to extract for local explanations.
        """
        self.model = joblib.load(model_path)
        self.tfidf = joblib.load(tfidf_path)
        self.feature_names = self.tfidf.get_feature_names_out()
        
        self.explainer = ExplainerFactory.get_explainer(self.model)
        self.local_exp = LocalExplanation(self.feature_names)
        self.highlighter = TextHighlighter(self.feature_names)
        self.top_k_words = top_k_words
        
        # Get class names from model if available
        self.class_names = list(self.model.classes_) if hasattr(self.model, "classes_") else []
        self.model_name = type(self.model).__name__

    def predict_with_explanation(self, text: str) -> Dict[str, Any]:
        """Run prediction and explainability on a single text.

        Args:
            text: The input text to predict.

        Returns:
            Dict containing the serialized PredictionResult.
        """
        start_time = time.time()

        # 1. Transform text
        x_transformed = self.tfidf.transform([text])

        # 2. Predict
        prediction = self.model.predict(x_transformed)[0]
        
        # Safely get predicted index
        try:
            pred_idx = self.class_names.index(prediction)
        except ValueError:
            pred_idx = 0

        # 3. Probabilities
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x_transformed)[0]
            confidence = float(proba[pred_idx])
            probabilities = [
                ProbabilityItem(class_name=str(cls), probability=float(p))
                for cls, p in zip(self.class_names, proba)
            ]
        else:
            confidence = 1.0
            probabilities = [ProbabilityItem(class_name=str(prediction), probability=1.0)]

        # 4. Generate SHAP/Explanation values
        # Explain instance returns (1, num_features, num_classes) or (1, num_features)
        shap_values = self.explainer.explain_instance(x_transformed)
        instance_shap = shap_values[0]

        # 5. Extract top positive and negative words
        explanation_dict = self.local_exp.get_explanation(
            instance_shap_values=instance_shap,
            predicted_class_index=pred_idx,
            top_k=self.top_k_words
        )

        top_positive_words = [
            WordImportance(
                word=item["feature"],
                importance=item["score"],
                direction="positive"
            )
            for item in explanation_dict.get("positive_features", [])
        ]

        top_negative_words = [
            WordImportance(
                word=item["feature"],
                importance=item["score"],
                direction="negative"
            )
            for item in explanation_dict.get("negative_features", [])
        ]

        # 6. Extract highlighted tokens
        highlighted_tokens = self.highlighter.get_highlight_data(
            raw_text=text,
            instance_shap_values=instance_shap,
            predicted_class_index=pred_idx
        )

        inference_time = time.time() - start_time

        # 7. Construct result
        result = PredictionResult(
            prediction=str(prediction),
            confidence=confidence,
            probabilities=probabilities,
            top_positive_words=top_positive_words,
            top_negative_words=top_negative_words,
            highlighted_tokens=highlighted_tokens,
            model_name=self.model_name,
            inference_time=inference_time
        )

        return result.to_dict()


# For convenience, expose a global default instance interface if needed.
# But for Streamlit, they will initialize InferencePipeline themselves to leverage caching.
