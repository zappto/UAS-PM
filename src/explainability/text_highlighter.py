"""Text highlighter module.

Provides functionalities to map SHAP values to text tokens
for structured highlighting with positional information.
"""

import re
from typing import List
import numpy as np
from src.explainability.prediction_schema import HighlightedToken


class TextHighlighter:
    """Maps feature importance to text tokens."""

    def __init__(self, feature_names: np.ndarray):
        """Initialize with feature names.

        Args:
            feature_names: Array of feature names from vectorizer.
        """
        self.feature_names = feature_names
        self.feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    def get_highlight_data(
        self,
        raw_text: str,
        instance_shap_values: np.ndarray,
        predicted_class_index: int
    ) -> List[HighlightedToken]:
        """Get structured data for text highlighting with token positions.

        Args:
            raw_text: The original input text.
            instance_shap_values: SHAP values for the instance.
            predicted_class_index: Index of the predicted class.

        Returns:
            List of HighlightedToken objects containing word, position, score, and direction.
        """
        if instance_shap_values.ndim == 2:
            class_shap_values = instance_shap_values[:, predicted_class_index]
        else:
            class_shap_values = instance_shap_values

        highlight_data = []
        
        # Tokenize by word boundaries to preserve start/end positions
        for match in re.finditer(r'\b\w+\b', raw_text):
            token = match.group()
            start_pos = match.start()
            end_pos = match.end()
            token_lower = token.lower()

            if token_lower in self.feature_to_idx:
                idx = self.feature_to_idx[token_lower]
                score = float(class_shap_values[idx])
                direction = "positive" if score > 0 else "negative" if score < 0 else "neutral"
                
                highlight_data.append(HighlightedToken(
                    word=token,
                    start_position=start_pos,
                    end_position=end_pos,
                    importance=score,
                    direction=direction
                ))
            else:
                highlight_data.append(HighlightedToken(
                    word=token,
                    start_position=start_pos,
                    end_position=end_pos,
                    importance=0.0,
                    direction="neutral"
                ))

        return highlight_data
