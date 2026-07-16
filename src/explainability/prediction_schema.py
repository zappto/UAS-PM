"""Prediction schema module.

Provides strongly typed data models for prediction results and explanations.
"""

from dataclasses import dataclass, asdict
from typing import List


@dataclass
class ProbabilityItem:
    """Class probability details."""
    class_name: str
    probability: float


@dataclass
class WordImportance:
    """Word contribution to the prediction."""
    word: str
    importance: float
    direction: str  # e.g., "positive", "negative"


@dataclass
class HighlightedToken:
    """Token with positional information for highlighting."""
    word: str
    start_position: int
    end_position: int
    importance: float
    direction: str


@dataclass
class PredictionResult:
    """Structured output for the inference API."""
    prediction: str
    confidence: float
    probabilities: List[ProbabilityItem]
    top_positive_words: List[WordImportance]
    top_negative_words: List[WordImportance]
    highlighted_tokens: List[HighlightedToken]
    model_name: str
    inference_time: float

    def to_dict(self) -> dict:
        """Serialize the result to a dictionary."""
        return asdict(self)
