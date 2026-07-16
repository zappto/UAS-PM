"""Multinomial Naive Bayes model wrapper.

Baseline classifier using default parameters.
No hyperparameter tuning — uses scikit-learn defaults.
"""

from sklearn.naive_bayes import MultinomialNB

from src.config.model_config import ModelConfig
from src.training.base_model import BaseModel


class NaiveBayesModel(BaseModel):
    """Multinomial Naive Bayes classifier wrapper.

    Uses sklearn.naive_bayes.MultinomialNB with default parameters.
    Suitable for text classification with TF-IDF features.

    Default sklearn parameters:
        - alpha=1.0 (Laplace smoothing)
        - fit_prior=True
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        """Initialize Naive Bayes model.

        Args:
            config: Training configuration.
        """
        super().__init__(config)

    def build(self) -> None:
        """Build MultinomialNB with default parameters."""
        self._model = MultinomialNB()

    @property
    def model_name(self) -> str:
        """Return model name."""
        return "NaiveBayes"
