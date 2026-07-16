"""Logistic Regression model wrapper.

Baseline classifier using default parameters.
No hyperparameter tuning — uses scikit-learn defaults
with max_iter=1000 as a convergence safeguard.
"""

from sklearn.linear_model import LogisticRegression

from src.config.model_config import ModelConfig
from src.training.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier wrapper.

    Uses sklearn.linear_model.LogisticRegression with default
    parameters. max_iter is set to 1000 to ensure convergence
    on large sparse TF-IDF datasets.

    Default sklearn parameters:
        - C=1.0
        - penalty='l2'
        - solver='lbfgs'
        - multi_class='auto'
        - max_iter=1000 (increased from default 100)
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        """Initialize Logistic Regression model.

        Args:
            config: Training configuration.
        """
        super().__init__(config)

    def build(self) -> None:
        """Build LogisticRegression with default parameters."""
        self._model = LogisticRegression(
            random_state=self._config.random_state,
            max_iter=1000,
        )

    @property
    def model_name(self) -> str:
        """Return model name."""
        return "LogisticRegression"
