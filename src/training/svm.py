"""Support Vector Machine model wrapper.

Linear SVM with probability estimation enabled.
Uses default parameters — no hyperparameter tuning.
"""

from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from src.config.model_config import ModelConfig
from src.training.base_model import BaseModel


class SVMModel(BaseModel):
    """Linear SVM classifier wrapper.

    Uses sklearn.svm.SVC with linear kernel, wrapped in 
    CalibratedClassifierCV to enable probability estimation (predict_proba).
    This avoids the deprecation warning from SVC(probability=True).

    Default sklearn parameters:
        - kernel='linear' (per research scope)
        - C=1.0
        - ensemble=False (for CalibratedClassifierCV)
        - random_state=42
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        """Initialize SVM model.

        Args:
            config: Training configuration.
        """
        super().__init__(config)

    def build(self) -> None:
        """Build SVC with linear kernel and probability estimation."""
        base_svm = SVC(
            kernel="linear",
            random_state=self._config.random_state,
        )
        self._model = CalibratedClassifierCV(
            estimator=base_svm,
            ensemble=False,
            cv=5,
        )

    @property
    def model_name(self) -> str:
        """Return model name."""
        return "SVM"
