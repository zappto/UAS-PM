"""Model training configuration.

Provides a configuration dataclass for controlling training
behavior. Does NOT contain hyperparameters — those belong
to the hyperparameter tuning stage.
"""

from dataclasses import dataclass

from src.config.settings import MODELS_DIR


@dataclass
class ModelConfig:
    """Configuration for baseline model training.

    Controls training behavior only. No hyperparameters are
    included — hyperparameter tuning is handled in a separate
    notebook (05_hyperparameter_tuning.ipynb).

    Attributes:
        random_state: Random seed for reproducibility.
        model_output_directory: Directory to save trained models.
        overwrite: Allow overwriting existing model files.
        save_model: Whether to save trained models to disk.
        verbose: Print training progress and status messages.
    """

    random_state: int = 42
    model_output_directory: str = MODELS_DIR
    overwrite: bool = False
    save_model: bool = True
    verbose: bool = True

    def to_dict(self) -> dict:
        """Serialize configuration to dictionary.

        Returns:
            Dictionary of all configuration parameters.
        """
        return {
            "random_state": self.random_state,
            "model_output_directory": self.model_output_directory,
            "overwrite": self.overwrite,
            "save_model": self.save_model,
            "verbose": self.verbose,
        }
