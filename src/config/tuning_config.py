"""GridSearchCV configuration for hyperparameter tuning.

Provides a configuration dataclass to control all GridSearchCV
parameters without modifying implementation code.
"""

from dataclasses import dataclass


@dataclass
class TuningConfig:
    """Configuration for GridSearchCV hyperparameter tuning.

    All parameters map directly to scikit-learn's GridSearchCV.
    Defaults follow the project TRD and MRD requirements.

    Attributes:
        cv: Number of cross-validation folds (stratified).
        scoring: Scoring metric for model selection.
        verbose: Verbosity level for GridSearchCV output.
        n_jobs: Number of parallel jobs (-1 = all cores).
        refit: Refit the best estimator on full training set.
        return_train_score: Include training scores in results.
        save_results: Whether to save tuning results to disk.
        random_state: Random seed for reproducibility.
    """

    cv: int = 5
    scoring: str = "f1_macro"
    verbose: int = 1
    n_jobs: int = -1
    refit: bool = True
    return_train_score: bool = True
    save_results: bool = True
    random_state: int = 42

    def get_grid_search_params(self) -> dict:
        """Return parameters for GridSearchCV constructor.

        Returns:
            Dictionary of GridSearchCV keyword arguments.
        """
        return {
            "cv": self.cv,
            "scoring": self.scoring,
            "verbose": self.verbose,
            "n_jobs": self.n_jobs,
            "refit": self.refit,
            "return_train_score": self.return_train_score,
        }

    def to_dict(self) -> dict:
        """Serialize all configuration values to a dictionary.

        Returns:
            Dictionary of all configuration parameters.
        """
        return {
            "cv": self.cv,
            "scoring": self.scoring,
            "verbose": self.verbose,
            "n_jobs": self.n_jobs,
            "refit": self.refit,
            "return_train_score": self.return_train_score,
            "save_results": self.save_results,
            "random_state": self.random_state,
        }
