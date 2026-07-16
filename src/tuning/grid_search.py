"""Generic GridSearchCV wrapper.

Provides a reusable function to run GridSearchCV on any
scikit-learn estimator with any parameter grid and
configurable search settings.
"""

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.config.tuning_config import TuningConfig


def run_grid_search(
    estimator,
    param_grid: dict,
    X_train,
    y_train,
    config: TuningConfig | None = None,
) -> GridSearchCV:
    """Run GridSearchCV on an estimator with the given parameter grid.

    Uses stratified K-fold cross-validation to maintain class
    distribution in each fold.

    Args:
        estimator: Unfitted scikit-learn estimator.
        param_grid: Dictionary of parameter names to search values.
        X_train: Training feature matrix (sparse or dense).
        y_train: Training labels.
        config: Tuning configuration. Uses defaults if None.

    Returns:
        Fitted GridSearchCV object with access to:
            - best_estimator_: Best fitted model.
            - best_params_: Best parameter combination.
            - best_score_: Best cross-validation score.
            - cv_results_: Full cross-validation results.

    Raises:
        ValueError: If param_grid is empty.
    """
    if config is None:
        config = TuningConfig()

    if not param_grid:
        raise ValueError("Parameter grid cannot be empty.")

    # Use StratifiedKFold for class distribution preservation
    cv_strategy = StratifiedKFold(
        n_splits=config.cv,
        shuffle=True,
        random_state=config.random_state,
    )

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring=config.scoring,
        verbose=config.verbose,
        n_jobs=config.n_jobs,
        refit=config.refit,
        return_train_score=config.return_train_score,
    )

    grid_search.fit(X_train, y_train)

    return grid_search
