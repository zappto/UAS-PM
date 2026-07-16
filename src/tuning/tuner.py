"""Algorithm-specific tuning orchestrator.

Provides per-model tuning functions and a combined
tune_all_models function. Each tuner returns a standardized
result dictionary with best model, params, score, and timing.

The SVM tuner has special handling: GridSearchCV runs on the
bare SVC, then the best SVC is wrapped in CalibratedClassifierCV
to restore predict_proba() support.
"""

import time

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from src.config.tuning_config import TuningConfig
from src.tuning.grid_search import run_grid_search
from src.tuning.model_selector import get_base_estimator
from src.tuning.parameter_grid import get_param_grid
from src.tuning.validator import validate_tuning_inputs


def tune_naive_bayes(
    X_train,
    y_train,
    config: TuningConfig | None = None,
) -> dict:
    """Tune Multinomial Naive Bayes using GridSearchCV.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        config: Tuning configuration.

    Returns:
        Dictionary with:
            - model_name (str)
            - best_model: Best fitted estimator.
            - best_params (dict): Best parameter combination.
            - best_score (float): Best CV score.
            - cv_results (dict): Full CV results from GridSearchCV.
            - duration (float): Search time in seconds.
            - n_candidates (int): Number of parameter combos tested.
            - n_cv_iterations (int): Total CV fits.
    """
    if config is None:
        config = TuningConfig()

    param_grid = get_param_grid("naive_bayes")
    validate_tuning_inputs(X_train, y_train, param_grid)

    estimator = get_base_estimator("naive_bayes", config.random_state)

    print(f"  Tuning NaiveBayes...")
    print(f"  Param grid: {param_grid}")

    start = time.time()
    gs = run_grid_search(estimator, param_grid, X_train, y_train, config)
    duration = time.time() - start

    n_candidates = len(gs.cv_results_["params"])

    print(f"  ✓ Best score: {gs.best_score_:.4f}")
    print(f"  ✓ Best params: {gs.best_params_}")
    print(f"  ✓ Duration: {duration:.2f}s")

    return {
        "model_name": "NaiveBayes",
        "model_key": "naive_bayes",
        "best_model": gs.best_estimator_,
        "best_params": gs.best_params_,
        "best_score": round(gs.best_score_, 6),
        "cv_results": gs.cv_results_,
        "duration": round(duration, 4),
        "n_candidates": n_candidates,
        "n_cv_iterations": n_candidates * config.cv,
        "param_grid": param_grid,
    }


def tune_logistic_regression(
    X_train,
    y_train,
    config: TuningConfig | None = None,
) -> dict:
    """Tune Logistic Regression using GridSearchCV.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        config: Tuning configuration.

    Returns:
        Standardized result dictionary (same as tune_naive_bayes).
    """
    if config is None:
        config = TuningConfig()

    param_grid = get_param_grid("logistic_regression")
    validate_tuning_inputs(X_train, y_train, param_grid)

    estimator = get_base_estimator("logistic_regression", config.random_state)

    print(f"  Tuning LogisticRegression...")
    print(f"  Param grid: {param_grid}")

    start = time.time()
    gs = run_grid_search(estimator, param_grid, X_train, y_train, config)
    duration = time.time() - start

    n_candidates = len(gs.cv_results_["params"])

    print(f"  ✓ Best score: {gs.best_score_:.4f}")
    print(f"  ✓ Best params: {gs.best_params_}")
    print(f"  ✓ Duration: {duration:.2f}s")

    return {
        "model_name": "LogisticRegression",
        "model_key": "logistic_regression",
        "best_model": gs.best_estimator_,
        "best_params": gs.best_params_,
        "best_score": round(gs.best_score_, 6),
        "cv_results": gs.cv_results_,
        "duration": round(duration, 4),
        "n_candidates": n_candidates,
        "n_cv_iterations": n_candidates * config.cv,
        "param_grid": param_grid,
    }


def tune_svm(
    X_train,
    y_train,
    config: TuningConfig | None = None,
) -> dict:
    """Tune Linear SVM using GridSearchCV.

    Special handling:
        1. GridSearchCV runs on bare SVC(kernel='linear')
        2. Best C is extracted from results
        3. Best SVC is wrapped in CalibratedClassifierCV
        4. Calibrated model is fitted on full X_train
        5. Calibrated model is returned as best_model

    This ensures the saved model supports predict_proba()
    without using the deprecated probability=True.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        config: Tuning configuration.

    Returns:
        Standardized result dictionary. best_model is
        a CalibratedClassifierCV wrapping the best SVC.
    """
    if config is None:
        config = TuningConfig()

    param_grid = get_param_grid("svm")
    validate_tuning_inputs(X_train, y_train, param_grid)

    estimator = get_base_estimator("svm", config.random_state)

    print(f"  Tuning SVM (Linear)...")
    print(f"  Param grid: {param_grid}")

    start = time.time()
    gs = run_grid_search(estimator, param_grid, X_train, y_train, config)
    search_duration = time.time() - start

    n_candidates = len(gs.cv_results_["params"])

    # Wrap best SVC in CalibratedClassifierCV for predict_proba
    best_c = gs.best_params_["C"]
    best_svc = SVC(
        kernel="linear",
        C=best_c,
        random_state=config.random_state,
    )
    calibrated_model = CalibratedClassifierCV(
        estimator=best_svc,
        ensemble=False,
        cv=5,
    )

    print(f"  Fitting CalibratedClassifierCV with best C={best_c}...")
    calibrated_model.fit(X_train, y_train)
    total_duration = time.time() - start

    print(f"  ✓ Best score: {gs.best_score_:.4f}")
    print(f"  ✓ Best params: {gs.best_params_}")
    print(f"  ✓ Duration: {total_duration:.2f}s "
          f"(search={search_duration:.2f}s)")

    return {
        "model_name": "SVM",
        "model_key": "svm",
        "best_model": calibrated_model,
        "best_params": gs.best_params_,
        "best_score": round(gs.best_score_, 6),
        "cv_results": gs.cv_results_,
        "duration": round(total_duration, 4),
        "n_candidates": n_candidates,
        "n_cv_iterations": n_candidates * config.cv,
        "param_grid": param_grid,
    }


def tune_all_models(
    X_train,
    y_train,
    config: TuningConfig | None = None,
) -> list[dict]:
    """Tune all registered models sequentially.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        config: Tuning configuration.

    Returns:
        List of result dictionaries, one per model.
    """
    if config is None:
        config = TuningConfig()

    print("=" * 60)
    print("  Hyperparameter Tuning (GridSearchCV)")
    print("=" * 60)
    print(f"  Scoring: {config.scoring}")
    print(f"  CV Folds: {config.cv}")
    print(f"  Parallel Jobs: {config.n_jobs}")
    print()

    tuners = [
        ("naive_bayes", tune_naive_bayes),
        ("logistic_regression", tune_logistic_regression),
        ("svm", tune_svm),
    ]

    results = []
    for i, (name, tuner_fn) in enumerate(tuners, 1):
        print(f"[{i}/{len(tuners)}] {name}")
        result = tuner_fn(X_train, y_train, config)
        results.append(result)
        print()

    print("=" * 60)
    print("  ✓ All models tuned")
    print("=" * 60)

    return results
