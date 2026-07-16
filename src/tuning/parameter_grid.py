"""Parameter grids for hyperparameter tuning.

Defines search spaces for each candidate algorithm.
Grids are intentionally kept reasonable to avoid
unnecessary computational cost.
"""


# ─── Naive Bayes ──────────────────────────────────────────────────────────────

NB_PARAM_GRID: dict = {
    "alpha": [0.1, 0.5, 1.0, 2.0, 5.0],
}

# ─── Logistic Regression ─────────────────────────────────────────────────────

LR_PARAM_GRID: dict = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "solver": ["lbfgs"],
    "max_iter": [500, 1000, 2000],
}

# ─── Linear SVM ──────────────────────────────────────────────────────────────
# Kernel is fixed as 'linear'. Do NOT tune kernel.

SVM_PARAM_GRID: dict = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
}

# ─── Registry ────────────────────────────────────────────────────────────────

_PARAM_GRIDS: dict[str, dict] = {
    "naive_bayes": NB_PARAM_GRID,
    "logistic_regression": LR_PARAM_GRID,
    "svm": SVM_PARAM_GRID,
}


def get_param_grid(model_name: str) -> dict:
    """Return the parameter grid for a specific model.

    Args:
        model_name: Name of the model.

    Returns:
        Dictionary of parameter names to search values.

    Raises:
        ValueError: If model_name is not recognized.
    """
    if model_name not in _PARAM_GRIDS:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available: {list(_PARAM_GRIDS.keys())}"
        )
    return _PARAM_GRIDS[model_name].copy()


def get_all_param_grids() -> dict[str, dict]:
    """Return parameter grids for all models.

    Returns:
        Dictionary mapping model names to their param grids.
    """
    return {name: grid.copy() for name, grid in _PARAM_GRIDS.items()}
