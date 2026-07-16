"""Base estimator factory for GridSearchCV.

Provides unfitted base scikit-learn estimators for each
algorithm. These are the raw estimators that GridSearchCV
will clone and fit during the search.
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def get_base_estimator(
    model_name: str,
    random_state: int = 42,
) -> object:
    """Return an unfitted base sklearn estimator for GridSearchCV.

    Args:
        model_name: Name of the model to create.
        random_state: Random seed for reproducibility.

    Returns:
        An unfitted scikit-learn estimator instance.

    Raises:
        ValueError: If model_name is not recognized.
    """
    if model_name == "naive_bayes":
        return MultinomialNB()
    elif model_name == "logistic_regression":
        return LogisticRegression(random_state=random_state)
    elif model_name == "svm":
        # GridSearchCV tunes the bare SVC.
        # CalibratedClassifierCV is applied AFTER finding best C.
        return SVC(kernel="linear", random_state=random_state)
    else:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available: ['naive_bayes', 'logistic_regression', 'svm']"
        )


def get_tunable_models() -> list[str]:
    """Return list of model names that can be tuned.

    Returns:
        List of model name strings.
    """
    return ["naive_bayes", "logistic_regression", "svm"]
