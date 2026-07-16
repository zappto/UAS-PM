"""TF-IDF Vectorizer factory.

Creates a configured scikit-learn TfidfVectorizer instance
based on the project's TfidfConfig. The vectorizer is returned
unfitted — fitting is handled by the pipeline.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

from src.config.tfidf_config import TfidfConfig


def create_tfidf_vectorizer(
    config: TfidfConfig | None = None,
) -> TfidfVectorizer:
    """Create a configured but unfitted TF-IDF vectorizer.

    All parameters are sourced from TfidfConfig to ensure
    consistency and reproducibility across experiments.

    Args:
        config: TF-IDF configuration. Uses defaults if None.

    Returns:
        An unfitted TfidfVectorizer instance ready for fitting.

    Examples:
        >>> config = TfidfConfig(max_features=5000)
        >>> vectorizer = create_tfidf_vectorizer(config)
        >>> type(vectorizer)
        <class 'sklearn.feature_extraction.text.TfidfVectorizer'>
    """
    if config is None:
        config = TfidfConfig()

    params = config.get_vectorizer_params()
    vectorizer = TfidfVectorizer(**params)

    return vectorizer
