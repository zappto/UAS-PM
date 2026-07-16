"""TF-IDF and data split configuration.

Provides a configuration-driven approach to control all
TF-IDF vectorizer parameters and train/test split settings
without modifying implementation code.
"""

from dataclasses import dataclass


@dataclass
class TfidfConfig:
    """Configuration for TF-IDF vectorization and data split.

    All parameters map directly to scikit-learn's TfidfVectorizer
    and train_test_split. Defaults are tuned for Indonesian text
    classification with classical ML algorithms.

    Attributes:
        max_features: Maximum vocabulary size. None = no limit.
        min_df: Ignore terms with document frequency below this.
            Int = absolute count, float = proportion.
        max_df: Ignore terms with document frequency above this.
        ngram_range: Range of n-grams to extract.
        norm: Normalization method ('l1', 'l2', or None).
        use_idf: Enable inverse-document-frequency reweighting.
        smooth_idf: Smooth IDF weights to prevent zero divisions.
        sublinear_tf: Apply sublinear TF scaling (1 + log(tf)).
        analyzer: Feature type ('word', 'char', 'char_wb').
        lowercase: Convert to lowercase before tokenizing.
            False because preprocessing already applied case folding.
        token_pattern: Regex pattern for tokenization.
        test_size: Proportion of data for testing.
        random_state: Random seed for reproducibility.
        stratify: Enable stratified split to maintain class distribution.
    """

    max_features: int | None = None
    min_df: int | float = 2
    max_df: float = 0.95
    ngram_range: tuple[int, int] = (1, 1)
    norm: str = "l2"
    use_idf: bool = True
    smooth_idf: bool = True
    sublinear_tf: bool = True
    analyzer: str = "word"
    lowercase: bool = False
    token_pattern: str = r"(?u)\b\w+\b"
    test_size: float = 0.20
    random_state: int = 42
    stratify: bool = True

    def get_vectorizer_params(self) -> dict:
        """Return parameters for TfidfVectorizer constructor.

        Returns:
            Dictionary of TF-IDF vectorizer keyword arguments.
        """
        return {
            "max_features": self.max_features,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "ngram_range": self.ngram_range,
            "norm": self.norm,
            "use_idf": self.use_idf,
            "smooth_idf": self.smooth_idf,
            "sublinear_tf": self.sublinear_tf,
            "analyzer": self.analyzer,
            "lowercase": self.lowercase,
            "token_pattern": self.token_pattern,
        }

    def get_split_params(self) -> dict:
        """Return parameters for train_test_split.

        Returns:
            Dictionary of split keyword arguments.
        """
        return {
            "test_size": self.test_size,
            "random_state": self.random_state,
        }

    def to_dict(self) -> dict:
        """Serialize all configuration values to a dictionary.

        Returns:
            Dictionary of all configuration parameters.
        """
        return {
            "max_features": self.max_features,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "ngram_range": str(self.ngram_range),
            "norm": self.norm,
            "use_idf": self.use_idf,
            "smooth_idf": self.smooth_idf,
            "sublinear_tf": self.sublinear_tf,
            "analyzer": self.analyzer,
            "lowercase": self.lowercase,
            "token_pattern": self.token_pattern,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "stratify": self.stratify,
        }
