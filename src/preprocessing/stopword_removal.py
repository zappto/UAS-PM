"""Indonesian stopword removal module.

Uses NLTK's Indonesian stopword list as the base, extended with
common informal Indonesian words. The stopword list is extensible
via the extra_stopwords parameter.

This is step 4 of the TRD preprocessing pipeline:
... → Tokenizing → **Stopword Removal** → Stemming → ...
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def get_stopword_list() -> frozenset[str]:
    """Load and return the Indonesian stopword set.

    Combines NLTK's built-in Indonesian stopwords with additional
    common informal/social-media Indonesian words that carry
    little semantic meaning.

    Returns:
        Frozen set of Indonesian stopwords.

    Raises:
        RuntimeError: If NLTK stopwords corpus is not available.
            Run: python3 -c "import nltk; nltk.download('stopwords')"
    """
    try:
        from nltk.corpus import stopwords
        base_stopwords = set(stopwords.words("indonesian"))
    except LookupError:
        raise RuntimeError(
            "NLTK Indonesian stopwords not found. "
            "Run: python3 -c \"import nltk; nltk.download('stopwords')\""
        )

    # Common informal Indonesian words not in NLTK's list
    extra = {
        "yg", "dgn", "nya", "aja", "udah", "gak", "gue", "gw",
        "lu", "lo", "emang", "banget", "bgt", "sih", "dong",
        "deh", "nih", "tuh", "lah", "kan", "kok", "ya", "nah",
        "tau", "tdk", "ga", "gk", "sm", "dr", "utk", "krn",
        "jd", "kalo", "kl", "tp", "jg", "udh", "blm", "bkn",
        "org", "org2", "lbh", "skrg", "sdh", "hrs", "bs",
        "user", "rt",  # Social media artifacts
    }

    return frozenset(base_stopwords | extra)


def remove_stopwords(
    tokens: list[str],
    extra_stopwords: set[str] | None = None,
) -> list[str]:
    """Remove Indonesian stopwords from a list of tokens.

    Args:
        tokens: List of word tokens to filter.
        extra_stopwords: Optional additional stopwords to include
            beyond the base list. Useful for domain-specific
            filtering without modifying the source.

    Returns:
        Filtered list of tokens with stopwords removed.

    Examples:
        >>> remove_stopwords(["saya", "suka", "makan", "yang", "enak"])
        ['suka', 'makan', 'enak']
    """
    if not tokens:
        return []

    stopwords = get_stopword_list()

    if extra_stopwords:
        stopwords = stopwords | frozenset(extra_stopwords)

    return [token for token in tokens if token not in stopwords]
