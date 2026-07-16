"""Indonesian stemming module using Sastrawi.

Applies morphological stemming to reduce Indonesian words to
their root form. Uses a cached singleton stemmer instance
to avoid re-initialization overhead per row.

This is step 5 of the TRD preprocessing pipeline:
... → Stopword Removal → **Stemming** → Join Token
"""

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def get_stemmer() -> Any:
    """Create and cache a Sastrawi stemmer instance.

    The stemmer is expensive to initialize (loads dictionary).
    This function creates it once and caches it for reuse.

    Returns:
        A Sastrawi Stemmer instance.

    Raises:
        ImportError: If Sastrawi is not installed.
            Run: pip install Sastrawi
    """
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    except ImportError:
        raise ImportError(
            "Sastrawi is not installed. "
            "Run: pip install Sastrawi"
        )
    factory = StemmerFactory()
    return factory.create_stemmer()


def stem_tokens(tokens: list[str]) -> list[str]:
    """Stem a list of Indonesian tokens using Sastrawi.

    Each token is individually stemmed to its root form.
    Empty tokens are filtered out.

    Args:
        tokens: List of word tokens to stem.

    Returns:
        List of stemmed tokens.

    Examples:
        >>> stem_tokens(["memakan", "berlari", "pelajaran"])
        ['makan', 'lari', 'ajar']
    """
    if not tokens:
        return []

    stemmer = get_stemmer()
    return [stemmer.stem(token) for token in tokens if token]


def stem_text(text: str) -> str:
    """Stem a complete text string using Sastrawi.

    Convenience function that tokenizes, stems, and re-joins.

    Args:
        text: Input text string.

    Returns:
        Stemmed text with tokens joined by spaces.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    stemmer = get_stemmer()
    return stemmer.stem(text)
