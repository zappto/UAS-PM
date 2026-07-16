"""Tokenization module for Indonesian text.

Splits text into word tokens using whitespace-based tokenization,
which is appropriate for Indonesian text combined with TF-IDF.

This is step 3 of the TRD preprocessing pipeline:
Raw Text → Lowercase → Cleaning → **Tokenizing** → ...
"""


def tokenize(text: str) -> list[str]:
    """Tokenize Indonesian text into a list of word tokens.

    Uses whitespace-based splitting which is effective for
    Indonesian text, especially after cleaning has removed
    punctuation and special characters.

    Args:
        text: Input text string (ideally already cleaned).

    Returns:
        List of word tokens. Empty list if input is invalid.

    Examples:
        >>> tokenize("saya suka makan nasi")
        ['saya', 'suka', 'makan', 'nasi']
        >>> tokenize("")
        []
    """
    if not isinstance(text, str) or not text.strip():
        return []
    return text.split()
