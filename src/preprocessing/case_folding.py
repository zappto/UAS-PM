"""Case folding module — lowercase conversion and Unicode normalization.

This is the first step of the preprocessing pipeline as defined
in the TRD: Raw Text → Lowercase → Cleaning → ...
"""

import unicodedata


def case_fold(text: str) -> str:
    """Apply case folding to text.

    Performs two operations:
    1. Unicode NFKD normalization to decompose special characters.
    2. Lowercase conversion.

    Args:
        text: Input text string.

    Returns:
        Lowercased, Unicode-normalized text.

    Examples:
        >>> case_fold("Halo DUNIA!")
        'halo dunia!'
        >>> case_fold("Café Résumé")
        'café résumé'
    """
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.lower()
