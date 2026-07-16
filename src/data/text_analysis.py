"""Text analysis utilities for EDA — operates on raw, unprocessed text."""

from collections import Counter

import pandas as pd


def compute_text_lengths(df: pd.DataFrame, col: str = "text") -> pd.DataFrame:
    """Add character count and word count columns to the DataFrame.

    Args:
        df: Input DataFrame.
        col: Name of the text column.

    Returns:
        Copy of the DataFrame with added 'char_count' and 'word_count' columns.
    """
    result = df.copy()
    result["char_count"] = result[col].astype(str).apply(len)
    result["word_count"] = result[col].astype(str).apply(lambda x: len(x.split()))
    return result


def get_text_statistics(df: pd.DataFrame, col: str = "text") -> dict:
    """Compute descriptive statistics for text lengths.

    Args:
        df: Input DataFrame (should have 'char_count' and 'word_count'
            columns; if not, they will be computed).
        col: Name of the text column used to compute lengths if needed.

    Returns:
        Dictionary with statistics for both char_count and word_count:
            mean, median, mode, std, min, max.
    """
    if "word_count" not in df.columns or "char_count" not in df.columns:
        df = compute_text_lengths(df, col)

    stats = {}
    for metric in ["word_count", "char_count"]:
        series = df[metric]
        mode_val = series.mode()
        stats[metric] = {
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "mode": int(mode_val.iloc[0]) if len(mode_val) > 0 else 0,
            "std": round(float(series.std()), 2),
            "min": int(series.min()),
            "max": int(series.max()),
        }
    return stats


def get_vocabulary_stats(df: pd.DataFrame, col: str = "text") -> dict[str, int]:
    """Compute vocabulary statistics from the text column.

    Args:
        df: Input DataFrame.
        col: Name of the text column.

    Returns:
        Dictionary with unique_words, vocabulary_size, total_words.
    """
    all_words: list[str] = []
    for text in df[col].astype(str):
        all_words.extend(text.split())

    unique = set(all_words)
    return {
        "unique_words": len(unique),
        "vocabulary_size": len(unique),
        "total_words": len(all_words),
    }


def get_top_words(
    df: pd.DataFrame, col: str = "text", n: int = 20
) -> list[tuple[str, int]]:
    """Get the top N most frequent words (whitespace-split, no preprocessing).

    Args:
        df: Input DataFrame.
        col: Name of the text column.
        n: Number of top words to return.

    Returns:
        List of (word, count) tuples sorted by count descending.
    """
    word_counts: Counter = Counter()
    for text in df[col].astype(str):
        word_counts.update(text.split())
    return word_counts.most_common(n)


def get_top_ngrams(
    df: pd.DataFrame, col: str = "text", n: int = 2, top_k: int = 20
) -> list[tuple[str, int]]:
    """Get the top K n-grams from the text column.

    Uses simple whitespace tokenization and sliding window.
    No external NLP libraries are used.

    Args:
        df: Input DataFrame.
        col: Name of the text column.
        n: N-gram size (2 for bigrams, 3 for trigrams).
        top_k: Number of top n-grams to return.

    Returns:
        List of (ngram_string, count) tuples sorted by count descending.
    """
    ngram_counts: Counter = Counter()
    for text in df[col].astype(str):
        words = text.split()
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i : i + n])
            ngram_counts[ngram] += 1
    return ngram_counts.most_common(top_k)
