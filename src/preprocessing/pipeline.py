"""Preprocessing pipeline orchestrator.

Applies the complete text preprocessing pipeline in the order
defined by the TRD:

    1. Case Folding (lowercase + Unicode normalization)
    2. Text Cleaning (URL, HTML, mention, hashtag, emoji, number, punctuation)
    3. Tokenization
    4. Stopword Removal
    5. Stemming (Sastrawi)
    6. Join Tokens

Each step is configurable via PreprocessingConfig. The pipeline
can operate on single texts or entire DataFrames.
"""

import time
from typing import Callable

import pandas as pd

from src.config.preprocessing_config import PreprocessingConfig
from src.preprocessing.case_folding import case_fold
from src.preprocessing.text_cleaning import clean_text
from src.preprocessing.tokenization import tokenize
from src.preprocessing.stopword_removal import remove_stopwords
from src.preprocessing.stemming import stem_tokens


def preprocess_text(
    text: str,
    config: PreprocessingConfig | None = None,
) -> str:
    """Apply the full preprocessing pipeline to a single text.

    Follows the TRD pipeline order exactly:
    Raw Text → Lowercase → Cleaning → Tokenizing →
    Stopword Removal → Stemming → Join Token

    Args:
        text: Input raw text string.
        config: Preprocessing configuration. Uses default config
            (all steps enabled) if None.

    Returns:
        Preprocessed text as a single string with tokens joined.
    """
    if config is None:
        config = PreprocessingConfig()

    if not isinstance(text, str) or not text.strip():
        return ""

    result = text

    # Step 1: Case Folding
    if config.enable_case_folding:
        result = case_fold(result)

    # Step 2: Text Cleaning
    result = clean_text(
        result,
        remove_urls=config.enable_url_removal,
        remove_html=config.enable_html_removal,
        remove_mentions=config.enable_mention_removal,
        remove_hashtags=config.enable_hashtag_removal,
        remove_emojis=config.enable_emoji_removal,
        remove_numbers=config.enable_number_removal,
        remove_punctuation=config.enable_punctuation_removal,
    )

    # Step 3: Tokenization
    tokens = tokenize(result)

    # Step 4: Stopword Removal
    if config.enable_stopword_removal:
        tokens = remove_stopwords(
            tokens,
            extra_stopwords=config.extra_stopwords or None,
        )

    # Step 5: Stemming
    if config.enable_stemming:
        tokens = stem_tokens(tokens)

    # Step 6: Join Tokens
    return " ".join(tokens)


def preprocess_dataframe(
    df: pd.DataFrame,
    config: PreprocessingConfig | None = None,
    text_col: str = "text",
    output_col: str = "text_clean",
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Apply preprocessing to an entire DataFrame.

    Processes each row through the full pipeline and adds the
    result as a new column. Tracks progress and timing.

    Args:
        df: Input DataFrame with a text column.
        config: Preprocessing configuration. Uses default if None.
        text_col: Name of the input text column.
        output_col: Name of the output column for cleaned text.
        progress_callback: Optional callback function called with
            (current_row, total_rows) for progress reporting.

    Returns:
        Tuple of (processed_df, stats_dict):
            - processed_df: DataFrame with the new output column.
            - stats_dict: Dictionary with processing statistics.

    Raises:
        ValueError: If text_col is not found in the DataFrame.
    """
    if config is None:
        config = PreprocessingConfig()

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame")

    result_df = df.copy()
    total_rows = len(result_df)

    # Pre-processing statistics
    text_lengths_before = (
        result_df[text_col]
        .astype(str)
        .apply(lambda x: len(x.split()))
    )

    # Apply preprocessing
    start_time = time.time()
    processed_texts = []

    for idx, text in enumerate(result_df[text_col].astype(str)):
        processed = preprocess_text(text, config)
        processed_texts.append(processed)

        # Progress reporting
        if progress_callback and (idx + 1) % 1000 == 0:
            progress_callback(idx + 1, total_rows)

    elapsed = time.time() - start_time

    result_df[output_col] = processed_texts

    # Post-processing statistics
    text_lengths_after = (
        result_df[output_col]
        .astype(str)
        .apply(lambda x: len(x.split()))
    )

    # Count empty results
    empty_after = int(
        (result_df[output_col].astype(str).str.strip() == "").sum()
    )

    stats = {
        "total_rows": total_rows,
        "processed_rows": total_rows,
        "empty_after_preprocessing": empty_after,
        "avg_word_count_before": round(float(text_lengths_before.mean()), 2),
        "avg_word_count_after": round(float(text_lengths_after.mean()), 2),
        "min_word_count_before": int(text_lengths_before.min()),
        "max_word_count_before": int(text_lengths_before.max()),
        "min_word_count_after": int(text_lengths_after.min()),
        "max_word_count_after": int(text_lengths_after.max()),
        "processing_time_seconds": round(elapsed, 2),
        "rows_per_second": round(total_rows / elapsed, 1) if elapsed > 0 else 0,
        "active_steps": config.get_active_steps(),
    }

    return result_df, stats


def get_example_transformations(
    df: pd.DataFrame,
    text_col: str = "text",
    output_col: str = "text_clean",
    n: int = 5,
) -> list[dict[str, str]]:
    """Extract example before/after transformation pairs.

    Selects examples that show meaningful changes between
    the original and preprocessed text.

    Args:
        df: DataFrame with both original and cleaned text columns.
        text_col: Name of the original text column.
        output_col: Name of the cleaned text column.
        n: Number of examples to return.

    Returns:
        List of dicts with 'before' and 'after' keys.
    """
    examples = []
    for _, row in df.head(n * 3).iterrows():
        before = str(row[text_col])
        after = str(row[output_col])
        if before != after and after.strip():
            examples.append({"before": before, "after": after})
        if len(examples) >= n:
            break
    return examples[:n]
