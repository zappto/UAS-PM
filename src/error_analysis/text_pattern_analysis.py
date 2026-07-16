"""Text pattern analysis for misclassified samples.

Analyzes text length, token count, and frequent words in errors.
(No Explainability/SHAP/LIME logic).
"""

import pandas as pd
from collections import Counter
import re


def analyze_text_characteristics(correct_df: pd.DataFrame, incorrect_df: pd.DataFrame) -> dict:
    """Calculate average text length and token count for correct vs incorrect.

    Args:
        correct_df: DataFrame of correctly predicted samples.
        incorrect_df: DataFrame of incorrectly predicted samples.

    Returns:
        Dictionary of statistics.
    """
    def _calc_stats(df):
        if len(df) == 0:
            return {"avg_char_length": 0.0, "avg_token_count": 0.0}
            
        char_len = df["text"].apply(lambda x: len(str(x)))
        token_len = df["text"].apply(lambda x: len(str(x).split()))
        
        return {
            "avg_char_length": char_len.mean(),
            "avg_token_count": token_len.mean(),
        }

    return {
        "correct": _calc_stats(correct_df),
        "incorrect": _calc_stats(incorrect_df),
    }


def find_frequent_error_words(incorrect_df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """Find the most frequently occurring words in misclassified samples.

    This is a basic token counting implementation to fulfill the requirement
    without entering the 'Explainability' phase.

    Args:
        incorrect_df: DataFrame of incorrectly predicted samples.
        top_k: Number of top words to return.

    Returns:
        DataFrame of top words and their counts.
    """
    if len(incorrect_df) == 0:
        return pd.DataFrame()
        
    all_text = " ".join(incorrect_df["text"].astype(str).tolist())
    
    # Basic tokenization (lowercase, alphanumeric only)
    words = re.findall(r'\b[a-z]{3,}\b', all_text.lower())
    
    # We should exclude basic stopwords if we had them, but since we are just doing
    # basic analysis, we rely on the fact that TF-IDF might have handled some.
    # We just count raw tokens here.
    word_counts = Counter(words)
    
    common_words = word_counts.most_common(top_k)
    
    return pd.DataFrame(common_words, columns=["word", "error_frequency"])
