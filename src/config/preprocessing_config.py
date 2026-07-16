"""Preprocessing configuration dataclass.

Provides a configuration-driven approach to enable or disable
individual preprocessing steps without modifying source code.
"""

from dataclasses import dataclass, field


@dataclass
class PreprocessingConfig:
    """Configuration for the text preprocessing pipeline.

    Each flag controls whether a specific preprocessing step
    is applied. All steps are enabled by default except
    slang normalization, which is optional.

    Attributes:
        enable_case_folding: Apply lowercase + Unicode normalization.
        enable_url_removal: Remove URLs from text.
        enable_html_removal: Remove HTML tags from text.
        enable_mention_removal: Remove @mentions from text.
        enable_hashtag_removal: Remove #hashtags from text.
        enable_emoji_removal: Remove emojis from text.
        enable_punctuation_removal: Remove punctuation from text.
        enable_number_removal: Remove numbers from text.
        enable_stopword_removal: Remove Indonesian stopwords.
        enable_stemming: Apply Indonesian stemming (Sastrawi).
        enable_slang_normalization: Normalize slang/alay words.
            Disabled by default as it is not explicitly required
            by the DRD, but can be enabled for improved results.
        extra_stopwords: Additional stopwords to include beyond
            the base NLTK Indonesian stopword list.
    """

    enable_case_folding: bool = True
    enable_url_removal: bool = True
    enable_html_removal: bool = True
    enable_mention_removal: bool = True
    enable_hashtag_removal: bool = True
    enable_emoji_removal: bool = True
    enable_punctuation_removal: bool = True
    enable_number_removal: bool = True
    enable_stopword_removal: bool = True
    enable_stemming: bool = True
    enable_slang_normalization: bool = False
    extra_stopwords: set[str] = field(default_factory=set)

    def get_active_steps(self) -> list[str]:
        """Return a list of enabled preprocessing step names.

        Returns:
            List of step names that are currently enabled.
        """
        steps = []
        step_map = {
            "Case Folding": self.enable_case_folding,
            "URL Removal": self.enable_url_removal,
            "HTML Removal": self.enable_html_removal,
            "Mention Removal": self.enable_mention_removal,
            "Hashtag Removal": self.enable_hashtag_removal,
            "Emoji Removal": self.enable_emoji_removal,
            "Punctuation Removal": self.enable_punctuation_removal,
            "Number Removal": self.enable_number_removal,
            "Stopword Removal": self.enable_stopword_removal,
            "Stemming": self.enable_stemming,
            "Slang Normalization": self.enable_slang_normalization,
        }
        for name, enabled in step_map.items():
            if enabled:
                steps.append(name)
        return steps
