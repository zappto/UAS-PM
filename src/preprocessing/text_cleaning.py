"""Text cleaning module — regex-based noise removal.

Removes URLs, HTML tags, mentions, hashtags, emojis, numbers,
punctuation, and extra whitespace from text. Each removal type
is independently controllable via boolean flags.

This is step 2 of the TRD preprocessing pipeline:
Raw Text → Lowercase → **Cleaning** → Tokenizing → ...
"""

import re


# ─── Compiled Regex Patterns ─────────────────────────────────────────────────

_URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+", re.IGNORECASE
)
_HTML_PATTERN = re.compile(
    r"<[^>]+>"
)
_MENTION_PATTERN = re.compile(
    r"@\w+"
)
_HASHTAG_PATTERN = re.compile(
    r"#\w+"
)
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Symbols & pictographs
    "\U0001F680-\U0001F6FF"  # Transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # Flags
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"  # Enclosed characters
    "\U0001F900-\U0001F9FF"  # Supplemental symbols
    "\U0001FA00-\U0001FA6F"  # Chess symbols
    "\U0001FA70-\U0001FAFF"  # Symbols extended-A
    "\U00002600-\U000026FF"  # Misc symbols
    "\U0000FE00-\U0000FE0F"  # Variation selectors
    "\U0000200D"             # Zero-width joiner
    "\U00000023\U0000FE0F\U000020E3"  # Keycap
    "]+",
    flags=re.UNICODE,
)
_NUMBER_PATTERN = re.compile(
    r"\d+"
)
_PUNCTUATION_PATTERN = re.compile(
    r"[^\w\s]"
)
_MULTISPACE_PATTERN = re.compile(
    r"\s{2,}"
)


def clean_text(
    text: str,
    remove_urls: bool = True,
    remove_html: bool = True,
    remove_mentions: bool = True,
    remove_hashtags: bool = True,
    remove_emojis: bool = True,
    remove_numbers: bool = True,
    remove_punctuation: bool = True,
) -> str:
    """Remove noise from text using regex patterns.

    Each removal type can be independently enabled or disabled
    via its boolean flag. Extra whitespace is always cleaned.

    Args:
        text: Input text string.
        remove_urls: Remove URLs (http/https/www).
        remove_html: Remove HTML tags.
        remove_mentions: Remove @mentions.
        remove_hashtags: Remove #hashtags.
        remove_emojis: Remove emoji characters.
        remove_numbers: Remove numeric characters.
        remove_punctuation: Remove punctuation characters.

    Returns:
        Cleaned text string with noise removed and
        extra whitespace collapsed.

    Examples:
        >>> clean_text("Cek http://example.com @user #tag 😂 123!")
        'Cek'
        >>> clean_text("Hello 123", remove_numbers=False)
        'Hello 123'
    """
    if not isinstance(text, str):
        return ""

    if remove_urls:
        text = _URL_PATTERN.sub(" ", text)

    if remove_html:
        text = _HTML_PATTERN.sub(" ", text)

    if remove_mentions:
        text = _MENTION_PATTERN.sub(" ", text)

    if remove_hashtags:
        text = _HASHTAG_PATTERN.sub(" ", text)

    if remove_emojis:
        text = _EMOJI_PATTERN.sub(" ", text)

    if remove_numbers:
        text = _NUMBER_PATTERN.sub(" ", text)

    if remove_punctuation:
        text = _PUNCTUATION_PATTERN.sub(" ", text)

    # Always collapse multiple spaces and strip
    text = _MULTISPACE_PATTERN.sub(" ", text)
    return text.strip()
