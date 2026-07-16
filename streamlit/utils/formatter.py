"""Formatter utility for Streamlit UI.

Provides string and number formatting functions for consistent UI presentation.
"""

def format_confidence(confidence: float) -> str:
    """Format confidence score as a percentage.
    
    Args:
        confidence: Float value between 0 and 1.
        
    Returns:
        Formatted percentage string (e.g., '95.2%').
    """
    return f"{confidence * 100:.1f}%"


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text for UI display if it exceeds max_length.
    
    Args:
        text: Input string.
        max_length: Maximum allowed characters.
        
    Returns:
        Truncated string with ellipsis if necessary.
    """
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text
