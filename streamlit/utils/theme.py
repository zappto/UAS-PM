"""Theme utility for Streamlit application.

Defines standardized colors and styles to ensure light/dark mode compatibility.
"""

# Semantic Colors
COLORS = {
    "primary": "#4F8BF9",
    "positive": "#28a745",
    "negative": "#dc3545",
    "neutral": "#6c757d",
    "background_light": "#f8f9fa",
    "background_dark": "#1e1e1e",
    "text_light": "#ffffff",
    "text_dark": "#000000"
}

# Typography and Layout
STYLE_CONFIG = {
    "font_main": "sans-serif",
    "card_padding": "1.5rem",
    "border_radius": "8px",
}

# Matplotlib specific styles for matching Streamlit theme
PLOT_COLORS = {
    "bar_default": "#4F8BF9",
    "grid": "#e0e0e0",
    "text": "#333333"
}
