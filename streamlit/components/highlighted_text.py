"""Highlighted text component.

Reconstructs and displays the input text with highlights 
based on the model's feature importance.
"""
import streamlit as st
from typing import List, Dict

def render_highlighted_text(raw_text: str, highlighted_tokens: List[Dict]):
    """Render text with highlighted words.
    
    Args:
        raw_text: The original input string.
        highlighted_tokens: List of token dictionaries containing 
                            start_position, end_position, importance, and direction.
    """
    st.markdown("### Explainability: Important Words")
    st.caption("Highlighting indicates words that influenced the prediction (Green = Positive, Red = Negative).")
    
    # We will build HTML string by iterating through the text
    # We need to sort tokens by start_position to safely reconstruct
    sorted_tokens = sorted(highlighted_tokens, key=lambda x: x.get('start_position', 0))
    
    html_out = ""
    current_idx = 0
    
    for token in sorted_tokens:
        start = token.get('start_position', 0)
        end = token.get('end_position', 0)
        direction = token.get('direction', 'neutral')
        importance = token.get('importance', 0.0)
        
        # Add non-highlighted text before the token
        if start > current_idx:
            html_out += raw_text[current_idx:start]
            
        # Determine background color opacity
        # Cap importance for visual scaling
        opacity = min(abs(importance) * 0.5, 0.8) 
        
        if direction == 'positive':
            bg_color = f"rgba(40, 167, 69, {opacity})"
        elif direction == 'negative':
            bg_color = f"rgba(220, 53, 69, {opacity})"
        else:
            bg_color = "transparent"
            
        # Add the highlighted token
        token_text = raw_text[start:end]
        if bg_color != "transparent":
            html_out += f"<span style='background-color: {bg_color}; border-radius: 3px; padding: 0 2px;'>{token_text}</span>"
        else:
            html_out += token_text
            
        current_idx = end
        
    # Add remaining text
    if current_idx < len(raw_text):
        html_out += raw_text[current_idx:]
        
    st.markdown(f"<div style='padding: 15px; border: 1px solid #ddd; border-radius: 8px; line-height: 1.6;'>{html_out}</div>", unsafe_allow_html=True)
