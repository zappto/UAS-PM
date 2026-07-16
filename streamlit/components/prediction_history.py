"""Prediction history component.

Displays previous predictions from the session state.
"""
import streamlit as st
from utils.formatter import format_confidence

def render_prediction_history():
    """Render an accordion displaying past predictions."""
    history = st.session_state.get('prediction_history', [])
    
    if not history:
        return
        
    st.markdown("### Prediction History")
    
    with st.expander("View past predictions from this session"):
        for item in history:
            text = item.get('text', '')
            # Truncate text for summary
            trunc_text = text[:50] + "..." if len(text) > 50 else text
            
            st.markdown(f"**{item.get('timestamp')}** | Predicted: `{item.get('prediction')}` (Conf: {format_confidence(item.get('confidence', 0))})")
            st.text(f"Input: {trunc_text}")
            st.markdown("---")
