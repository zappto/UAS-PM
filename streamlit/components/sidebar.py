"""Sidebar component for Streamlit application.

Handles navigation and project information display.
"""
import streamlit as st
from utils.constants import APP_NAME, APP_VERSION

def render_sidebar():
    """Render the standard application sidebar."""
    with st.sidebar:
        st.title(APP_NAME)
        st.caption(f"Version {APP_VERSION}")
        
        st.markdown("---")
        
        st.markdown("### Navigation")
        st.page_link("app.py", label="Home", icon="🏠")
        st.page_link("pages/02_Prediction.py", label="Prediction", icon="🎯")
        st.page_link("pages/03_Model_Comparison.py", label="Model Comparison", icon="📊")
        st.page_link("pages/04_Model_Information.py", label="Model Information", icon="ℹ️")
        st.page_link("pages/05_About.py", label="About", icon="📖")
        
        st.markdown("---")
        st.markdown(
            "Built with [Streamlit](https://streamlit.io/)\n\n"
            "Cyberbullying Text Classification."
        )
