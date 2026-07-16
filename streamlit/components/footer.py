"""Footer component.

Provides a standardized footer for the application.
"""
import streamlit as st

def render_footer():
    """Render the application footer."""
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "<p>Machine Learning Research Project</p>"
        "<p>Cyberbullying Type Classification on Indonesian Text</p>"
        "</div>",
        unsafe_allow_html=True
    )
