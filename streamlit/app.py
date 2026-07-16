"""Main application entry point.

Configures the Streamlit page layout, theme, and sets up multi-page navigation.
"""
import sys
import os
import streamlit as st

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.constants import APP_NAME
from components.sidebar import render_sidebar
from components.footer import render_footer

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

render_sidebar()

st.title(f"Welcome to {APP_NAME}")

st.markdown("""
### Please use the navigation sidebar to explore the application.
- **Home**: Project overview and workflow.
- **Prediction**: Test the model and view explainability.
- **Model Comparison**: View evaluation results.
- **Model Information**: Model metadata.
- **About**: Research background.
""")

render_footer()
