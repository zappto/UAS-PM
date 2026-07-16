"""Session management utility for Streamlit.

Handles caching of ML models and prediction history using Streamlit's Session State.
"""

import sys
import os
import streamlit as st
import datetime

# Ensure project root is in path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.explainability.inference import InferencePipeline


@st.cache_resource(show_spinner=False)
def load_pipeline() -> InferencePipeline:
    """Load the inference pipeline exactly once.
    
    Returns:
        InferencePipeline: Instantiated pipeline ready for predictions.
    """
    return InferencePipeline()


def initialize_session_state() -> None:
    """Initialize necessary session state variables for the application."""
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    
    if "current_input_text" not in st.session_state:
        st.session_state.current_input_text = ""


def add_to_history(input_text: str, prediction_result: dict) -> None:
    """Add a prediction to the session history.
    
    Args:
        input_text: The user's input text.
        prediction_result: Dictionary output from the inference pipeline.
    """
    history_item = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": input_text,
        "prediction": prediction_result.get("prediction"),
        "confidence": prediction_result.get("confidence"),
        "inference_time": prediction_result.get("inference_time")
    }
    # Prepend to keep newest first
    st.session_state.prediction_history.insert(0, history_item)


def reset_session() -> None:
    """Clear the session history."""
    st.session_state.prediction_history = []
    st.session_state.current_input_text = ""
