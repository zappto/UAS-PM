"""Prediction card component.

Displays the main prediction result, confidence score, and inference time.
"""
import streamlit as st
from utils.formatter import format_confidence

def render_prediction_card(prediction: str, confidence: float, inference_time: float):
    """Render the primary prediction outcome.
    
    Args:
        prediction: The predicted class label.
        confidence: The confidence score (0 to 1).
        inference_time: Time taken for prediction in seconds.
    """
    st.markdown("### Prediction Result")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Predicted Class", value=prediction.replace('_', ' ').title())
        
    with col2:
        st.metric(label="Confidence", value=format_confidence(confidence))
        
    with col3:
        st.metric(label="Inference Time", value=f"{inference_time:.3f} s")
