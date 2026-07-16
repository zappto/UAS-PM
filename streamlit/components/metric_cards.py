"""Metric cards component.

Reusable metric display cards.
"""
import streamlit as st

def render_metric_cards(metrics_dict):
    """Render a grid of metric cards.
    
    Args:
        metrics_dict: Dictionary mapping metric name to value.
    """
    cols = st.columns(len(metrics_dict))
    
    for col, (label, value) in zip(cols, metrics_dict.items()):
        with col:
            st.metric(label=label, value=value)
