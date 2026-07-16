"""Probability chart component.

Renders class probabilities using matplotlib.
"""
import streamlit as st
import matplotlib.pyplot as plt
from typing import List, Dict

def render_probability_chart(probabilities: List[Dict]):
    """Render a horizontal bar chart of probabilities.
    
    Args:
        probabilities: List of dictionaries with 'class_name' and 'probability'.
    """
    st.markdown("### Probability Distribution")
    
    # Sort probabilities descending
    sorted_probs = sorted(probabilities, key=lambda x: x['probability'], reverse=False)
    
    classes = [item['class_name'].replace('_', ' ').title() for item in sorted_probs]
    probs = [item['probability'] * 100 for item in sorted_probs]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Matplotlib styling for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    bars = ax.barh(classes, probs, color="#4F8BF9")
    
    ax.set_xlabel("Probability (%)")
    ax.set_xlim(0, 100)
    
    # Add data labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 1 if width < 90 else width - 5
        color = 'black' if width < 90 else 'white'
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                ha='left' if width < 90 else 'right', va='center', color=color)
        
    st.pyplot(fig)
