"""Model Information Page.

Displays metadata regarding the trained models and configurations.
"""
import streamlit as st
import json
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from components.sidebar import render_sidebar
from components.footer import render_footer
from utils.constants import MODEL_CONFIG_PATH

st.set_page_config(page_title="Model Information", page_icon="ℹ️", layout="wide")
render_sidebar()

st.title("Model Information")
st.write("Detailed metadata about the models and feature extraction pipeline.")

if os.path.exists(MODEL_CONFIG_PATH):
    with open(MODEL_CONFIG_PATH, 'r') as f:
        config = json.load(f)
        
    st.json(config, expanded=True)
else:
    st.info("Metadata file not found. Falling back to default project information.")
    
    st.markdown("### Algorithms")
    st.markdown("- Multinomial Naive Bayes")
    st.markdown("- Logistic Regression")
    st.markdown("- Linear Support Vector Machine")
    
    st.markdown("### Feature Extraction")
    st.markdown("- TF-IDF (Term Frequency-Inverse Document Frequency)")
    
    st.markdown("### Classes")
    st.markdown("1. Normal")
    st.markdown("2. Insult")
    st.markdown("3. Harassment")
    st.markdown("4. Threat")
    st.markdown("5. Hate Speech")
    
render_footer()
