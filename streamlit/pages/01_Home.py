"""Home Page.

Displays research title, objectives, and dataset overview.
"""
import streamlit as st
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from components.sidebar import render_sidebar
from components.footer import render_footer

st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")
render_sidebar()

st.title("Performance Analysis of Machine Learning Algorithms for Cyberbullying Type Classification")

st.markdown("### Project Description")
st.write(
    "This research focuses on classifying cyberbullying types in Indonesian text using "
    "classical Machine Learning algorithms and TF-IDF feature extraction. The models have been "
    "evaluated, and this application serves as a presentation layer for the trained models."
)

st.markdown("### Objectives")
st.markdown("""
- Analyze performance of ML algorithms.
- Compare models based on evaluation metrics.
- Deploy the best model for interactive prediction.
- Provide explainability for predictions.
""")

st.markdown("### Algorithms Used")
st.markdown("""
- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine
""")

st.markdown("### Project Workflow")
st.markdown("""
1. Dataset Preparation & Cleaning
2. Text Preprocessing
3. Feature Extraction (TF-IDF)
4. Model Training & Evaluation
5. Hyperparameter Tuning
6. Explainable AI Implementation
7. **Streamlit Deployment (Current Phase)**
""")

render_footer()
