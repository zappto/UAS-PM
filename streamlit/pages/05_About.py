"""About Page.

Displays research background, author, and methodology.
"""
import streamlit as st
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from components.sidebar import render_sidebar
from components.footer import render_footer

st.set_page_config(page_title="About", page_icon="📖", layout="wide")
render_sidebar()

st.title("About the Research")

st.markdown("### Background")
st.write(
    "Cyberbullying has become a significant issue in the digital era. As text-based "
    "interactions grow, the need to automatically classify and identify different types "
    "of cyberbullying in Indonesian social media contexts has become increasingly important."
)

st.markdown("### Methodology")
st.write(
    "This research collected publicly available data, standardized labels across various "
    "cyberbullying types, applied text preprocessing, and evaluated three classical ML "
    "algorithms (Naive Bayes, Logistic Regression, SVM) using TF-IDF feature representations."
)

st.markdown("### Author Information")
st.markdown("- **Author**: [Author Name]")
st.markdown("- **Supervisor**: [Supervisor Name]")
st.markdown("- **University**: [University Name]")
st.markdown("- **Year**: 2026")

st.markdown("### References")
st.markdown("- [Placeholder for Academic References]")

render_footer()
