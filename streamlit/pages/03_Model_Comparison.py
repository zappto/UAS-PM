"""Model Comparison Page.

Displays interactive table and charts from the existing evaluation reports.
"""
import streamlit as st
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from components.sidebar import render_sidebar
from components.footer import render_footer
from utils.constants import MODEL_COMPARISON_PATH

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")
render_sidebar()

st.title("Model Comparison")
st.write("Review the evaluation metrics across all trained models.")

if os.path.exists(MODEL_COMPARISON_PATH):
    df = pd.read_csv(MODEL_COMPARISON_PATH)
    
    st.markdown("### Interactive Table")
    st.dataframe(df, use_container_width=True)
    
    st.markdown("### Performance Charts")
    
    # Check if df has required columns (e.g., 'Model', 'F1-Score', 'Accuracy')
    # Typically, reports contain these. We make a safe plotting assumption.
    model_col = next((col for col in df.columns if col.lower() in ['model', 'algorithm']), None)
    metric_cols = [col for col in df.columns if col.lower() in ['f1-score', 'accuracy', 'precision', 'recall', 'f1_score', 'f1']]
    
    if model_col and metric_cols:
        primary_metric = metric_cols[0]
        st.write(f"Comparing {primary_metric} across models:")
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(df[model_col], df[primary_metric], color="#4F8BF9")
        ax.set_ylabel(primary_metric)
        ax.set_title(f"{primary_metric} by Model")
        
        # Add labels
        for i, v in enumerate(df[primary_metric]):
            ax.text(i, v - 0.05 if v > 0.1 else v + 0.05, f"{v:.3f}", ha='center', color='white' if v > 0.1 else 'black')
            
        st.pyplot(fig)
    else:
        st.info("Insufficient columns for automated charting. Please check the CSV format.")
else:
    st.warning("Model comparison report not found. Please ensure the evaluation phase has been executed.")

render_footer()
