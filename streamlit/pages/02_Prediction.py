"""Prediction Page.

Interactive prediction and explainability visualization.
"""
import streamlit as st
import sys
import os
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from components.sidebar import render_sidebar
from components.footer import render_footer
from components.prediction_card import render_prediction_card
from components.probability_chart import render_probability_chart
from components.highlighted_text import render_highlighted_text
from components.prediction_history import render_prediction_history
from utils.constants import EXAMPLE_TEXTS, LABEL_PREDICT_BUTTON, LABEL_CLEAR_BUTTON, LABEL_DOWNLOAD_JSON
from utils.session import load_pipeline, initialize_session_state, add_to_history, reset_session

st.set_page_config(page_title="Prediction", page_icon="🎯", layout="wide")
render_sidebar()
initialize_session_state()

# Load Model Pipeline
pipeline = load_pipeline()

st.title("Interactive Prediction")
st.write("Enter text below to classify its cyberbullying type and view explainability insights.")

# Examples Row
st.markdown("**Try an Example:**")
cols = st.columns(len(EXAMPLE_TEXTS))
for idx, (label, text) in enumerate(EXAMPLE_TEXTS.items()):
    if cols[idx].button(label):
        st.session_state.current_input_text = text

# Input Area
input_text = st.text_area("Input Text", value=st.session_state.current_input_text, height=150)

# Controls
col1, col2, col3 = st.columns([2, 1, 1])
predict_btn = col1.button(LABEL_PREDICT_BUTTON, type="primary", use_container_width=True)

if col2.button(LABEL_CLEAR_BUTTON, use_container_width=True):
    st.session_state.current_input_text = ""
    st.rerun()

if col3.button("Reset Session", use_container_width=True):
    reset_session()
    st.rerun()

st.markdown("---")

if predict_btn:
    if not input_text.strip():
        st.warning("Please enter some text to predict.")
    else:
        with st.spinner("Analyzing text..."):
            try:
                # 1. API Call
                result = pipeline.predict_with_explanation(input_text)
                
                # 2. Add to History
                add_to_history(input_text, result)
                
                # 3. Main Result
                render_prediction_card(
                    prediction=result["prediction"],
                    confidence=result["confidence"],
                    inference_time=result["inference_time"]
                )
                
                # 4. Charts and Highlights
                colA, colB = st.columns(2)
                with colA:
                    render_probability_chart(result["probabilities"])
                with colB:
                    render_highlighted_text(input_text, result["highlighted_tokens"])
                
                # 5. Top Words
                st.markdown("### Top Influential Words")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Positive Contribution (Pushed towards prediction):**")
                    for w in result["top_positive_words"]:
                        st.markdown(f"- `{w['word']}` ({w['importance']:.3f})")
                with c2:
                    st.markdown("**Negative Contribution (Pushed away from prediction):**")
                    for w in result["top_negative_words"]:
                        st.markdown(f"- `{w['word']}` ({w['importance']:.3f})")
                
                # 6. Export Options
                st.markdown("---")
                json_data = json.dumps(result, indent=2)
                st.download_button(
                    label=LABEL_DOWNLOAD_JSON,
                    data=json_data,
                    file_name="prediction_result.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# History
st.markdown("---")
render_prediction_history()

render_footer()
