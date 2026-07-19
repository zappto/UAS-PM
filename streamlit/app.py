import streamlit as st
import pandas as pd
import numpy as np
import pathlib
import joblib
import json
import re
from PIL import Image

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False
    
try:
    from lime.lime_text import LimeTextExplainer
    import matplotlib.pyplot as plt
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

# ==========================================
# 1. CONFIGURATION & PATH RESOLUTION
# ==========================================
st.set_page_config(
    page_title="Cyberbullying Classification",
    page_icon="🛡️",
    layout="wide"
)

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TFIDF_DIR = DATA_DIR / "processed" / "tfidf"
ERROR_DIR = REPORTS_DIR / "error_analysis"
EXPLAIN_DIR = REPORTS_DIR / "explainability"

# ==========================================
# 2. CACHED HELPER FUNCTIONS
# ==========================================

@st.cache_resource
def init_sastrawi():
    if not SASTRAWI_AVAILABLE:
        return None, None
    stemmer = StemmerFactory().create_stemmer()
    stopword_factory = StopWordRemoverFactory()
    stopwords_list = stopword_factory.get_stop_words()
    
    # Preserve negation
    negation_words = ['tidak', 'bukan', 'jangan', 'belum', 'kurang']
    stopwords_list = [w for w in stopwords_list if w not in negation_words]
    
    return stemmer, stopwords_list

@st.cache_resource
def load_tfidf_vectorizer():
    vec_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    if vec_path.exists():
        return joblib.load(vec_path)
    return None

@st.cache_data
def load_model_selection_meta():
    meta_path = REPORTS_DIR / "model_selection.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            return json.load(f)
    return None

@st.cache_resource
def load_champion_model():
    meta = load_model_selection_meta()
    if meta:
        model_name = meta.get('selected_model')
        model_path = MODELS_DIR / f"{model_name}.pkl"
        if model_path.exists():
            return joblib.load(model_path), model_name
    return None, None

@st.cache_resource
def load_xgb_mapping():
    mapping_path = MODELS_DIR / "xgboost_label_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
            return {v: k for k, v in mapping.items()} # Reverse mapping
    return None

@st.cache_data
def load_csv(path):
    if pathlib.Path(path).exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_json(path):
    if pathlib.Path(path).exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None

# Text Preprocessing Function
def preprocess_text(text, stemmer, stopwords_list):
    if not SASTRAWI_AVAILABLE:
        return text.lower() # Fallback
        
    # 1. Case Folding
    text = text.lower()
    # 2. Cleansing
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # 3. Stopword Removal
    words = text.split()
    words = [w for w in words if w not in stopwords_list]
    # 4. Stemming
    text = stemmer.stem(' '.join(words))
    return text

# ==========================================
# 3. PAGE LOGIC
# ==========================================

def render_home():
    st.title("🛡️ Cyberbullying Text Classification")
    st.subheader("Indonesian Language NLP Research Project")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Research Objective
        This dashboard presents the findings of the research titled:
        > *"Analisis Performa Algoritma Machine Learning untuk Klasifikasi Jenis dan Tingkat Keparahan Cyberbullying pada Teks Bahasa Indonesia Menggunakan TF-IDF"*
        
        The goal is to automatically detect and classify Indonesian text into specific categories of cyberbullying to assist in digital moderation and linguistic research.
        """)
        
        st.markdown("""
        ### The Machine Learning Pipeline
        1. **Input**: Raw Indonesian Text
        2. **Preprocessing**: Case folding, URL/Mention removal, Sastrawi Stemming, Negation-preserved Stopword Removal.
        3. **Feature Extraction**: TF-IDF (Term Frequency - Inverse Document Frequency) with N-Grams.
        4. **Modeling**: Logistic Regression, LinearSVC, and XGBoost (Baseline & Tuned).
        5. **Output**: Cyberbullying Category Prediction & Interpretability.
        """)
        
    with col2:
        st.info("### Quick Stats")
        meta = load_model_selection_meta()
        if meta:
            st.success(f"**Champion Model:** {meta.get('selected_model')}")
            st.metric(label="Best F1-Macro Score", value=f"{meta.get('selected_f1_macro'):.4f}")
        else:
            st.warning("Model Selection data not available. Please complete Stage 09.")
            
        st.markdown("**Status:** Research Pipeline Completed.")

def render_eda():
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("Overview of the dataset structure and text characteristics before modeling.")
    st.markdown("---")
    
    # Try loading metadata to show current shape
    df_train = load_csv(TFIDF_DIR / "train_metadata.csv")
    if df_train is not None:
        st.success(f"Dataset active. Training samples: {len(df_train):,}")
        st.dataframe(df_train.head(5))
    else:
        st.info("EDA Artifacts are currently being processed or unavailable. Please complete Stage 02 and 06.")

def render_performance():
    st.title("📈 Model Performance & Evaluation")
    st.markdown("Comparing Baseline vs. Tuned models on unseen Test Data.")
    st.markdown("---")
    
    meta = load_model_selection_meta()
    if not meta:
        st.warning("Evaluation artifacts not found. Run `09_model_evaluation.ipynb` first.")
        return
        
    st.header("🏆 The Champion Model")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Name", meta.get('selected_model', 'Unknown'))
    with col2:
        st.metric("Test F1-Macro Score", f"{meta.get('selected_f1_macro', 0):.4f}")
        
    st.markdown("---")
    st.subheader("Model Comparison Leaderboard")
    
    eval_df = load_csv(REPORTS_DIR / "model_evaluation_metrics.csv")
    if eval_df is not None:
        st.dataframe(eval_df.style.highlight_max(subset=['F1_Macro', 'Accuracy'], color='lightgreen'))
    else:
        st.info("Comparison leaderboard not available.")

def render_prediction():
    st.title("🔮 Live Cyberbullying Prediction")
    st.markdown("Test the champion model against real-world Indonesian text.")
    st.markdown("---")
    
    model, model_name = load_champion_model()
    vectorizer = load_tfidf_vectorizer()
    stemmer, stopwords = init_sastrawi()
    xgb_mapping = load_xgb_mapping()
    
    if not model or not vectorizer:
        st.error("Missing Model or TF-IDF Vectorizer artifacts. Cannot perform prediction.")
        return
        
    if not SASTRAWI_AVAILABLE:
        st.warning("Sastrawi library not found. Preprocessing will be degraded (lowercase only).")
    
    with st.form("prediction_form"):
        user_input = st.text_area("Masukkan teks Bahasa Indonesia untuk dianalisis...", height=100)
        submitted = st.form_submit_button("Analyze Text")
        
    if submitted:
        if not user_input.strip():
            st.error("Teks tidak boleh kosong!")
            return
            
        with st.spinner("Analyzing text..."):
            # 1. Preprocess
            clean_text = preprocess_text(user_input, stemmer, stopwords)
            
            # 2. Vectorize
            X_vec = vectorizer.transform([clean_text])
            
            # 3. Predict
            pred_raw = model.predict(X_vec)[0]
            
            # Map back if XGBoost
            if 'xgboost' in model_name.lower() and xgb_mapping:
                predicted_class = xgb_mapping.get(pred_raw, str(pred_raw))
            else:
                predicted_class = pred_raw
                
            # 4. Probabilities
            probs = None
            classes = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_vec)[0]
                if 'xgboost' in model_name.lower() and xgb_mapping:
                    classes = [xgb_mapping.get(i, str(i)) for i in range(len(probs))]
                else:
                    classes = model.classes_
                    
        # 5. Display Result
        st.markdown("### Hasil Analisis")
        st.info(f"**Cleaned Text Internal:** '{clean_text}'")
        
        # Color coding based on prediction (assuming 'Non-bullying' might be a class)
        if str(predicted_class).lower() in ['non-bullying', 'bukan bullying', 'not bullying']:
            st.success(f"**Prediksi Kategori:** {predicted_class}")
        else:
            st.error(f"**Prediksi Kategori:** {predicted_class} 🚩")
            
        if probs is not None and classes is not None:
            st.markdown("**Tingkat Keyakinan (Probabilitas):**")
            prob_df = pd.DataFrame({'Kategori': classes, 'Probabilitas': probs})
            prob_df = prob_df.sort_values(by='Probabilitas', ascending=False)
            
            st.bar_chart(prob_df.set_index('Kategori'))

def render_error_analysis():
    st.title("🕵️ Error Analysis")
    st.markdown("Investigating the blind spots of the champion model.")
    st.markdown("---")
    
    error_summary = load_json(ERROR_DIR / "error_analysis_summary.json")
    if error_summary:
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Error Rate", f"{error_summary.get('overall_error_rate_percent')}%")
        col2.metric("Highest Error Class", error_summary.get('highest_error_class'))
        col3.metric("High-Confidence Errors", error_summary.get('high_confidence_errors_count'))
    
    st.subheader("Confusion Pairs (Most confused classes)")
    conf_pairs = load_csv(ERROR_DIR / "confusion_pairs.csv")
    if conf_pairs is not None:
        st.dataframe(conf_pairs.head(10))
    else:
        st.info("Artifacts not found. Run `10_error_analysis.ipynb`.")
        
    st.subheader("Potential Annotation Issues (High-Confidence Errors)")
    anno_issues = load_csv(ERROR_DIR / "potential_annotation_issues_for_review.csv")
    if anno_issues is not None:
        st.dataframe(anno_issues)

def render_explainability():
    st.title("🧠 Explainable AI (XAI)")
    st.markdown("Understanding *why* the model makes its decisions by decoding TF-IDF matrices.")
    st.markdown("---")
    
    st.header("Global Explainability")
    st.markdown("What words broadly influence the model?")
    
    try:
        global_img = Image.open(EXPLAIN_DIR / "top_words_per_class.png")
        st.image(global_img, caption="Top influential words per class")
    except FileNotFoundError:
        st.info("Global word visualization not available. Run `11_explainability.ipynb`.")
        
    class_imp = load_csv(EXPLAIN_DIR / "class_feature_importance.csv")
    if class_imp is not None:
        with st.expander("View Raw Importance Weights DataFrame"):
            st.dataframe(class_imp)

def render_about():
    st.title("ℹ️ About the Research")
    st.markdown("---")
    st.markdown("""
    ### Methodology
    This project strictly adheres to a classical Machine Learning NLP pipeline:
    - **Representation**: TF-IDF was chosen for its interpretability and robust performance on domain-specific vocabulary compared to dense embeddings.
    - **Models**: Explored both Linear architectures (for speed and exact weight mapping) and Tree-based architectures (XGBoost) for complex non-linear feature interactions.
    - **Validation**: Stratified K-Fold Cross Validation was used to ensure the metrics are reliable across imbalanced classes.
    
    ### Limitations
    - **Context Blindness**: Because the feature extraction relies on TF-IDF N-grams, the model struggles with complex sarcasm or long-distance semantic dependencies that a Transformer (like IndoBERT) might catch.
    - **Static Vocabulary**: Words that were not present in the training set will simply be ignored during live prediction (Out of Vocabulary problem).
    
    *Built with Streamlit & Scikit-Learn.*
    """)

# ==========================================
# 4. APP ROUTING
# ==========================================
def main():
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    pages = {
        "🏠 Home / Overview": render_home,
        "📊 EDA Dashboard": render_eda,
        "📈 Model Performance": render_performance,
        "🔮 Cyberbullying Prediction": render_prediction,
        "🕵️ Error Analysis": render_error_analysis,
        "🧠 Explainability": render_explainability,
        "ℹ️ About the Research": render_about
    }
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Research Pipeline Demo")
    
    # Execute selected page
    pages[selection]()

if __name__ == "__main__":
    main()
