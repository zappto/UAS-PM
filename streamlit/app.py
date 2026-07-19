import streamlit as st
import streamlit.components.v1 as components
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
    import seaborn as sns
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
def load_model_by_name(model_name):
    model_path = MODELS_DIR / f"{model_name}.pkl"
    if model_path.exists():
        return joblib.load(model_path)
    return None

@st.cache_resource
def load_all_models():
    # Load 3 main models for comparison
    models = {}
    for name in ["logistic_regression_type_tuned", "linear_svm_type_tuned", "xgboost_type_tuned"]:
        m = load_model_by_name(name)
        if not m:
            # Fallback to baseline
            baseline_name = name.replace("_type_tuned", "_baseline")
            m = load_model_by_name(baseline_name)
            if m:
                models[baseline_name] = m
        else:
            models[name] = m
    return models

@st.cache_resource
def load_xgb_mapping():
    mapping_path = MODELS_DIR / "xgboost_label_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
            return {v: k for k, v in mapping.items()} 
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

def preprocess_text(text, stemmer, stopwords_list):
    if not SASTRAWI_AVAILABLE:
        return text.lower() 
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in stopwords_list]
    text = stemmer.stem(' '.join(words))
    return text

def get_probabilities(model, X_vec):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_vec)[0]
    elif hasattr(model, "decision_function"):
        dec = model.decision_function(X_vec)[0]
        # Simulate probabilities using softmax
        exp_d = np.exp(dec - np.max(dec))
        return exp_d / np.sum(exp_d)
    return None

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
    with col2:
        st.info("### Quick Stats")
        meta = load_model_selection_meta()
        if meta:
            st.success(f"**Champion Model:** {meta.get('selected_model')}")
            f1 = meta.get('final_metrics', {}).get('f1_score', 0)
            st.metric(label="Best F1-Macro Score", value=f"{f1:.4f}")
        else:
            st.warning("Model Selection data not available. Please complete Stage 09.")
            
def render_eda():
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("---")
    df_train = load_csv(TFIDF_DIR / "train_metadata.csv")
    if df_train is not None:
        st.success(f"Dataset active. Training samples: {len(df_train):,}")
        st.dataframe(df_train.head(5))
    else:
        st.info("EDA Artifacts are currently being processed or unavailable.")

def render_performance():
    st.title("📈 Model Performance & Evaluation")
    st.markdown("---")
    meta = load_model_selection_meta()
    if not meta:
        st.warning("Evaluation artifacts not found. Run `09_model_evaluation.ipynb` first.")
        return
        
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Name", meta.get('selected_model', 'Unknown'))
    with col2:
        f1 = meta.get('final_metrics', {}).get('f1_score', 0)
        st.metric("Test F1-Macro Score", f"{f1:.4f}")
        
    st.markdown("---")
    eval_df = load_csv(REPORTS_DIR / "model_evaluation_metrics.csv")
    if eval_df is not None:
        st.dataframe(eval_df.style.highlight_max(subset=['F1_Macro', 'Accuracy'], color='lightgreen'))
    else:
        st.info("Comparison leaderboard not available.")

def render_prediction():
    st.title("🔬 Text Analysis & Prediction")
    st.markdown("---")
    
    # Load Artifacts
    models = load_all_models()
    vectorizer = load_tfidf_vectorizer()
    stemmer, stopwords = init_sastrawi()
    xgb_mapping = load_xgb_mapping()
    meta = load_model_selection_meta()
    
    if not models or not vectorizer:
        st.error("Required model artifact not found. Please complete the previous pipeline stage first.")
        return
        
    champ_model_name = meta.get('selected_model') if meta else list(models.keys())[0]
    champ_model = models.get(champ_model_name, list(models.values())[0])
    
    st.header("TEXT ANALYSIS")
    with st.form("prediction_form"):
        user_input = st.text_area("Masukkan teks Bahasa Indonesia yang ingin dianalisis...", height=150)
        submitted = st.form_submit_button("Analisis Teks")
        
    if submitted:
        if not user_input or not user_input.strip():
            st.warning("Teks tidak boleh kosong! Silakan masukkan teks yang valid.")
            return
            
        with st.spinner("Memproses teks dan menjalankan komputasi model..."):
            
            # --- SECTION 2: ORIGINAL TEXT ---
            st.markdown("---")
            st.subheader("ORIGINAL TEXT")
            st.info(user_input)
            
            # --- SECTION 3: PREPROCESSING PREVIEW ---
            st.markdown("---")
            st.subheader("PREPROCESSING PREVIEW")
            clean_text = preprocess_text(user_input, stemmer, stopwords)
            st.success(clean_text)
            
            if not clean_text.strip():
                st.error("Semua kata dibuang saat preprocessing (hanya berisi simbol/stopword). Analisis dihentikan.")
                return
            
            # --- SECTION 4: TF-IDF REPRESENTATION ---
            st.markdown("---")
            st.subheader("TF-IDF REPRESENTATION")
            X_vec = vectorizer.transform([clean_text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract non-zero elements
            non_zero_indices = X_vec.nonzero()[1]
            if len(non_zero_indices) == 0:
                st.warning("Tidak ada kata dalam teks Anda yang dikenali oleh model (Out of Vocabulary).")
            else:
                tfidf_data = []
                for idx in non_zero_indices:
                    tfidf_data.append({
                        "Feature": feature_names[idx],
                        "TF-IDF Score": X_vec[0, idx]
                    })
                df_tfidf = pd.DataFrame(tfidf_data).sort_values(by="TF-IDF Score", ascending=False).head(10)
                st.write(f"Total fitur dikenali: **{len(non_zero_indices)}** (Menampilkan maksimal top 10 fitur tertinggi)")
                st.table(df_tfidf.set_index("Feature"))
            
            # Prepare Model Mapping
            is_xgb_champ = 'xgboost' in champ_model_name.lower()
            if is_xgb_champ and xgb_mapping:
                class_names = [xgb_mapping.get(i, str(i)) for i in range(len(xgb_mapping))]
            else:
                class_names = list(champ_model.classes_)
            
            pred_raw = champ_model.predict(X_vec)[0]
            if is_xgb_champ and xgb_mapping:
                predicted_class = xgb_mapping.get(pred_raw, str(pred_raw))
            else:
                predicted_class = pred_raw
                
            probs = get_probabilities(champ_model, X_vec)
            confidence = np.max(probs) if probs is not None else None
            
            # --- SECTION 5: PREDICTION RESULT ---
            st.markdown("---")
            st.subheader("PREDICTION RESULT")
            st.markdown(f"**Predicted Type:** `{predicted_class}`")
            st.markdown(f"**Model:** `{champ_model_name}`")
            if confidence is not None:
                st.markdown(f"**Model Confidence:** `{confidence*100:.2f}%`")
                
            # --- SECTION 7: PROBABILITY DISTRIBUTION ---
            st.markdown("---")
            st.subheader("PROBABILITY DISTRIBUTION")
            if probs is not None:
                prob_df = pd.DataFrame({'Class': class_names, 'Probability': probs})
                prob_df = prob_df.sort_values(by='Probability', ascending=True)
                
                fig_prob, ax_prob = plt.subplots(figsize=(8, max(3, len(class_names)*0.5)))
                ax_prob.barh(prob_df['Class'], prob_df['Probability'] * 100, color='skyblue')
                ax_prob.set_xlabel('Probability (%)')
                st.pyplot(fig_prob)
            else:
                st.info("Probability distribution is not available for this model.")
                
            # --- SECTION 9/10: MODEL EXPLANATION (LIME) ---
            st.markdown("---")
            st.subheader("MODEL EXPLANATION")
            
            if LIME_AVAILABLE and len(non_zero_indices) > 0:
                def lime_pipeline(texts):
                    X = vectorizer.transform(texts)
                    if hasattr(champ_model, "predict_proba"):
                        return champ_model.predict_proba(X)
                    else:
                        dec = champ_model.decision_function(X)
                        exp_d = np.exp(dec - np.max(dec, axis=1, keepdims=True))
                        return exp_d / np.sum(exp_d, axis=1, keepdims=True)

                explainer = LimeTextExplainer(class_names=class_names)
                exp = explainer.explain_instance(clean_text, lime_pipeline, num_features=10, top_labels=1)
                
                pred_label_idx = exp.available_labels()[0]
                
                # Render HTML LIME in Streamlit
                components.html(exp.as_html(), height=350, scrolling=True)
                
                st.markdown("#### Fitur yang Berkontribusi terhadap Prediksi")
                word_weights = exp.as_list(label=pred_label_idx)
                if word_weights:
                    df_weights = pd.DataFrame(word_weights, columns=['Feature', 'Contribution'])
                    df_weights['Contribution'] = df_weights['Contribution'].apply(lambda x: f"{x:+.4f}")
                    st.table(df_weights.set_index('Feature'))
                
                # --- SECTION 11: INTERPRETATION TEXT ---
                st.markdown("---")
                st.subheader("INTERPRETATION")
                top_features = ", ".join([w[0] for w in word_weights[:3]]) if word_weights else "tidak ada fitur kuat yang terdeteksi"
                conf_text = f"dengan tingkat probabilitas {confidence*100:.2f}%" if confidence is not None else ""
                st.info(f"Model memprediksi kategori **{predicted_class}** {conf_text}. Prediksi ini secara signifikan dipengaruhi oleh kehadiran kata/fitur seperti **{top_features}**. Perlu dicatat bahwa eksplanasi fitur ini mencerminkan asosiasi matematis yang dipelajari oleh algoritma dari data latih, dan tidak selalu menyiratkan hubungan sebab-akibat secara langsung dalam bahasa manusia.")
            else:
                st.info("LIME Explainability tidak dapat dijalankan (Mungkin OOV atau library tidak tersedia).")

def render_error_analysis():
    st.title("🕵️ Error Analysis")
    st.markdown("---")
    error_summary = load_json(ERROR_DIR / "error_analysis_summary.json")
    if error_summary:
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Error Rate", f"{error_summary.get('overall_error_rate_percent')}%")
        col2.metric("Highest Error Class", error_summary.get('highest_error_class'))
        col3.metric("High-Confidence Errors", error_summary.get('high_confidence_errors_count'))

def render_explainability():
    st.title("🧠 Global Explainability")
    st.markdown("---")
    try:
        global_img = Image.open(EXPLAIN_DIR / "top_words_per_class.png")
        st.image(global_img, caption="Top influential words per class")
    except FileNotFoundError:
        st.info("Global word visualization not available. Run `11_explainability.ipynb`.")

def render_about():
    st.title("ℹ️ About the Research")
    st.markdown("---")
    st.markdown("This project strictly adheres to a classical Machine Learning NLP pipeline using TF-IDF.")

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
        "🔮 Prediction & XAI": render_prediction,
        "🕵️ Error Analysis": render_error_analysis,
        "🧠 Global Explainability": render_explainability,
        "ℹ️ About the Research": render_about
    }
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    st.sidebar.markdown("---")
    st.sidebar.caption("Research Pipeline Demo")
    
    pages[selection]()

if __name__ == "__main__":
    main()
