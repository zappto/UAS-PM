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
    st.title("🔬 Deep Live Prediction & Explainability Analysis")
    st.markdown("Analisis mendalam teks menggunakan **perbandingan 3 model sekaligus** dan bedah kata (XAI LIME).")
    st.markdown("---")
    
    models = load_all_models()
    vectorizer = load_tfidf_vectorizer()
    stemmer, stopwords = init_sastrawi()
    xgb_mapping = load_xgb_mapping()
    meta = load_model_selection_meta()
    
    if not models or not vectorizer:
        st.error("Missing Models or Vectorizer.")
        return
        
    champ_model_name = meta.get('selected_model') if meta else list(models.keys())[0]
    
    with st.form("prediction_form"):
        user_input = st.text_area("Masukkan teks Bahasa Indonesia untuk dianalisis...", height=100)
        submitted = st.form_submit_button("Analisis Mendalam 🚀")
        
    if submitted:
        if not user_input.strip():
            st.error("Teks tidak boleh kosong!")
            return
            
        with st.spinner("Memproses teks, menjalankan komputasi 3 model, dan membangun LIME Explainer..."):
            clean_text = preprocess_text(user_input, stemmer, stopwords)
            X_vec = vectorizer.transform([clean_text])
            
            # --- 1. MODEL COMPARISON ---
            st.subheader("1. Perbandingan Keputusan Model")
            st.info(f"**Teks setelah dibersihkan (Preprocessed):** '{clean_text}'")
            
            comp_data = []
            for m_name, m in models.items():
                pred_raw = m.predict(X_vec)[0]
                is_xgb = 'xgboost' in m_name.lower()
                
                # Class Mapping
                if is_xgb and xgb_mapping:
                    predicted_class = xgb_mapping.get(pred_raw, str(pred_raw))
                    classes = [xgb_mapping.get(i, str(i)) for i in range(len(xgb_mapping))]
                else:
                    predicted_class = pred_raw
                    classes = list(m.classes_)
                    
                # Probs
                probs = get_probabilities(m, X_vec)
                confidence = f"{np.max(probs)*100:.2f}%" if probs is not None else "N/A"
                
                is_champ = "🏆 Champion" if m_name == champ_model_name else ""
                comp_data.append({
                    "Model": f"{m_name} {is_champ}",
                    "Prediksi Kelas": predicted_class,
                    "Confidence": confidence
                })
                
            comp_df = pd.DataFrame(comp_data)
            st.table(comp_df)
            
            # --- 2. XAI LIME ANALYSIS (On Champion Model) ---
            st.markdown("---")
            st.subheader(f"2. Analisis LIME (Model Pemenang: `{champ_model_name}`)")
            
            if LIME_AVAILABLE and len(clean_text.split()) > 0:
                champ_model = models.get(champ_model_name, list(models.values())[0])
                is_xgb_champ = 'xgboost' in champ_model_name.lower()
                
                if is_xgb_champ and xgb_mapping:
                    class_names = [xgb_mapping.get(i, str(i)) for i in range(len(xgb_mapping))]
                else:
                    class_names = list(champ_model.classes_)
                
                # LIME needs a pipeline function
                def lime_pipeline(texts):
                    X = vectorizer.transform(texts)
                    if hasattr(champ_model, "predict_proba"):
                        return champ_model.predict_proba(X)
                    else:
                        # Fallback for SVM
                        dec = champ_model.decision_function(X)
                        exp_d = np.exp(dec - np.max(dec, axis=1, keepdims=True))
                        return exp_d / np.sum(exp_d, axis=1, keepdims=True)

                explainer = LimeTextExplainer(class_names=class_names)
                # Explain the cleaned text
                exp = explainer.explain_instance(clean_text, lime_pipeline, num_features=10, top_labels=1)
                pred_label_idx = exp.available_labels()[0]
                pred_label_name = class_names[pred_label_idx]
                
                st.markdown(f"**Prediksi Akhir Model Pemenang:** `{pred_label_name}`")
                
                # Render HTML LIME in Streamlit
                components.html(exp.as_html(), height=350, scrolling=True)
                
                # Render Bar Chart
                st.markdown("### Beban Kata (Feature Weights)")
                st.write(f"Grafik di bawah menunjukkan kata apa yang mendorong model memprediksi **{pred_label_name}** (Positif/Biru) dan kata apa yang justru membantah prediksi tersebut (Negatif/Oranye).")
                
                word_weights = exp.as_list(label=pred_label_idx)
                if word_weights:
                    df_weights = pd.DataFrame(word_weights, columns=['Kata', 'Bobot'])
                    df_weights['Arah'] = df_weights['Bobot'].apply(lambda x: 'Mendukung Prediksi' if x > 0 else 'Membantah Prediksi')
                    
                    fig, ax = plt.subplots(figsize=(8, max(4, len(df_weights)*0.5)))
                    sns.barplot(data=df_weights, x='Bobot', y='Kata', hue='Arah', palette={'Mendukung Prediksi': 'royalblue', 'Membantah Prediksi': 'coral'}, ax=ax)
                    ax.set_title(f"Kontribusi Kata terhadap prediksi '{pred_label_name}'")
                    st.pyplot(fig)
            elif len(clean_text.split()) == 0:
                st.warning("Teks setelah di-preprocessing kosong (semua kata dibuang karena dianggap tidak bermakna/stopword).")
            else:
                st.warning("Library LIME tidak tersedia. Harap install dengan `pip install lime`.")

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
