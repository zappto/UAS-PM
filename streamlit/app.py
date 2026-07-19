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
    import plotly.express as px
    import plotly.graph_objects as go
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
BASE_REPORTS_DIR = PROJECT_ROOT / "reports"

TFIDF_DIR = DATA_DIR / "processed" / "tfidf"
ERROR_DIR = BASE_REPORTS_DIR / "error_analysis"
EXPLAIN_DIR = BASE_REPORTS_DIR / "explainability"

# ==========================================
# 2. CACHED HELPER FUNCTIONS
# ==========================================

@st.cache_resource
def load_lexicons():
    def load_lex(filename):
        path = DATA_DIR / "raw" / filename
        if path.exists():
            df = pd.read_csv(path)
            col = df.columns[0]
            return set(df[col].dropna().astype(str).str.lower().str.strip())
        return set()
    
    abusive_lex = load_lex('abusive.csv')
    harassment_lex = load_lex('harassment.csv')
    insult_lex = load_lex('insult.csv')
    threat_lex = load_lex('threat.csv')
    return (abusive_lex, harassment_lex, insult_lex, threat_lex)

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
    meta_path = BASE_REPORTS_DIR / "model_selection.json"
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

def preprocess_text(text, stemmer, stopwords_list, lexicons):
    if not SASTRAWI_AVAILABLE:
        return text.lower() 
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    
    # Inject Lexicon Tags
    tags = []
    abusive_lex, harassment_lex, insult_lex, threat_lex = lexicons
    for w in words:
        if w in abusive_lex: tags.append('tagabusive')
        if w in harassment_lex: tags.append('tagharassment')
        if w in insult_lex: tags.append('taginsult')
        if w in threat_lex: tags.append('tagthreat')
        
    words = [w for w in words if w not in stopwords_list]
    words.extend(tags)
    
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
    st.markdown("### Indonesian Language NLP Research Project Dashboard")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Selamat datang di Dasbor Penelitian Analisis Sentimen!**
        
        Sistem ini mendemonstrasikan kapabilitas kecerdasan buatan (*Machine Learning*) dalam mendeteksi dan mengklasifikasikan teks perundungan siber (*cyberbullying*) ke dalam **5 kategori spesifik**:
        - 🟢 **Normal**: Teks aman tanpa indikasi pelecehan.
        - 🟡 **Harassment**: Pelecehan/Pelecehan Seksual.
        - 🟠 **Insult**: Penghinaan atau cacian personal.
        - 🔴 **Abusive**: Penggunaan bahasa kasar yang merendahkan.
        - 💀 **Threat**: Ancaman fisik atau psikologis.

        **⚙️ Arsitektur Sistem (Pipeline):**
        1. **Data Preprocessing**: Pembersihan *Noise* & injeksi leksikon Sastrawi.
        2. **Feature Engineering**: Ekstraksi bobot kata via *Term Frequency - Inverse Document Frequency* (TF-IDF).
        3. **Classification Engine**: Support Vector Machine (Linear SVM).
        4. **Explainable AI (XAI)**: Transparansi logika model menggunakan algoritma LIME.
        """)
    with col2:
        st.info("📊 **Statistik Utama Riset**")
        meta = load_model_selection_meta()
        if meta:
            st.success(f"**🏅 Champion Model:**\n\n{meta.get('selected_model')}")
            f1 = meta.get('final_metrics', {}).get('f1_score', 0)
            st.metric(label="🏆 Best F1-Macro Score", value=f"{f1*100:.2f}%")
            
            acc = meta.get('final_metrics', {}).get('accuracy', 0)
            if acc == 0: # Fallback if accuracy isn't explicitly in final_metrics but rather model_evaluation_results.csv
                eval_df = load_csv(BASE_REPORTS_DIR / "model_evaluation_results.csv")
                if eval_df is not None and not eval_df.empty:
                    acc = eval_df.loc[eval_df['Model'] == meta.get('selected_model'), 'Accuracy'].values[0]
            st.metric(label="🎯 Overall Accuracy", value=f"{acc*100:.2f}%")
        else:
            st.warning("Model Selection data not available. Please complete Stage 09.")
            
def render_eda():
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("---")
    
    st.markdown("""
    Analisis data eksploratif ini menyoroti karakteristik asli dataset, termasuk distribusi kelas yang tidak seimbang (*Class Imbalance*) dan distribusi panjang kata yang diketik oleh pengguna internet berbahasa Indonesia.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Kelas (Class Imbalance)")
        try:
            img1 = Image.open(BASE_REPORTS_DIR / "eda" / "class_distribution.png")
            st.image(img1, use_column_width=True)
        except Exception:
            st.info("Gambar class_distribution.png belum tersedia.")
            
    with col2:
        st.subheader("Distribusi Panjang Kata")
        try:
            img2 = Image.open(BASE_REPORTS_DIR / "eda" / "word_length_distribution.png")
            st.image(img2, use_column_width=True)
        except Exception:
            st.info("Gambar word_length_distribution.png belum tersedia.")

    st.markdown("---")
    st.subheader("Sampel Data Mentah")
    df_train = load_csv(TFIDF_DIR / "train_metadata.csv")
    if df_train is not None:
        st.dataframe(df_train.head(10))
    else:
        st.info("EDA Artifacts are currently being processed or unavailable.")

def render_performance():
    st.title("📈 Model Performance & Evaluation")
    st.markdown("---")
    meta = load_model_selection_meta()
    
    if not meta:
        st.warning("Evaluation artifacts not found. Run `09_model_evaluation.ipynb` first.")
        return
        
    st.markdown("### 🏆 Champion Model")
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Model Name:** {meta.get('selected_model', 'Unknown')}")
    with col2:
        f1 = meta.get('final_metrics', {}).get('f1_score', 0)
        st.success(f"**Test F1-Macro Score:** {f1:.4f}")
        
    st.info(f"**Alasan Pemilihan:** {meta.get('reason', 'Tidak ada data')}")
        
    st.markdown("---")
    st.subheader("Leaderboard Algoritma")
    eval_df = load_csv(BASE_REPORTS_DIR / "model_evaluation_results.csv")
    if eval_df is not None:
        st.dataframe(
            eval_df.style.format({
                'Accuracy': '{:.4f}', 
                'Precision': '{:.4f}', 
                'Recall': '{:.4f}', 
                'F1-Score': '{:.4f}'
            }).highlight_max(subset=['F1-Score', 'Accuracy'], color='#2e7d32'), 
            use_container_width=True
        )
        
        # Plot comparison
        st.subheader("Grafik Perbandingan F1-Score")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='F1-Score', y='Model', data=eval_df.sort_values('F1-Score', ascending=False), palette='magma', ax=ax)
        ax.set_title("Komparasi F1-Macro antar Model")
        st.pyplot(fig)
    else:
        st.info("Comparison leaderboard not available.")
        
    st.markdown("---")
    st.subheader("Confusion Matrix (Model Terbaik)")
    try:
        best_model_name = meta.get('selected_model', 'linear_svm_baseline')
        img_cm = Image.open(BASE_REPORTS_DIR / f"confusion_matrix_{best_model_name}.png")
        st.image(img_cm, width=700)
    except Exception:
        st.info("Confusion Matrix tidak ditemukan.")

def render_prediction():
    st.title("🔬 Text Analysis & Prediction")
    st.markdown("---")
    
    # Load Artifacts
    models = load_all_models()
    vectorizer = load_tfidf_vectorizer()
    stemmer, stopwords = init_sastrawi()
    lexicons = load_lexicons()
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
            clean_text = preprocess_text(user_input, stemmer, stopwords, lexicons)
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
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown(f"#### Hasil Analisis: `{predicted_class.upper()}`")
                st.markdown(f"**Mesin Inferensi:** `{champ_model_name}`")
            
            with res_col2:
                if confidence is not None:
                    # Modern Plotly Gauge Chart for Confidence
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Tingkat Keyakinan Model (Confidence)", 'font': {'size': 14}},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#2e7d32" if confidence > 0.7 else "#f57c00"},
                            'steps': [
                                {'range': [0, 50], 'color': "rgba(255, 0, 0, 0.1)"},
                                {'range': [50, 75], 'color': "rgba(255, 165, 0, 0.1)"},
                                {'range': [75, 100], 'color': "rgba(0, 255, 0, 0.1)"}],
                        }
                    ))
                    fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
            # --- SECTION 7: PROBABILITY DISTRIBUTION ---
            st.markdown("---")
            st.subheader("PROBABILITY DISTRIBUTION")
            if probs is not None:
                prob_df = pd.DataFrame({'Class': class_names, 'Probability': probs})
                prob_df = prob_df.sort_values(by='Probability', ascending=True)
                
                fig_prob = px.bar(prob_df, x='Probability', y='Class', orientation='h', 
                                  title="Distribusi Probabilitas antar Kelas",
                                  color='Probability', color_continuous_scale='viridis')
                fig_prob.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_prob, use_container_width=True)
            else:
                st.info("Probability distribution is not available for this model.")
                
            # --- SECTION 9/10: MODEL EXPLANATION (LIME) ---
            st.markdown("---")
            st.subheader("MODEL EXPLANATION (LIME)")
            st.markdown("Kecerdasan Buatan (AI) bukanlah '*Black Box*'. Melalui pustaka **LIME (Local Interpretable Model-Agnostic Explanations)**, sistem ini secara transparan membongkar kata apa yang mendikte keputusan AI. Lihat grafik di bawah untuk memahami mengapa model menjatuhkan vonis tersebut.")
            
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
                word_weights = exp.as_list(label=pred_label_idx)
                
                # --- Modern Plotly Visualization for LIME ---
                if word_weights:
                    df_weights = pd.DataFrame(word_weights, columns=['Feature', 'Contribution'])
                    # Determine direction for coloring
                    df_weights['Arah'] = df_weights['Contribution'].apply(
                        lambda x: 'Mendukung Prediksi' if x > 0 else 'Bertentangan dengan Prediksi'
                    )
                    
                    fig = px.bar(
                        df_weights, 
                        x='Contribution', 
                        y='Feature', 
                        color='Arah',
                        orientation='h',
                        color_discrete_map={
                            'Mendukung Prediksi': '#2e7d32', # Forest Green
                            'Bertentangan dengan Prediksi': '#c62828' # Deep Red
                        },
                        title=f"Analisis Kontribusi Kata terhadap Kelas '{predicted_class}'"
                    )
                    
                    fig.update_layout(
                        yaxis={'categoryorder':'total ascending'},
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=14)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("LIME tidak dapat menemukan fitur yang signifikan pada kalimat ini.")
                
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
    st.markdown("Di halaman ini, kita membongkar secara objektif titik-titik kelemahan algoritma kecerdasan buatan. Model ML tidak pernah sempurna; memahami di mana model tersebut gagal adalah esensi dari penelitian Data Science.")
    st.markdown("---")
    
    error_summary = load_json(ERROR_DIR / "error_analysis_summary.json")
    if error_summary:
        st.subheader("Ringkasan Metrik Kegagalan")
        col1, col2, col3 = st.columns(3)
        col1.error(f"❌ Overall Error Rate\n\n# {error_summary.get('overall_error_rate_percent')}%")
        col2.warning(f"⚠️ Kelas Paling Rentan Meleset\n\n# {error_summary.get('highest_error_class')}")
        col3.info(f"🤔 High-Confidence Errors\n\n# {error_summary.get('high_confidence_errors_count')} Kasus")
        
    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Distribusi Error berdasarkan Kelas")
        st.markdown("Grafik ini menunjukkan persentase kalimat pada kelas tertentu yang gagal dikenali oleh sistem (salah tebak kelas).")
        try:
            img_err1 = Image.open(ERROR_DIR / "error_rate_by_class.png")
            st.image(img_err1, use_column_width=True)
        except Exception:
            st.info("Gambar error rate tidak ditemukan.")
            
    with colB:
        st.subheader("Korelasi Panjang Kata vs Error")
        st.markdown("Apakah kalimat yang terlalu panjang (atau terlalu pendek) menyebabkan AI kebingungan? Grafik ini menjawab fenomena tersebut.")
        try:
            img_err2 = Image.open(ERROR_DIR / "word_count_vs_errors.png")
            st.image(img_err2, use_column_width=True)
        except Exception:
            st.info("Gambar word count vs errors tidak ditemukan.")

def render_explainability():
    st.title("🧠 Global Explainability")
    st.markdown("Analisis XAI (*Explainable AI*) secara global berfungsi untuk memetakan '*kosakata otak*' dari model TF-IDF yang telah dilatih. Kata apa saja yang secara matematis menjadi pemicu terkuat untuk masing-masing kelas perundungan?")
    st.markdown("---")
    
    try:
        global_img = Image.open(EXPLAIN_DIR / "top_words_per_class.png")
        st.image(global_img, caption="Bobot Kosakata Global (Top Influential Words per Class)", use_column_width=True)
        
        st.info("""
        **Cara Membaca Grafik:**
        - Tiap panel merepresentasikan satu kelas sentimen.
        - Semakin panjang balok kata ke arah kanan (positif), semakin besar pengaruh kata tersebut dalam memicu model untuk memilih kelas tersebut.
        - Sebaliknya, kata dengan nilai negatif berarti kemunculan kata tersebut justru "menjauhkan" model dari kelas tersebut.
        """)
    except FileNotFoundError:
        st.info("Global word visualization not available. Run `11_explainability.ipynb`.")

def render_about():
    st.title("ℹ️ About & System Analysis")
    st.markdown("Proyek ini merepresentasikan perpaduan komprehensif antara rekayasa fitur (*Feature Engineering*) klasik dan teknik transparansi kecerdasan buatan (*Explainable AI*).")
    st.markdown("---")
    
    st.header("Kelebihan dan Kekurangan Sistem")
    st.markdown("Berdasarkan evaluasi arsitektur dan hasil performa, berikut adalah analisis objektif mengenai kapabilitas sistem ini:")
    
    col_pro, col_con = st.columns(2)
    
    with col_pro:
        st.success("### ✅ Kelebihan (Strengths)")
        st.markdown("""
        1. **Injeksi Pengetahuan (Lexicon Integration):**  
           Pendekatan hibrida yang menyuntikkan kamus leksikon (Sastrawi & kamus pelecehan) berhasil mengatasi kelemahan mendasar dari model statistik (TF-IDF) yang pada dasarnya buta terhadap sentimen lokal/bahasa gaul.
        2. **Efisiensi Komputasi (Resource Friendly):**  
           Sistem dioptimalkan untuk menekan kebocoran memori (RAM) melalui pembatasan fitur TF-IDF (maks. 60.000) dan utilitas CPU (*GridSearch `n_jobs`*). Model ini sangat ringan dan cepat dieksekusi bahkan pada perangkat keras standar.
        3. **Transparansi Absolut (White-box Model):**  
           Penerapan LIME (Local Interpretable Model-agnostic Explanations) mengeliminasi stigma *'Black Box'* pada AI. Setiap keputusan prediksi dapat dilacak kembali ke probabilitas kata per kata.
        """)
        
    with col_con:
        st.error("### ❌ Kekurangan (Limitations)")
        st.markdown("""
        1. **Kelemahan Semantik Terstruktur (Bag-of-Words):**  
           Berbasis TF-IDF, model ini menghitung frekuensi kata terlepas dari urutannya. Akibatnya, model masih rentan terhadap sarkasme tingkat tinggi atau ambiguitas kalimat panjang yang bergantung pada urutan subjek-predikat.
        2. **Ketergantungan Kamus Statis:**  
           Kecerdasan model dalam mendeteksi ancaman baru sangat bergantung pada leksikon (kamus) statis. Jika muncul varian bahasa *slang* baru di internet, model tidak dapat beradaptasi secara otomatis tanpa pembaruan kamus manual.
        3. **Sensitivitas Kelas Minoritas (Threat):**  
           Walaupun algoritma *Linear SVM* berhasil mengoptimalkan keseimbangan *F1-Macro*, mendeteksi kelas yang populasinya sangat langka (seperti ancaman pembunuhan/fisik) tetap menjadi tantangan heuristik tersendiri karena minimnya sampel latih.
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
        "🔮 Prediction & XAI": render_prediction,
        "🕵️ Error Analysis": render_error_analysis,
        "🧠 Global Explainability": render_explainability,
        "ℹ️ About & System Analysis": render_about
    }
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    st.sidebar.markdown("---")
    st.sidebar.caption("Research Pipeline Demo")
    
    pages[selection]()

if __name__ == "__main__":
    main()
