"""Constants utility for the Streamlit application.

Stores application metadata, standard paths, labels, and predefined text examples.
"""

import os

# App Information
APP_NAME = "Cyberbullying Classification"
APP_VERSION = "1.0.0"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_COMPARISON_PATH = os.path.join(BASE_DIR, "reports", "evaluation", "comparison", "model_comparison.csv")
MODEL_CONFIG_PATH = os.path.join(BASE_DIR, "reports", "evaluation", "model_metadata.json")

# Predefined Examples
EXAMPLE_TEXTS = {
    "Normal": "Wah hari ini cuacanya sangat cerah, cocok untuk jalan-jalan.",
    "Insult": "Dasar bodoh, kerjaan gampang gini aja kamu nggak bisa.",
    "Harassment": "Hai cantik, sendirian aja nih? Boleh kenalan nggak?",
    "Threat": "Awas ya kalau berani macam-macam, habis kamu di tanganku.",
    "Hate Speech": "Orang-orang dari ras itu memang semuanya penjahat dan tidak berguna."
}

# General UI Labels
LABEL_PREDICT_BUTTON = "Predict Cyberbullying Type"
LABEL_CLEAR_BUTTON = "Clear Text"
LABEL_DOWNLOAD_JSON = "Download JSON"
LABEL_DOWNLOAD_CSV = "Download CSV"
