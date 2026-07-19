# Project Context

## 1. Project Overview

This project is a Machine Learning research project focused on classifying types of cyberbullying in Indonesian-language text.

The main objective is to analyze and compare the performance of several classical Machine Learning algorithms in classifying cyberbullying-related text using TF-IDF as the primary feature extraction method.

The project is designed as a research project, not as a production-grade Machine Learning framework.

The implementation should prioritize:

- Simplicity
- Reproducibility
- Clear experimentation
- Easy debugging
- Transparent data processing
- Clear separation between research stages

The project must avoid unnecessary software architecture complexity.

The Machine Learning experimentation process is notebook-centered.

---

# 2. Research Title

## Main Title

**Performance Analysis of Machine Learning Algorithms for Cyberbullying Type Classification on Indonesian Text Using TF-IDF**

The title may be refined if necessary, but the research scope must remain focused on:

- Cyberbullying
- Indonesian text
- Cyberbullying type classification
- TF-IDF
- Comparison of Machine Learning models

The research must not expand into unrelated topics.

---

# 3. Research Problem

Cyberbullying content in Indonesian-language social media text can appear in various forms.

Different types of cyberbullying may have similar linguistic characteristics.

For example, a text may contain insulting words but also express harassment or threats.

This creates a classification challenge for Machine Learning models.

The primary research problem is:

> How well can classical Machine Learning algorithms classify different types of cyberbullying in Indonesian-language text using TF-IDF features, and which algorithm provides the best classification performance?

The research focuses on one main problem:

> Classification of cyberbullying types in Indonesian text.

The project must not expand into multiple unrelated prediction problems.

---

# 4. Research Objective

The main objective is to:

1. Prepare and improve the quality of an Indonesian cyberbullying text dataset.

2. Preprocess Indonesian text for Machine Learning.

3. Extract text features using TF-IDF.

4. Train multiple classical Machine Learning algorithms.

5. Compare their classification performance.

6. Optimize selected models using hyperparameter tuning.

7. Evaluate the final models using appropriate classification metrics.

8. Analyze model errors.

9. Analyze important features that influence model predictions.

10. Implement the best-performing model in a Streamlit application as a proof-of-concept output.

---

# 5. Research Scope

The project focuses on:

- Indonesian-language text.
- Social media-style text.
- Cyberbullying type classification.
- Multi-class classification.
- TF-IDF feature extraction.
- Classical Machine Learning algorithms.

The primary classification labels are:

- `normal`
- `insult`
- `harassment`
- `threat`
- `hate_speech`

The exact label definitions must be maintained consistently throughout the entire pipeline.

---

# 6. Machine Learning Algorithms

The primary algorithms used in the research are:

1. Multinomial Naive Bayes

2. Logistic Regression

3. Linear Support Vector Machine

These algorithms are selected because they are:

- Suitable for sparse TF-IDF features.
- Commonly used for text classification.
- Computationally efficient.
- Suitable for comparison in a research context.
- Interpretable compared to many complex models.

The purpose of comparing these models is to determine their relative performance for Indonesian cyberbullying type classification.

---

# 7. Dataset Context

The dataset is obtained from multiple existing datasets, primarily from public dataset repositories such as Kaggle.

The dataset preparation process may include:

1. Downloading multiple relevant datasets.

2. Inspecting the dataset structure.

3. Standardizing column names.

4. Selecting relevant text and label columns.

5. Merging compatible datasets.

6. Performing exploratory data analysis.

7. Reviewing inconsistent labels.

8. Validating the final dataset.

The dataset must contain at least:

```text
text
label
```

The final dataset must use the following label categories:

```text
normal
insult
harassment
threat
hate_speech
```

---

# 8. Dataset Quality Strategy

The dataset must go through a quality improvement process before Machine Learning training.

The quality process consists of:

```text
Raw Datasets
      ↓
Dataset Merging
      ↓
Exploratory Data Analysis
      ↓
Relabeling
      ↓
Dataset Validation
      ↓
Final Validated Dataset
```

The dataset must be validated before preprocessing and model training.

The original dataset must not be silently overwritten.

Each major stage should generate a new dataset version or output file.

---

# 9. Label Relabeling

The relabeling stage exists to improve label consistency.

The relabeling process may investigate:

- Inconsistent label formats.
- Duplicate texts with different labels.
- Invalid label names.
- Label conflicts.

Example:

```text
Text:
"dasar bodoh"

Label 1:
insult

Label 2:
harassment
```

This type of conflict requires manual review.

The system must not automatically choose a label when the same text has conflicting labels.

The relabeling stage should preserve the original dataset and generate a separate relabeled dataset.

---

# 10. Dataset Validation

The validation stage is performed after relabeling.

The validation process should check:

- Missing text.
- Empty text.
- Missing labels.
- Invalid labels.
- Duplicate texts.
- Exact duplicate rows.
- Duplicate texts with the same label.
- Duplicate texts with conflicting labels.

Duplicate data must be interpreted carefully.

A duplicate text with the same label is a redundancy issue.

A duplicate text with different labels is a label consistency issue.

These two cases must not be treated identically.

The final validated dataset must be suitable for preprocessing and Machine Learning training.

---

# 11. Text Preprocessing

The preprocessing stage is performed after dataset validation.

The general preprocessing pipeline is:

```text
Original Text
      ↓
Lowercase
      ↓
Remove URLs
      ↓
Remove Mentions
      ↓
Remove Unnecessary Symbols
      ↓
Normalize Slang
      ↓
Handle Repeated Characters
      ↓
Tokenization
      ↓
Stopword Removal
      ↓
Stemming
      ↓
Clean Text
```

The original text must be preserved.

The processed dataset should contain both:

```text
text
clean_text
label
```

Example:

```text
Original:
"Dasar GOBLOKKK banget!!!"

Clean Text:
"dasar goblok banget"
```

Preprocessing must be performed consistently for all Machine Learning models.

---

# 12. Feature Engineering

TF-IDF is the primary feature extraction method.

The correct workflow is:

```text
Validated Dataset
      ↓
Preprocessing
      ↓
Train/Test Split
      ↓
Fit TF-IDF on Training Data Only
      ↓
Transform Training Data
      ↓
Transform Testing Data
```

The TF-IDF vectorizer must not be fitted on the entire dataset before the train/test split.

This is necessary to avoid data leakage.

The fitted TF-IDF vectorizer must be saved for later use by the Streamlit application.

---

# 13. Model Training Strategy

The model training process consists of two main stages.

## Baseline Training

The three primary algorithms are trained using baseline configurations:

- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM

The purpose is to establish an initial performance benchmark.

The models must use the same dataset split and the same TF-IDF representation to ensure a fair comparison.

---

## Hyperparameter Tuning

After baseline training, the models are optimized using hyperparameter tuning.

The tuning process should use cross-validation where appropriate.

The purpose is to identify better parameter configurations.

The tuning process must not change the dataset or preprocessing pipeline.

---

# 14. Model Evaluation

The models must be evaluated using classification metrics.

Primary metrics include:

- Accuracy
- Precision
- Recall
- F1-score

For multi-class classification, the following metrics are especially important:

- Macro F1-score
- Weighted F1-score

A confusion matrix must also be generated.

The final model comparison must be based on consistent evaluation data.

The best model should be selected based on the research objective and evaluation results.

---

# 15. Error Analysis

Error analysis is performed after model evaluation.

The purpose is to understand why the model makes incorrect predictions.

The analysis should investigate:

- Frequently confused classes.
- Similar language between cyberbullying categories.
- Ambiguous text.
- Slang.
- Sarcasm.
- Short text.
- Context-dependent language.
- Text with overlapping cyberbullying characteristics.

Example:

```text
Actual:
harassment

Predicted:
insult
```

The error analysis must focus on identifying patterns in model mistakes.

---

# 16. Explainability

Explainability is performed using the best-performing model.

The purpose is to identify important TF-IDF features associated with each class.

For linear models, the analysis may include:

- Important positive features.
- Important negative features.
- Top features per class.
- Feature weights.

The results should help explain which words or terms contribute to classification decisions.

Example:

```text
Class:
insult

Important Features:
- goblok
- bodoh
- tolol
```

The results must be interpreted carefully.

Important features do not necessarily mean that a single word alone determines the true meaning of a text.

---

# 17. Streamlit Application

The Streamlit application is the final proof-of-concept output.

The application should load:

```text
models/tfidf_vectorizer.pkl
```

and:

```text
models/best_model_tuned.pkl
```

The prediction pipeline is:

```text
User Input
      ↓
Preprocessing
      ↓
TF-IDF Transformation
      ↓
Best Machine Learning Model
      ↓
Predicted Cyberbullying Type
      ↓
Confidence or Probability Output
```

The Streamlit application is not responsible for training models.

It should only perform inference using the already-trained model.

---

# 18. Project Architecture

The project uses a notebook-centered architecture.

The project does NOT use a complex `src/` architecture.

The project structure is:

```text
project/
│
├── data/
│   ├── raw/
│   ├── merged/
│   ├── relabeled/
│   ├── validated/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_merging.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_relabeling.ipynb
│   ├── 04_validation.ipynb
│   ├── 05_preprocessing.ipynb
│   ├── 06_tfidf.ipynb
│   ├── 07_model_training.ipynb
│   ├── 08_hyperparameter_tuning.ipynb
│   ├── 09_model_evaluation.ipynb
│   ├── 10_error_analysis.ipynb
│   └── 11_explainability.ipynb
│
├── models/
│
├── reports/
│
├── streamlit/
│   └── app.py
│
├── requirements.txt
│
└── README.md
```

---

# 19. Notebook Responsibilities

## 01_data_merging.ipynb

Responsibilities:

- Load raw datasets.
- Inspect datasets.
- Standardize columns.
- Merge datasets.
- Save merged dataset.

Output:

```text
data/merged/merged_dataset.csv
```

---

## 02_eda.ipynb

Responsibilities:

- Analyze dataset structure.
- Analyze label distribution.
- Analyze text length.
- Analyze missing values.
- Analyze duplicates.
- Generate visualizations.

The dataset must not be modified.

---

## 03_relabeling.ipynb

Responsibilities:

- Identify label inconsistencies.
- Identify duplicate text with conflicting labels.
- Generate review files.
- Apply manually reviewed labels.
- Save relabeled dataset.

Output:

```text
data/relabeled/relabeled_dataset.csv
```

---

## 04_validation.ipynb

Responsibilities:

- Validate the relabeled dataset.
- Check missing values.
- Check invalid labels.
- Check duplicates.
- Check label conflicts.
- Generate validation reports.

Output:

```text
data/validated/validated_dataset.csv
```

---

## 05_preprocessing.ipynb

Responsibilities:

- Clean Indonesian text.
- Normalize text.
- Tokenize.
- Remove stopwords where appropriate.
- Apply stemming where appropriate.
- Save processed dataset.

Output:

```text
data/processed/preprocessed_dataset.csv
```

---

## 06_tfidf.ipynb

Responsibilities:

- Load processed dataset.
- Perform train/test split.
- Fit TF-IDF on training data only.
- Transform training data.
- Transform testing data.
- Save vectorizer and feature matrices.

---

## 07_model_training.ipynb

Responsibilities:

- Train baseline models.
- Use the same train/test split.
- Compare initial model performance.
- Save baseline results.

Models:

- Multinomial Naive Bayes.
- Logistic Regression.
- Linear SVM.

---

## 08_hyperparameter_tuning.ipynb

Responsibilities:

- Tune model hyperparameters.
- Use cross-validation.
- Identify best configurations.
- Save the best model.

---

## 09_model_evaluation.ipynb

Responsibilities:

- Evaluate final models.
- Generate classification reports.
- Generate confusion matrices.
- Compare models.
- Select the best-performing model.

---

## 10_error_analysis.ipynb

Responsibilities:

- Identify misclassified samples.
- Analyze confusion patterns.
- Investigate common model errors.
- Generate error analysis reports.

---

## 11_explainability.ipynb

Responsibilities:

- Analyze important features.
- Identify top TF-IDF terms per class.
- Explain model behavior.

---

# 20. Data Flow

The complete data flow is:

```text
Raw Datasets
      │
      ▼
01 Data Merging
      │
      ▼
Merged Dataset
      │
      ▼
02 EDA
      │
      ▼
03 Relabeling
      │
      ▼
Relabeled Dataset
      │
      ▼
04 Validation
      │
      ▼
Validated Dataset
      │
      ▼
05 Preprocessing
      │
      ▼
Processed Dataset
      │
      ▼
06 TF-IDF
      │
      ▼
Train/Test Feature Matrices
      │
      ▼
07 Baseline Model Training
      │
      ▼
08 Hyperparameter Tuning
      │
      ▼
09 Model Evaluation
      │
      ▼
Best Model
      │
      ├───────────────┐
      ▼               ▼
10 Error Analysis    11 Explainability
      │
      ▼
Streamlit Application
```

---

# 21. Core Research Principle

The project must prioritize research clarity over software architecture complexity.

Each notebook should:

1. Load the output of the previous stage.

2. Perform one clearly defined research stage.

3. Generate visible results.

4. Save the output.

5. Make the process reproducible.

The project should be easy to debug.

If an error occurs, the researcher should be able to identify:

- Which notebook failed.
- Which input data was used.
- Which processing step caused the problem.
- Which output was generated.

---

# 22. Important Constraints

Do NOT:

- Create a complex `src/` architecture.
- Create unnecessary abstraction layers.
- Create excessive Python modules.
- Hide important research logic inside unknown utility files.
- Train models inside Streamlit.
- Fit TF-IDF before train/test splitting.
- Modify the original dataset without creating a new output.
- Mix multiple research stages inside one notebook unnecessarily.

The project must remain simple, transparent, and easy to debug.

---

# 23. Final Research Pipeline

The final pipeline is:

```text
Dataset Collection
      ↓
01 Data Merging
      ↓
02 Exploratory Data Analysis
      ↓
03 Relabeling
      ↓
04 Dataset Validation
      ↓
05 Text Preprocessing
      ↓
06 TF-IDF Feature Engineering
      ↓
07 Baseline Model Training
      ↓
08 Hyperparameter Tuning
      ↓
09 Model Evaluation
      ↓
10 Error Analysis
      ↓
11 Explainability
      ↓
Streamlit Proof-of-Concept
```

This pipeline must be followed consistently throughout the project.
