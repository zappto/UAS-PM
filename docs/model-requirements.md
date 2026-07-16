# Model Requirements Document (MRD)

## Project Information

| Item           | Description                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Project        | Cyberbullying Type Classification                                                                                         |
| Research Title | Performance Analysis of Machine Learning Algorithms for Cyberbullying Type Classification on Indonesian Text Using TF-IDF |
| Domain         | Natural Language Processing (NLP)                                                                                         |
| Learning Type  | Supervised Learning                                                                                                       |
| Problem Type   | Multi-Class Classification                                                                                                |

---

# Purpose

This document defines the Machine Learning modeling requirements for the research project.

It describes the model development process, feature extraction, candidate algorithms, evaluation strategy, explainability, and deployment requirements.

The objective is to ensure every experiment follows a reproducible and consistent methodology.

---

# Research Objective

Develop and evaluate several classical Machine Learning algorithms to classify cyberbullying types in Indonesian text.

The research focuses on comparing model performance using the same dataset, preprocessing pipeline, and feature extraction method.

---

# Problem Definition

Input

Text (Indonesian social media comment)

Example

```text
Dasar bodoh kamu.
```

Output

One predicted class.

Example

```text
Prediction

Insult
```

Along with

- Prediction Confidence
- Probability Distribution
- Explainability Result

---

# Dataset

Dataset Source

dataset/processed/final_dataset.csv

Required Columns

| Column | Description  |
| ------ | ------------ |
| text   | Input text   |
| label  | Ground Truth |

---

# Feature Extraction

Technique

TF-IDF (Term Frequency – Inverse Document Frequency)

Reason

- Suitable for text classification.
- Efficient for classical Machine Learning algorithms.
- Easy to interpret.
- Consistent with research scope.

Output

Sparse numerical feature vectors.

---

# Candidate Models

The following algorithms will be evaluated.

## Naive Bayes

Purpose

Baseline classifier.

Advantages

- Fast
- Lightweight
- Common benchmark for NLP

---

## Logistic Regression

Purpose

Linear probabilistic classifier.

Advantages

- Stable performance
- Produces probability scores
- Easy to interpret

---

## Support Vector Machine (SVM)

Purpose

High-performance classifier for sparse text data.

Advantages

- Excellent performance with TF-IDF
- Strong generalization capability

---

# Experiment Pipeline

```mermaid
flowchart LR

Dataset

-->

EDA

-->

Preprocessing

-->

TF-IDF

-->

Train Test Split

-->

Naive Bayes

Dataset

-->

EDA

-->

Preprocessing

-->

TF-IDF

-->

Train Test Split

-->

Logistic Regression

Dataset

-->

EDA

-->

Preprocessing

-->

TF-IDF

-->

Train Test Split

-->

Support Vector Machine

Naive Bayes --> Evaluation

Logistic Regression --> Evaluation

Support Vector Machine --> Evaluation

Evaluation --> Best Model
```

---

# Data Split Strategy

Training Set

80%

Testing Set

20%

Random State

42

Stratified Split

Enabled

Reason

Maintain class distribution.

---

# Hyperparameter Optimization

Optimization Method

GridSearchCV

Alternative

RandomizedSearchCV

Hyperparameter tuning will be applied individually for each candidate model.

---

# Evaluation Metrics

Every model must be evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve (if applicable)

Primary Metric

F1 Score

Reason

Suitable for imbalanced classification problems.

---

# Model Selection Criteria

The best model will be selected based on:

1. Highest F1 Score
2. Highest Recall
3. Highest Precision
4. Highest Accuracy
5. Model Simplicity

---

# Explainability

The selected model must support prediction explanation.

Technique

SHAP

Outputs

- Feature Importance
- Word Contribution
- Local Explanation
- Global Explanation

Purpose

To identify which words contribute the most to each prediction.

---

# Model Artifacts

The following files must be generated.

models/

```text
best_model.pkl
tfidf_vectorizer.pkl
label_encoder.pkl
```

---

# Prediction Output

Every prediction should return:

Example

```text
Input

Dasar bodoh kamu.

Prediction

Insult
```

Probability

| Class       | Probability |
| ----------- | ----------: |
| Insult      |       91.4% |
| Harassment  |        3.2% |
| Threat      |        2.1% |
| Hate Speech |        1.8% |
| Normal      |        1.5% |

Explainability

| Word  | Contribution |
| ----- | -----------: |
| bodoh |        +0.63 |
| dasar |        +0.27 |

---

# Expected Performance

| Metric    | Target |
| --------- | ------ |
| Accuracy  | ≥ 80%  |
| Precision | ≥ 80%  |
| Recall    | ≥ 80%  |
| F1 Score  | ≥ 80%  |

Target values are indicative and depend on dataset quality.

---

# Validation Strategy

The final model must be validated using the testing dataset.

Validation includes:

- Classification Report
- Confusion Matrix
- Error Analysis

---

# Error Analysis

Misclassified samples should be analyzed.

The analysis should identify:

- Common misclassified labels
- Possible causes
- Dataset limitations

---

# Risks

| Risk                | Mitigation                       |
| ------------------- | -------------------------------- |
| Overfitting         | Hyperparameter tuning            |
| Underfitting        | Compare multiple models          |
| Class imbalance     | Use F1 Score as primary metric   |
| Label inconsistency | Normalize labels before training |

---

# Assumptions

- Dataset has been cleaned.
- Labels have been standardized.
- TF-IDF is applied consistently to all models.
- Every model uses the same train-test split.

---

# Out of Scope

This research does not include:

- Deep Learning
- LSTM
- CNN
- Transformer
- BERT
- IndoBERT
- GPT
- Large Language Models (LLMs)
- Ensemble Learning
- Reinforcement Learning
- Online Learning
- Active Learning

---

# Deliverables

The modeling phase must produce:

- Trained Models
- Best Model
- TF-IDF Vectorizer
- Label Encoder
- Evaluation Report
- Classification Report
- Confusion Matrix
- SHAP Analysis
- Model Comparison Report
