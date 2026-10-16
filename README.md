# MindCare AI (MEDAI) — Clinical Mental Health Intelligence & Recommendation Engine

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Scikit--Learn%20%7C%20Flask-orange.svg)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Deep%20Neural%20Network-purple.svg)](https://pytorch.org/)
[![Inference Status](https://img.shields.io/badge/Inference-Local%20%26%20Private-success.svg)](#privacy-security--ethical-ai)
[![License](https://img.shields.io/badge/License-Research%20%26%20Educational-lightgrey.svg)](#license)

> **MindCare AI** is an enterprise-grade clinical decision support and mental health monitoring system. It unites high-dimensional survey preprocessing, rigorous feature engineering, deep neural network classification in PyTorch, baseline benchmarking with Scikit-Learn, a RESTful Flask API, and an interactive cybernetic frontend chatbot.

---

## Table of Contents

- [1. Executive Summary & Core Mission](#1-executive-summary--core-mission)
- [2. System Architecture & Information Flow](#2-system-architecture--information-flow)
- [3. Repository Structure & File Index](#3-repository-structure--file-index)
- [4. Dataset Deep-Dive & Clinical Feature Dictionary](#4-dataset-deep-dive--clinical-feature-dictionary)
  - [4.1 Dataset Demographics & Distribution](#41-dataset-demographics--distribution)
  - [4.2 The 15 Raw Clinical Input Features](#42-the-15-raw-clinical-input-features)
- [5. Preprocessing & Feature Engineering Pipeline](#5-preprocessing--feature-engineering-pipeline)
  - [5.1 Preprocessing Pipeline (`backend/preprocessing.py`)](#51-preprocessing-pipeline-backendpreprocessingpy)
  - [5.2 Feature Transformation & Encoding (`backend/feature_engineering.py`)](#52-feature-transformation--encoding-backendfeature_engineeringpy)
  - [5.3 Schema Isolation Principle (15 Raw vs 77 Model Features)](#53-schema-isolation-principle-15-raw-vs-77-model-features)
- [6. Model Architectures & Training Protocols](#6-model-architectures--training-protocols)
  - [6.1 Production Deep Neural Network (`MindCarePyTorchClassifier`)](#61-production-deep-neural-network-mindcarepytorchclassifier)
  - [6.2 PyTorch Training Pipeline (`backend/pytorch_trainer.py`)](#62-pytorch-training-pipeline-backendpytorch_trainerpy)
  - [6.3 Scikit-Learn Candidate Models (`backend/trainer.py`)](#63-scikit-learn-candidate-models-backendtrainerpy)
- [7. Benchmark Results, Evaluation & Model Selection](#7-benchmark-results-evaluation--model-selection)
  - [7.1 Comprehensive Model Comparison (Test Set: 58,473 Samples)](#71-comprehensive-model-comparison-test-set-58473-samples)
  - [7.2 PyTorch Deep Learning In-Depth Metrics](#72-pytorch-deep-learning-in-depth-metrics)
  - [7.3 Multi-Tier Selection Strategy (`backend/model_comparator.py`)](#73-multi-tier-selection-strategy-backendmodel_comparatorpy)
  - [7.4 Visual Metric Artifacts (`plots/`)](#74-visual-metric-artifacts-plots)
- [8. Recommendation Engine (Clinical Decision Support)](#8-recommendation-engine-clinical-decision-support)
  - [8.1 Tiered Recommendation Taxonomy](#81-tiered-recommendation-taxonomy)
  - [8.2 Supported Conditions & Knowledge Base Mapping](#82-supported-conditions--knowledge-base-mapping)
- [9. REST API Specification & Endpoints](#9-rest-api-specification--endpoints)
  - [9.1 Endpoints Overview](#91-endpoints-overview)
  - [9.2 Detailed Endpoint Reference](#92-detailed-endpoint-reference)
  - [9.3 Request & Response Payloads](#93-request--response-payloads)
- [10. Conversational Frontend Assistant](#10-conversational-frontend-assistant)
  - [10.1 UI/UX Architecture](#101-uiux-architecture)
  - [10.2 Conversational Flow & Dynamic Controls](#102-conversational-flow--dynamic-controls)
- [11. Installation & Operation Manual](#11-installation--operation-manual)
  - [11.1 Prerequisites & Environment Setup](#111-prerequisites--environment-setup)
  - [11.2 Training the Models](#112-training-the-models)
  - [11.3 Launching the API & Web App](#113-launching-the-api--web-app)
  - [11.4 Verifying the API](#114-verifying-the-api)
- [12. Engineering Notes, Gotchas & Troubleshooting](#12-engineering-notes-gotchas--troubleshooting)
- [13. Privacy, Security & Ethical AI](#13-privacy-security--ethical-ai)
- [14. License & Clinical Disclaimer](#14-license--clinical-disclaimer)

---

## 1. Executive Summary & Core Mission

Mental health distress in academic and corporate environments frequently goes unnoticed until severe clinical impairment occurs. **MindCare AI** provides early detection, risk stratification, and timely intervention recommendations.

### Key Pillars
1. **Clinical Sensitivity**: Optimized for **Recall (Sensitivity: 83.97%)** to drastically reduce false negatives, ensuring individuals in need of care are identified.
2. **Deterministic Preprocessing**: A bulletproof data engineering pipeline that reconstructs missing features at inference time using learned population medians and modes.
3. **Deep Representation Learning**: A 3-layer deep neural network with Batch Normalization, LeakyReLU activations, Dropout regularization, and mixed precision acceleration.
4. **Transparent Clinical Recommendations**: Rule-based, auditable, multi-tier recommendations linking users to immediate calming strategies, evidence-based psychotherapies (CBT, ACT), self-care regimens, and verified emergency helplines.
5. **Zero Data Leakage / 100% Private Inference**: All processing runs locally with zero telemetry or third-party cloud data routing.

---

## 2. System Architecture & Information Flow

The following diagram illustrates the complete end-to-end data processing lifecycle from raw user inputs to final recommendations:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        RAW INPUT INGESTION                             │
│  - Training Data: dataset/data.csv (292,364 rows x 17 columns)         │
│  - Realtime Input: JSON payload from Web App / REST API (15 features)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               STAGE 1: DATA PREPROCESSING (DataPreprocessor)           │
│  • Normalise column headers (snake_case, lower, strip)                 │
│  • Deduplication & numeric/categorical type detection                  │
│  • Impute missing numeric values via Population Medians                │
│  • Impute missing categorical values via Population Modes              │
│  • Clamp numeric outliers via IQR bounds [Q1 - 1.5*IQR, Q3 + 1.5*IQR]  │
│  • Label encode target: 'No' -> 0, 'Yes' -> 1                          │
│  • StandardScale numeric columns: z = (x - μ) / σ                      │
│  • Artefact Output: raw_feature_columns.pkl (15 columns)               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│            STAGE 2: FEATURE ENGINEERING (FeatureEngineer)              │
│  • One-Hot Encoding on categorical attributes (drop_first=False)       │
│  • Align with learned training dummy schema (ohe_columns.pkl)          │
│  • VarianceThreshold(threshold=0.0) removes zero-variance constants    │
│  • Artefact Output: Exact 77-feature tensor/matrix                     │
│    (feature_names.pkl, feature_engineer_meta.pkl)                     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
┌───────────────────────────────────┐   ┌────────────────────────────────┐
│      TRAINING & EVALUATION        │   │       REAL-TIME INFERENCE      │
│  • Stratified Split (60/20/20)    │   │  • MindCarePredictor singleton │
│  • 3-Fold Stratified CV           │   │  • PyTorch forward pass (CUDA/ │
│  • 5 Scikit-Learn Baselines       │   │    CPU with float32 tensors)   │
│  • PyTorch Deep Neural Network    │   │  • Fallback: Scikit-Learn      │
│    (AdamW, AMP, EarlyStopping)    │   │    Decision Tree classifier    │
│  • Evaluator (Accuracy, Recall,   │   │  • Output: Class index +       │
│    F1, ROC-AUC, ECE, LogLoss)     │   │    Posterior Probabilities     │
│  • ModelComparator selects BEST   │   └───────────────┬────────────────┘
└─────────────────┬─────────────────┘                   │
                  │                                     │
                  ▼                                     ▼
┌───────────────────────────────────┐   ┌────────────────────────────────┐
│         PERSISTED ARTEFACTS       │   │   STAGE 3: RECOMMENDATION      │
│  • models/best_model.pkl          │   │            ENGINE              │
│  • models/torch_model.pth         │   │  • Risk Tier Classification    │
│  • models/training_metadata.json  │   │  • Immediate Crisis Actions    │
│  • reports/model_comparison.csv   │   │  • Professional Referrals      │
│  • plots/*.png (ROC, PR, CM)      │   │  • Self-Care & Lifestyle       │
└───────────────────────────────────┘   │  • Emergency Helplines         │
                                        └───────────────┬────────────────┘
                                                        │
                                                        ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        CONSUMPTION INTERFACES                          │
│  • Flask REST API: POST /predict, GET /health, GET /api/metrics        │
│  • Web Chatbot Interface: GET /app (Cybernetic Dark UI)                │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Repository Structure & File Index

```
MEDAI/
│
├── backend/                             # Core Python engine & processing pipeline
│   ├── __init__.py                      # Package initialization & namespace exports
│   ├── config.py                        # Central configuration singleton & path resolver
│   ├── logger.py                        # Rotating file & console logging configuration
│   ├── utils.py                         # Shared helper functions (timestamps, serialization)
│   ├── data_loader.py                   # CSV loading, validation & DataLoadingError handling
│   ├── preprocessing.py                 # Data cleaning, median/mode imputation, IQR clamping
│   ├── feature_engineering.py           # One-Hot Encoding & VarianceThreshold pipeline
│   ├── trainer.py                       # 3-Fold CV training for 5 Scikit-Learn candidate models
│   ├── pytorch_model.py                 # PyTorch Deep Neural Network (MindCarePyTorchClassifier)
│   ├── pytorch_trainer.py               # PyTorch training loop, AMP, EarlyStopping & DataLoader
│   ├── evaluator.py                     # Metric calculation (F1, ROC-AUC, ECE) & Matplotlib plots
│   ├── evaluation.py                    # Re-export compatibility alias
│   ├── model_comparator.py              # Multi-tier model ranking & automated selection
│   ├── model_saver.py                   # Centralized persistence for PyTorch & Scikit-Learn
│   ├── recommendation_engine.py         # 4-tier clinical recommendation knowledge base
│   └── predictor.py                     # High-performance end-to-end inference singleton
│
├── frontend/                            # Pure Vanilla Web Client (No Node/NPM dependencies)
│   ├── index.html                       # Semantic HTML5 conversational chatbot & benchmark modal
│   ├── style.css                        # Cybernetic dark UI with glassmorphism & responsive layout
│   └── script.js                        # ES6+ dynamic multi-step clinical assessment controller
│
├── dataset/                             # Data storage
│   └── data.csv                         # Population survey dataset (292,364 rows x 17 columns)
│
├── models/                              # Trained model artifacts & learned schemas
│   ├── best_model.pkl                   # Champion Scikit-Learn estimator (Decision Tree)
│   ├── torch_model.pth                  # Champion Deep Learning weights (PyTorch state_dict)
│   ├── raw_feature_columns.pkl          # Exact 15 raw feature column names for input reconstruction
│   ├── ohe_columns.pkl                  # Complete column list after One-Hot Encoding
│   ├── feature_names.pkl                # Final 77 model feature columns after variance thresholding
│   ├── feature_encoder.pkl              # Preprocessing metadata (medians, modes, IQR bounds)
│   ├── feature_engineer_meta.pkl        # Feature engineering metadata & selector instance
│   ├── label_encoder.pkl                # Target LabelEncoder ('No' -> 0, 'Yes' -> 1)
│   ├── scaler.pkl                       # StandardScaler instance for numeric columns
│   └── training_metadata.json           # Comprehensive snapshot of training run & benchmarks
│
├── plots/                               # Auto-generated high-resolution evaluation figures (22 PNGs)
│   ├── confusion_matrix_*.png           # Raw confusion matrices per model
│   ├── confusion_matrix_normalized_*.png# Normalized (%) confusion matrices per model
│   ├── roc_curve_*.png                  # Receiver Operating Characteristic curves with AUC
│   └── pr_curve_*.png                   # Precision-Recall curves with Average Precision (AP)
│
├── reports/                             # Evaluation reports & tabular comparison summaries
│   ├── classification_report_*.txt      # Text classification reports (Precision, Recall, F1)
│   ├── metrics_*.json                   # Complete JSON metric dumps (LogLoss, Brier, ECE, MCC)
│   ├── model_comparison.csv             # Full benchmark comparison table across all models
│   └── training_summary.txt             # Text summary of model comparison & selection rationale
│
├── logs/                                # Application runtime logs
│   └── mindcare.log                     # Rotating execution and inference log file
│
├── app.py                               # Flask REST API server & static frontend host
├── train.py                             # Main command-line training orchestration entrypoint
├── test_api.ps1                         # Automated PowerShell API test script
├── requirements.txt                     # Pinned project dependencies
└── README.md                            # Comprehensive project manual & documentation
```

---

## 4. Dataset Deep-Dive & Clinical Feature Dictionary

### 4.1 Dataset Demographics & Distribution

The primary training dataset (`dataset/data.csv`) is derived from comprehensive global workplace mental health surveys.

- **Total Records**: `292,364` rows
- **Total Columns**: `17` columns (1 identifier timestamp, 15 clinical predictor features, 1 target)
- **Target Feature**: `treatment`
  - **`Yes`**: `147,606` instances (**50.49%**) — Individual has sought or requires mental health treatment.
  - **`No`**: `144,758` instances (**49.51%**) — Individual has not sought treatment.
  - **Class Balance**: Exceptionally well-balanced (near 1:1 ratio), mitigating majority-class bias.

---

### 4.2 The 15 Raw Clinical Input Features

The table below outlines each predictor feature, its clinical significance, data type, and valid categories:

| # | Feature Name | Type | Valid Domain / Categories | Clinical & Psychological Rationale |
|---|---|---|---|---|
| 1 | `gender` | Categorical | `Male`, `Female` | Demographic baseline; accounts for gender-disaggregated patterns in symptom presentation and treatment-seeking behavior. |
| 2 | `country` | Categorical | 35 Countries (e.g., `United States`, `United Kingdom`, `Canada`, `India`, `Australia`, `Germany`, etc.) | Adjusts for cultural stigma, local healthcare infrastructure, and geographic access to psychiatric care. |
| 3 | `occupation` | Categorical | `Corporate`, `Student`, `Business`, `Housewife`, `Others` | Captures work-life stress profiles, performance anxiety, and environmental pressures across occupations. |
| 4 | `self_employed` | Categorical | `No`, `Yes` | Evaluates employment autonomy, income stability, and absence of employer-sponsored health benefits. |
| 5 | `family_history` | Categorical | `No`, `Yes` | Major psychiatric risk factor; captures genetic predisposition to mood and anxiety disorders. |
| 6 | `days_indoors` | Categorical | `Go out Every day`, `1-14 days`, `15-30 days`, `31-60 days`, `More than 2 months` | Measures social isolation, confinement, and sunlight deprivation, which directly affect circadian rhythms and mood regulation. |
| 7 | `growing_stress` | Categorical | `No`, `Maybe`, `Yes` | Subjective perception of escalating acute stress over recent weeks; critical early warning indicator. |
| 8 | `changes_habits` | Categorical | `No`, `Maybe`, `Yes` | Uncharacteristic shifts in sleep patterns, appetite, or personal hygiene, reflecting biological disruption. |
| 9 | `mental_health_history` | Categorical | `No`, `Maybe`, `Yes` | Prior clinical diagnosis or episodes of mental health distress, increasing recurrence probability. |
| 10 | `mood_swings` | Categorical | `Low`, `Medium`, `High` | Evaluates affective instability, emotional volatility, and difficulty maintaining emotional equilibrium. |
| 11 | `coping_struggles` | Categorical | `No`, `Yes` | Assesses psychological resilience, emotional exhaustion, and acute difficulty managing routine stressors. |
| 12 | `work_interest` | Categorical | `No`, `Maybe`, `Yes` | Assesses anhedonia (loss of interest or pleasure in daily activities) and executive cognitive fatigue. |
| 13 | `social_weakness` | Categorical | `No`, `Maybe`, `Yes` | Interpersonal detachment, withdrawal from support networks, and feelings of social fatigue. |
| 14 | `mental_health_interview` | Categorical | `No`, `Maybe`, `Yes` | Willingness to discuss psychological health; measures internalized stigma and communicative openness. |
| 15 | `care_options` | Categorical | `No`, `Not sure`, `Yes` | Awareness of and access to employer-sponsored mental health programs, therapy benefits, or community clinics. |

> **Note on Timestamp**: The `Timestamp` column in `data.csv` is an administrative artifact and is automatically stripped by `DataPreprocessor` before modeling.

---

## 5. Preprocessing & Feature Engineering Pipeline

### 5.1 Preprocessing Pipeline (`backend/preprocessing.py`)

The preprocessing module prepares raw, noisy data for model consumption while avoiding data leakage:

1. **Header Normalization**: All column names are converted to lowercase, stripped of whitespace, and formatted with snake_case (`Changes_Habits` &rarr; `changes_habits`).
2. **Deduplication**: Identical duplicate rows are removed from training splits.
3. **Type Separation**: Automatically identifies numeric columns vs. categorical columns.
4. **Missing Value Imputation**:
   - **Numeric features**: Imputed using the **median** of the training split.
   - **Categorical features**: Imputed using the **mode** (most frequent class) of the training split.
   - All medians and modes are serialized into `feature_encoder.pkl` to impute missing inputs during inference.
5. **Outlier Clamping**: Computes the Interquartile Range ($\text{IQR} = Q_3 - Q_1$) on numeric attributes. Values outside $[Q_1 - 1.5 \times \text{IQR}, \, Q_3 + 1.5 \times \text{IQR}]$ are clamped.
6. **Target Encoding**: `LabelEncoder` encodes binary target values:
   $$\text{'No'} \mapsto 0, \quad \text{'Yes'} \mapsto 1$$
7. **Numeric Standardization**: `StandardScaler` standardizes numeric columns to zero mean and unit variance:
   $$z = \frac{x - \mu}{\sigma}$$
8. **Categorical Preservation**: Categorical values remain strings at this stage so that `FeatureEngineer` can perform consistent one-hot encoding.

---

### 5.2 Feature Transformation & Encoding (`backend/feature_engineering.py`)

`FeatureEngineer` converts preprocessed features into a numeric matrix:

1. **One-Hot Encoding**: Performs full dummy encoding (`pd.get_dummies(..., drop_first=False)`) across all categorical features.
2. **Schema Alignment**: The complete dummy column order is saved to `ohe_columns.pkl`. During inference, incoming payloads are reindexed against this exact list, filling unseen categories with `0`.
3. **Zero-Variance Filtering**: Applies `sklearn.feature_selection.VarianceThreshold(threshold=0.0)`. Any dummy feature with zero variance across the training split is removed, leaving **exactly 77 active features**.
4. **Strict Dimensionality Verification**: Validates that the transformed feature matrix contains exactly 77 columns, in the exact expected order, with no `NaN` values before passing to models.

---

### 5.3 Schema Isolation Principle (15 Raw vs 77 Model Features)

A common issue in ML systems is artifact collision between input schemas and model feature schemas. MindCare AI enforces a strict architectural boundary:

```
Raw User Input (15 Features)
        │
        ▼  [DataPreprocessor saves raw_feature_columns.pkl]
Preprocessed Intermediate
        │
        ▼  [FeatureEngineer saves ohe_columns.pkl & feature_names.pkl]
Final Transformed Matrix (77 Features)
        │
        ▼  [MindCarePyTorchClassifier / Sklearn Model consumes 77 Features]
```

- **`raw_feature_columns.pkl`**: Stores the **15 raw input column names**. If a user submits a partial JSON payload with missing keys, `_restore_missing_features()` reconstructs the missing fields using the learned medians and modes.
- **`feature_names.pkl`**: Stores the **77 final engineered model feature names**. Used to validate that the neural network and Scikit-Learn models receive the exact tensor structure they were trained on.

---

## 6. Model Architectures & Training Protocols

### 6.1 Production Deep Neural Network (`MindCarePyTorchClassifier`)

The primary production classifier is a custom deep feedforward neural network designed in PyTorch (`backend/pytorch_model.py`):

```
Input Vector [Batch Size x 77 Features]
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ Dense Layer 1: Linear(77 -> 128)                       │
│ Batch Normalization: BatchNorm1d(128)                  │
│ Activation: LeakyReLU(negative_slope=0.01)             │
│ Regularization: Dropout(p=0.2)                         │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Dense Layer 2: Linear(128 -> 64)                       │
│ Batch Normalization: BatchNorm1d(64)                   │
│ Activation: LeakyReLU(negative_slope=0.01)             │
│ Regularization: Dropout(p=0.2)                         │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Dense Layer 3: Linear(64 -> 32)                        │
│ Batch Normalization: BatchNorm1d(32)                   │
│ Activation: LeakyReLU(negative_slope=0.01)             │
│ Regularization: Dropout(p=0.2)                         │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Output Classification Head: Linear(32 -> 2 Logits)     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
       Softmax Normalization: P(y=c|x) = exp(z_c) / Σ exp(z_j)
                            │
                            ▼
      Output: [P(Class 0 = "No"), P(Class 1 = "Yes")]
```

#### Neural Network Design Details
- **Input Dimension**: 77 engineered features.
- **Hidden Layers**: `[128, 64, 32]` with continuous dimensionality reduction.
- **Batch Normalization**: Applied before each activation to stabilize internal covariate shift and accelerate convergence.
- **Activation Function**: `LeakyReLU(negative_slope=0.01)` prevents dying neurons.
- **Dropout Rate**: `0.2` on every hidden layer to prevent co-adaptation of features.
- **Weight Initialization**: **Kaiming Normal** initialization tailored for LeakyReLU:
  $$\text{std} = \sqrt{\frac{2}{(1 + a^2) \times \text{fan\_in}}}, \quad a = 0.01$$
  Biases are initialized to `0.0`, and BatchNorm scale weights are initialized to `1.0`.
- **Classification Head**: Standardized to $N = 2$ output logits combined with `nn.CrossEntropyLoss()`.

---

### 6.2 PyTorch Training Pipeline (`backend/pytorch_trainer.py`)

- **Optimizer**: `torch.optim.AdamW` with decoupled weight decay:
  - Initial Learning Rate: $\eta = 10^{-3}$
  - Weight Decay: $\lambda = 10^{-4}$
- **Learning Rate Scheduler**: `ReduceLROnPlateau` dynamically halves the learning rate when validation loss plateaus:
  - Mode: `min`
  - Factor: `0.5`
  - Patience: `3` epochs
- **Early Stopping**: Halts training if validation loss fails to improve by at least $\Delta_{\text{min}} = 10^{-4}$ over `5` consecutive epochs. The best model weights are automatically restored at the end of training.
- **Mixed Precision Acceleration (AMP)**: Uses `torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')` on CUDA-enabled GPUs, with automatic fallback to full float32 precision on CPU.
- **Reproducible DataLoaders**: Uses an explicit `torch.Generator().manual_seed(42)` for consistent batch shuffling.

---

### 6.3 Scikit-Learn Candidate Models (`backend/trainer.py`)

In addition to the PyTorch neural network, five Scikit-Learn models are trained using **Stratified 3-Fold Cross-Validation**:

1. **Logistic Regression**: $L_2$-regularized linear classifier (`max_iter=1000`).
2. **Decision Tree Classifier**: Non-linear tree model constrained to `max_depth=10` to prevent overfitting.
3. **Random Forest Classifier**: Ensemble of 50 de-correlated decision trees (`n_estimators=50`).
4. **Support Vector Machine (LinearSVC)**: Maximum-margin hyperplane classifier (`dual=False`, `max_iter=2000`).
5. **Gaussian Naive Bayes (`GaussianNB`)**: Probabilistic baseline assuming conditional feature independence.

> **Engineering Note on K-Nearest Neighbors (KNN)**: KNN was purposefully excluded from the training matrix. Storing and calculating distance matrices across 175,418 training instances with 77 features causes excessive memory consumption and slow inference times ($O(N \cdot D)$ per prediction), making it unsuitable for responsive API serving.

---

## 7. Benchmark Results, Evaluation & Model Selection

### 7.1 Comprehensive Model Comparison (Test Set: 58,473 Samples)

All candidate models were evaluated on the held-out stratified test split (**58,473 unseen patient records**). Below are the verified metrics:

| Model Architecture | Accuracy | Balanced Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | Fit Time (s) | Inference Time (s) | Status |
|---|---|---|---|---|---|---|---|---|---|
| **PyTorch Deep Learning** | **77.78%** | **77.72%** | **75.01%** | **83.97%** | **79.24%** | **86.93%** | 42.18s | 0.041s | **Active Production** |
| **Decision Tree (depth=10)** | 75.84% | 75.77% | 73.16% | 82.34% | 77.48% | 84.40% | 1.90s | 0.047s | **Sklearn Baseline** |
| **Random Forest (50 trees)** | 73.46% | 73.44% | 72.82% | 75.66% | 74.22% | 80.15% | 11.35s | 0.500s | Evaluated |
| **Support Vector Machine** | 71.70% | 71.69% | 71.45% | 73.18% | 72.31% | N/A* | 3.38s | 0.014s | Evaluated |
| **Logistic Regression** | 71.69% | 71.67% | 71.44% | 73.18% | 72.30% | 78.78% | 5.39s | 0.034s | Evaluated |
| **Gaussian Naive Bayes** | 54.85% | 54.41% | 52.79% | 100.00% | 69.10% | 54.41% | 0.26s | 0.072s | Evaluated |

*\*LinearSVC does not compute class probabilities, so its ROC-AUC is omitted.*

---

### 7.2 PyTorch Deep Learning In-Depth Metrics

Detailed test-set metrics for the production PyTorch model (`reports/metrics_pytorch_deep_learning.json`):

- **Matthews Correlation Coefficient (MCC)**: `0.5592` (indicates strong positive predictive correlation)
- **Cohen's Kappa ($\kappa$)**: `0.5551` (moderate-to-substantial inter-rater reliability agreement)
- **Cross-Entropy Log Loss**: `0.4258`
- **Brier Score**: `0.1458` (close to 0 indicates well-calibrated probabilities)
- **Expected Calibration Error (ECE)**: `0.0034` (near-zero probability drift)

#### Test Confusion Matrix (Total: 58,473 patients)
```
                        Actual Negative ("No")    Actual Positive ("Yes")
Predicted "No" (0)             20,693 (TN)               4,732 (FN)
Predicted "Yes" (1)             8,259 (FP)              24,789 (TP)
```
- **True Negatives (TN)**: 20,693
- **False Positives (FP)**: 8,259
- **False Negatives (FN)**: 4,732 (kept low by prioritizing Recall)
- **True Positives (TP)**: 24,789

---

### 7.3 Multi-Tier Selection Strategy (`backend/model_comparator.py`)

Model selection is automated using a multi-tier decision rule:
1. **Primary Metric**: Compare models by **Accuracy**.
2. **Tie-Breaking Rule**: If top models are within **0.5% (0.005)** accuracy:
   - Rank by **Recall (Sensitivity)** — in mental health screening, a false negative (missing someone who needs care) is clinically more critical than a false positive.
   - Rank by **F1-Score**.
   - Rank by **lower training time**.

**Outcome**: **PyTorch Deep Learning** achieved the highest overall accuracy (**77.78%**), the highest Recall (**83.97%**), and the highest ROC-AUC (**86.93%**), earning selection as the active production model. The Decision Tree classifier serves as the CPU fallback.

---

### 7.4 Visual Metric Artifacts (`plots/`)

The evaluation pipeline automatically exports publication-ready PNG figures to `plots/`:
- **`confusion_matrix_pytorch_deep_learning.png`**: Raw integer count matrix.
- **`confusion_matrix_normalized_pytorch_deep_learning.png`**: Normalized percentage matrix.
- **`roc_curve_pytorch_deep_learning.png`**: True Positive Rate vs. False Positive Rate across all discrimination thresholds ($\text{AUC} = 0.8693$).
- **`pr_curve_pytorch_deep_learning.png`**: Precision vs. Recall curve highlighting performance across classification cutoffs.

---

## 8. Recommendation Engine (Clinical Decision Support)

### 8.1 Tiered Recommendation Taxonomy

Predictions are fed into `RecommendationEngine` (`backend/recommendation_engine.py`), a deterministic, rule-based clinical intelligence engine.

```
                  Model Prediction & Confidence
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Risk Stratification: Minimal | Low | Moderate | High     │
├─────────────────────────────────────────────────────────────┤
│ 2. Immediate Clinical Actions (De-escalation / Crisis Help) │
├─────────────────────────────────────────────────────────────┤
│ 3. Professional Referral (Psychiatry, Clinical Psych, CBT) │
├─────────────────────────────────────────────────────────────┤
│ 4. Evidence-Based Self-Care (Sleep, Exercise, Nutrition)    │
├─────────────────────────────────────────────────────────────┤
│ 5. Verified Crisis Resources & Helplines (24/7 Access)      │
└─────────────────────────────────────────────────────────────┘
```

---

### 8.2 Supported Conditions & Knowledge Base Mapping

The engine supports multiple clinical risk profiles:

| Profile Key | Risk Level | Primary Clinical Actions | Professional Therapy Referrals | Verified Helplines |
|---|---|---|---|---|
| `yes` / `treatment` | `moderate_to_high` | Schedule consultation with a licensed mental health professional; contact crisis helpline if in acute distress. | Comprehensive psychiatric evaluation; Cognitive Behavioural Therapy (CBT). | iCall Helpline (India): 9152987821<br>Vandrevala Foundation: 1860-2662-345<br>US Lifeline: 988 |
| `no` / `no_treatment` | `minimal_to_low` | Maintain regular daily routines, consistent sleep hygiene, and balanced nutrition. | Periodic wellness check-ins during routine primary care visits. | WHO Mental Health Portal |
| `depression` | `high` | Reach out to a licensed clinician immediately; establish social contact with family. | Clinical Psychologist or Psychiatrist; CBT; Interpersonal Therapy (IPT). | iCall: 9152987821<br>Crisis Text Line: Text HOME to 741741 |
| `anxiety` | `moderate` | Practice slow diaphragmatic breathing (4-7-8 method); ground senses. | Psychologist trained in CBT or Acceptance & Commitment Therapy (ACT). | ADAA Resource Center<br>iCall: 9152987821 |
| `stress` | `low` | Take structured micro-breaks; enforce boundaries on work hours and screen time. | Workplace counselor or licensed counselor if symptoms persist > 2 weeks. | Vandrevala Foundation: 1860-2662-345 |
| `normal` | `minimal` | Continue proactive wellness habits, mindfulness, and gratitude journaling. | Routine annual check-in. | WHO Mental Health Portal |

---

## 9. REST API Specification & Endpoints

### 9.1 Endpoints Overview

The Flask API server (`app.py`) runs on port `5000` with full CORS support (`Access-Control-Allow-Origin: *`).

| Method | Route | Description | Auth |
|---|---|---|---|
| `GET` | `/` | API status, version, and route directory | None |
| `GET` | `/health` | Health check, predictor load state, and compute device telemetry | None |
| `GET` | `/api/info` | API metadata and documentation overview | None |
| `GET` | `/api/classes` | Supported class labels from the recommendation knowledge base | None |
| `GET` | `/api/metrics` | Verified test-set benchmark metrics (PyTorch vs Decision Tree) | None |
| `GET` | `/app` | Serves the interactive web chatbot interface | None |
| `GET` | `/frontend/<filename>` | Serves static frontend assets (HTML, CSS, JS) | None |
| `POST` | `/predict` | Primary prediction and recommendation endpoint | None |
| `POST` | `/api/predict` | API alias for `/predict` | None |

---

### 9.2 Detailed Endpoint Reference

#### `GET /health`
Returns system status, active compute device (`cpu` or `cuda`), and model loading state.

```json
{
  "status": "ok",
  "predictor_loaded": true,
  "device": "cpu",
  "timestamp": "2026-10-16T20:35:00Z"
}
```

#### `GET /api/metrics`
Returns verified offline benchmark metrics:

```json
{
  "production_model": "PyTorch Deep Learning",
  "pytorch": {
    "model_name": "PyTorch Deep Learning",
    "accuracy": 0.7778,
    "recall": 0.8397,
    "precision": 0.7501,
    "f1_score": 0.7924,
    "roc_auc": 0.8693
  },
  "sklearn_baseline": {
    "model_name": "Decision Tree",
    "accuracy": 0.7584,
    "recall": 0.8234,
    "precision": 0.7316,
    "f1_score": 0.7748,
    "roc_auc": 0.8440
  },
  "timestamp": "2026-10-16T20:35:00Z"
}
```

---

### 9.3 Request & Response Payloads

#### `POST /predict`
Runs inference on a patient record.

**Headers**:
```http
Content-Type: application/json
```

**Request Body (15 Clinical Predictor Features)**:
```json
{
  "gender": "Male",
  "country": "United States",
  "occupation": "Corporate",
  "self_employed": "No",
  "family_history": "Yes",
  "days_indoors": "1-14 days",
  "growing_stress": "Yes",
  "changes_habits": "Yes",
  "mental_health_history": "No",
  "mood_swings": "Medium",
  "coping_struggles": "Yes",
  "work_interest": "No",
  "social_weakness": "Yes",
  "mental_health_interview": "No",
  "care_options": "Not sure"
}
```

**Response Body (`200 OK`)**:
```json
{
  "success": true,
  "timestamp": "2026-10-16T20:35:12Z",
  "result": {
    "predicted_class": "Yes",
    "predicted_index": 1,
    "confidence": 0.9866,
    "probabilities": {
      "No": 0.0134,
      "Yes": 0.9866
    },
    "recommendation": {
      "predicted_class": "Yes",
      "risk_level": "moderate_to_high",
      "confidence": 0.9866,
      "immediate_actions": [
        "Schedule a consultation with a licensed mental health professional.",
        "If experiencing acute distress, contact a mental health crisis helpline immediately."
      ],
      "professional_referral": [
        "Consult a psychiatrist or clinical psychologist for a formal evaluation.",
        "Explore evidence-based therapies such as Cognitive Behavioural Therapy (CBT)."
      ],
      "self_care": [
        "Maintain a regular daily routine including consistent sleep hygiene and meals.",
        "Practice stress-reduction techniques such as mindfulness or structured breathing.",
        "Maintain strong social connections with trusted friends, family, or colleagues."
      ],
      "resources": [
        "iCall Helpline (India): 9152987821",
        "Vandrevala Foundation: 1860-2662-345",
        "988 Suicide & Crisis Lifeline (US): Call/Text 988"
      ],
      "disclaimer": "DISCLAIMER: These recommendations are for informational purposes only and do not constitute medical advice. Please consult a licensed mental health professional for a proper diagnosis and treatment plan."
    },
    "model_used": "PyTorch Deep Learning"
  }
}
```

---

## 10. Conversational Frontend Assistant

The web client in `frontend/` provides a guided clinical assessment interface.

```
┌─────────────────────────────────────────────────────────────┐
│                 MINDCARE AI CHATBOT INTERFACE               │
├─────────────────────────────────────────────────────────────┤
│  [Neural Avatar]                                            │
│  "Let's begin your assessment. What is your biological      │
│   sex or gender identity?"                                  │
│                                                             │
│  Options:  [ Male ]  [ Female ]                             │
│                                                             │
│  Progress: [■■■■■□□□□□□□□□□] 33% (Step 5 of 15)             │
│                                                             │
│  [ Type response or click option above...       ] [ Send ↵] │
└─────────────────────────────────────────────────────────────┘
```

### 10.1 UI/UX Architecture
- **Pure Vanilla Web Stack**: Built with semantic HTML5, CSS3, and modern ES6+ JavaScript — no frameworks, bundlers, or external dependencies required.
- **Cybernetic Dark Theme**: High-contrast dark palette (`#05070B`), glassmorphic panels, and glowing status indicators.
- **Responsive Layout**: Designed for seamless use across desktop, tablet, and mobile viewports.
- **Verified Benchmark Modal**: An interactive modal accessible from the header displays verified test-set metrics (PyTorch vs. Decision Tree baseline) across 58,473 test instances.

### 10.2 Conversational Flow & Dynamic Controls
- **15 Sequential Questions**: Guides the user through each diagnostic feature in order.
- **Smart Input Chips**: Dynamically generates clickable option chips above the text input for rapid, error-free responses.
- **Auto-Fill & Country Selector**: Includes a quick-select list of 35 countries.
- **Real-Time Progress Tracking**: Dynamic progress bar updates from 0% to 100% as questions are answered.
- **Rich Results View**: Displays predicted class, confidence gauge, risk tier badge, immediate actions, professional referrals, self-care steps, and emergency helplines.

---

## 11. Installation & Operation Manual

### 11.1 Prerequisites & Environment Setup

- **Python**: Version `3.12` or higher
- **Git**
- **Hardware**: CPU (Intel/AMD/Apple Silicon) or NVIDIA GPU with CUDA support

```bash
# 1. Clone repository
git clone https://github.com/Anirudh69btx/MEDAI.git
cd MEDAI

# 2. Create and activate a virtual environment
# Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 11.2 Training the Models

Run the complete training pipeline:

```bash
# Standard training (trains 5 Scikit-Learn models + PyTorch Deep Neural Network)
python train.py

# Optional: Train only Scikit-Learn models (skip PyTorch)
python train.py --no-pytorch

# Optional: Override the target column or dataset path
python train.py --target-column treatment --dataset-path dataset/data.csv
```

#### What happens during training:
1. Loads and validates `dataset/data.csv` (292,364 rows).
2. Performs a stratified 60/20/20 train/validation/test split.
3. Fits the preprocessor and feature engineering pipeline on the training split only.
4. Trains 5 Scikit-Learn candidate models using 3-fold stratified cross-validation.
5. Trains the PyTorch deep neural network with mixed precision and early stopping.
6. Evaluates all models on the 58,473 test instances, saving metric reports to `reports/` and curves to `plots/`.
7. `ModelComparator` selects the best model using the multi-tier selection rule.
8. Serializes all artifacts to `models/`.

---

### 11.3 Launching the API & Web App

Start the Flask server:

```bash
python app.py
```

The application will be available at:
- **Web Chatbot UI**: [`http://localhost:5000/app`](http://localhost:5000/app)
- **API Health Check**: [`http://localhost:5000/health`](http://localhost:5000/health)
- **Model Benchmarks Endpoint**: [`http://localhost:5000/api/metrics`](http://localhost:5000/api/metrics)

---

### 11.4 Verifying the API

#### Option A: PowerShell Script
```powershell
.\test_api.ps1
```

#### Option B: cURL
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "gender": "Female",
       "country": "United States",
       "occupation": "Corporate",
       "self_employed": "No",
       "family_history": "Yes",
       "days_indoors": "1-14 days",
       "growing_stress": "Yes",
       "changes_habits": "Yes",
       "mental_health_history": "Yes",
       "mood_swings": "High",
       "coping_struggles": "Yes",
       "work_interest": "No",
       "social_weakness": "Yes",
       "mental_health_interview": "No",
       "care_options": "Yes"
     }'
```

---

## 12. Engineering Notes, Gotchas & Troubleshooting

### 1. Schema Desynchronization (`raw_feature_columns.pkl` vs `feature_names.pkl`)
- **Issue**: Previously, `feature_names.pkl` was overwritten by both the preprocessor (15 features) and the model saver (77 features), causing shape mismatch errors during inference.
- **Fix**: The 15 raw feature column names are now stored exclusively in `raw_feature_columns.pkl`, while `feature_names.pkl` stores only the 77 model feature names. Both are required and loaded independently.

### 2. Feature Names Warning in Scikit-Learn
- **Issue**: Passing `X_test.values` (NumPy array) into a fitted Scikit-Learn estimator produces: `UserWarning: X does not have valid feature names...`.
- **Fix**: Data splits are maintained as `pd.DataFrame` with column names preserved when evaluating Scikit-Learn models. For PyTorch models, data is explicitly cast to `np.float32` tensors.

### 3. Missing Feature Imputation at Inference Time
- If a client sends an incomplete JSON payload (e.g., omitting `care_options`), `MindCarePredictor` automatically restores the missing key using the population median (for numeric fields) or mode (for categorical fields) saved in `feature_encoder.pkl`.

### 4. GPU vs. CPU Execution
- The system automatically selects CUDA if an NVIDIA GPU is present. If CUDA is unavailable, PyTorch automatically runs on CPU. Force CPU execution by setting `device="cpu"` in `backend/config.py`.

---

## 13. Privacy, Security & Ethical AI

- **100% Local Inference**: All data preprocessing, model inference, and recommendation generation happen locally. No survey responses or clinical profiles are transmitted to external cloud APIs or third parties.
- **Non-Diagnostic Tool**: MindCare AI is designed for **risk screening and decision support**, not definitive psychiatric diagnosis.
- **Fairness & Class Balance**: The training dataset maintains a near 1:1 balance (`50.49%` Yes vs. `49.51%` No), avoiding majority-class bias.
- **Recall Prioritization**: The system prioritizes recall over precision to minimize false negatives, ensuring individuals experiencing mental distress are not missed.

---

## 14. License & Clinical Disclaimer

### License
This project is licensed for educational, academic, and research purposes.

### Clinical Disclaimer
```
========================================================================================
                               CLINICAL & LEGAL DISCLAIMER
========================================================================================
MindCare AI is an artificial intelligence research application designed for informational,
educational, and screening support purposes only.

IT DOES NOT PROVIDE MEDICAL DIAGNOSES, CLINICAL JUDGMENTS, OR FORMAL TREATMENT PLANS.

The predictions and recommendations generated by this software MUST NOT replace consultation
with a qualified psychiatrist, clinical psychologist, or licensed physician.

If you or someone you know is in acute distress or experiencing a mental health crisis:
- In India: Call the iCall Helpline at 9152987821 or Vandrevala Foundation at 1860-2662-345
- In the US: Call or text 988 to reach the Suicide & Crisis Lifeline
- In the UK: Call 111 (NHS) or 999 in an emergency
- Worldwide: Contact your local emergency services or visit the nearest emergency room
========================================================================================
```
