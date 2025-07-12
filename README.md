
# Fraud Detection System

This project is a comprehensive machine learning pipeline for detecting fraudulent credit card transactions. It includes data preprocessing, model training, evaluation, and a Streamlit-based interactive dashboard for real-time predictions and model explainability.

## Project Architecture

The architecture is designed to be modular and reproducible, following standard MLOps best practices.

```
[Data Ingest (creditcard.csv)] -> [Data Preprocessing (data_prep.py)] -> [Model Training (model.py)] -> [Model Evaluation & Explainability (explain.py)] -> [Streamlit App (streamlit_app.py)]
```

- **Data**: Raw transaction data is stored in `data/`.
- **Preprocessing**: `src/data_prep.py` handles cleaning, scaling, and feature engineering.
- **Modeling**: `src/model.py` trains Logistic Regression, Random Forest, and XGBoost models, tunes them, and saves the best performers.
- **Outputs**: `outputs/` stores trained models, performance metrics (`metrics.json`), and visualizations (ROC curves, SHAP plots).
- **Dashboard**: `streamlit_app.py` provides a user-friendly interface to interact with the models.

## Features and Methodologies

This project showcases a range of techniques for handling real-world imbalanced data, from feature engineering to model explainability.

### 1. Data Loading & Basic Cleaning (`src/data_prep.py`)

- **Objective**: Load and validate the dataset.
- **Key Actions**:
    - Loaded the `creditcard.csv` dataset from Kaggle.
    - Performed an initial exploration to understand its structure and, most importantly, the severe class imbalance (a tiny fraction of transactions are fraudulent).
    - Confirmed the dataset's cleanliness, requiring no significant missing value imputation.

### 2. Feature Engineering (`src/data_prep.py`)

- **Objective**: Enhance features to improve model performance.
- **Key Actions**:
    - **Outlier Capping**: Applied IQR-based capping to the `Amount` feature to reduce the influence of extreme values.
    - **Normalization**: Scaled the `Amount` feature using a `StandardScaler` to bring its range in line with the other PCA-transformed features.

### 3. Train/Test Split + Class Imbalance Handling (`src/model.py`)

- **Objective**: Prepare the data for training and address class imbalance.
- **Key Actions**:
    - **Stratified Splitting**: Used a `stratified` train/test split to ensure the same proportion of fraudulent transactions in both the training and testing sets.
    - **SMOTE (Synthetic Minority Over-sampling Technique)**: Implemented SMOTE to generate synthetic samples of the minority class (fraudulent transactions), creating a more balanced training set for the models.

### 4. Model Training + Hyperparameter Tuning (`src/model.py`)

- **Objective**: Train and optimize multiple models.
- **Key Actions**:
    - **Multi-Model Approach**: Trained three distinct models to compare their performance:
        - **Logistic Regression**: As a simple, interpretable baseline.
        - **Random Forest**: An ensemble of decision trees to capture complex interactions.
        - **XGBoost**: A powerful gradient boosting model known for its performance in competitive settings.
    - **Hyperparameter Tuning**: Utilized `GridSearchCV` to systematically search for the optimal hyperparameters for each model, ensuring they are well-tuned for this specific problem.

### 5. Model Evaluation & Metrics (`src/model.py` and `outputs/`)

- **Objective**: Evaluate model performance with appropriate metrics.
- **Key Actions**:
    - **Targeted Metrics**: Focused on **Precision, Recall, F1-score, and ROC-AUC**, as accuracy is a misleading metric for imbalanced datasets like fraud detection.
    - **Visualizations**: Generated and saved a confusion matrix and ROC curve to visually assess model performance.
    - **Reproducibility**: Saved all key performance metrics to `metrics.json` and the trained models to disk, allowing for easy reproduction and comparison of results.

### 6. Model Explainability with SHAP (`src/explain.py`)

- **Objective**: Understand and interpret the model's predictions.
- **Key Actions**:
    - **SHAP Integration**: Used the `shap` library with a `TreeExplainer` to analyze the best-performing XGBoost model.
    - **Global & Local Explanations**: Generated both global feature importance plots (to see which features are most influential overall) and local SHAP plots (to understand the reasoning behind a single prediction).
    - **Insight into "Why"**: This provides crucial insight into why the model flags a transaction as fraudulent, a key requirement for real-world fraud detection systems.

### 7. Streamlit Dashboard (`streamlit_app.py`)

- **Objective**: Create an interactive and user-friendly interface for the models.
- **Key Actions**:
    - **Interactive UI**: Built a Streamlit dashboard that allows users to:
        - Simulate a transaction or upload a file of transactions.
        - Select from the trained models (Logistic Regression, Random Forest, XGBoost).
    - **Comprehensive Output**: The dashboard displays the model's prediction, the SHAP explanation for that prediction, the model's confusion matrix and ROC curve, and a summary of the key performance metrics from `metrics.json`.

## Key Visuals

### Model Performance (ROC Curves)

*[Alt Text](outputs\plots\roc_curves.png)*

### Feature Importance (SHAP)

*(You can add your `shap_feature_importance.png` here)*

### Streamlit Dashboard

*(You can add a screenshot of your Streamlit app here)*

## How to Run

### 1. Setup

Clone the repository and install the required dependencies:

```bash
git clone <your-repo-url>
cd fraud-detection
pip install -r requirements.txt
```

### 2. Run the ML Pipeline

Execute the main script to run the entire pipeline from data preprocessing to model training and evaluation:

```bash
python main.py
```

This will:
- Process the data in `data/raw.csv`.
- Train the models specified in `src/model.py`.
- Save the trained models, metrics, and plots to the `outputs/` directory.

### 3. Launch the Streamlit Dashboard

To start the interactive dashboard, run:

```bash
streamlit run streamlit_app.py
```

Navigate to the local URL provided by Streamlit in your browser to interact with the application.

