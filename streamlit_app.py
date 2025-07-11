import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# Import custom modules
from src.data_prep import load_data, clean_data, engineer_features, handle_outliers, normalize_features
from src.model import train_tuned_models, evaluate_models, plot_roc_curves, save_metrics
from src.explain import explain_model

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

@st.cache_data
def load_metrics():
    """Load saved metrics from JSON file."""
    try:
        with open('outputs/metrics.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@st.cache_resource
def load_trained_models():
    """Load pre-trained models."""
    models = {}
    model_files = {
        'Logistic Regression': 'outputs/logistic_model.pkl',
        'Random Forest': 'outputs/rf_model.pkl',
        'XGBoost': 'outputs/xgb_model.pkl'
    }
    
    for name, filepath in model_files.items():
        try:
            with open(filepath, 'rb') as f:
                models[name] = pickle.load(f)
        except FileNotFoundError:
            st.warning(f"Model {name} not found. Please train models first.")
    
    return models

def simulate_transaction():
    """Create input form for simulating a transaction."""
    st.subheader("Simulate Transaction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input("Amount ($)", min_value=0.0, value=100.0)
        time_hours = st.slider("Time (hours)", 0, 23, 12)
        
    with col2:
        # Simplified feature inputs (in practice, V1-V28 would be PCA components)
        v1 = st.number_input("V1 (PCA Component)", value=0.0)
        v2 = st.number_input("V2 (PCA Component)", value=0.0)
    
    if st.button("Predict Transaction"):
        # Create feature vector (simplified)
        features = {
            'Time': time_hours * 3600,
            'V1': v1, 'V2': v2,
            'Amount': amount,
            'Class': 0  # Placeholder
        }
        
        # Add dummy values for other V features
        for i in range(3, 29):
            features[f'V{i}'] = 0.0
        
        transaction_df = pd.DataFrame([features])
        return transaction_df
    
    return None

def display_prediction(transaction_df, model, model_name):
    """Display prediction results."""
    if transaction_df is not None:
        # Prepare features
        X = transaction_df.drop('Class', axis=1)
        
        # Make prediction
        prediction = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0]
        
        # Display results
        st.subheader("Prediction Results")
        
        if prediction == 1:
            st.error(f"🚨 **FRAUD DETECTED** (Confidence: {prediction_proba[1]:.2%})")
        else:
            st.success(f"✅ **LEGITIMATE** (Confidence: {prediction_proba[0]:.2%})")
        
        # Show probability breakdown
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Legitimate Probability", f"{prediction_proba[0]:.2%}")
        with col2:
            st.metric("Fraud Probability", f"{prediction_proba[1]:.2%}")

def display_metrics(metrics):
    """Display model performance metrics."""
    st.subheader("Model Performance Metrics")
    
    if metrics:
        # Create metrics comparison
        metrics_df = pd.DataFrame(metrics).T
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        for i, (model_name, model_metrics) in enumerate(metrics.items()):
            with [col1, col2, col3, col4][i % 4]:
                st.metric(
                    f"{model_name} - F1 Score",
                    f"{model_metrics['f1_score']:.3f}"
                )
                st.metric(
                    f"{model_name} - ROC AUC",
                    f"{model_metrics['roc_auc']:.3f}"
                )
        
        # Display detailed metrics table
        st.dataframe(metrics_df.round(3))

def display_plots():
    """Display saved plots."""
    st.subheader("Model Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists('outputs/plots/roc_curves.png'):
            st.image('outputs/plots/roc_curves.png', caption="ROC Curves")
        else:
            st.info("ROC curves not available. Train models first.")
    
    with col2:
        if os.path.exists('outputs/plots/shap_summary.png'):
            st.image('outputs/plots/shap_summary.png', caption="SHAP Feature Importance")
        else:
            st.info("SHAP plots not available. Generate explanations first.")

def main():
    st.title("🔍 Credit Card Fraud Detection Dashboard")
    st.markdown("**Detect fraudulent transactions using machine learning**")
    
    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    models = load_trained_models()
    
    if models:
        selected_model_name = st.sidebar.selectbox(
            "Choose Model", 
            list(models.keys())
        )
        selected_model = models[selected_model_name]
    else:
        st.error("No trained models found. Please train models first.")
        return
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["💳 Predict Transaction", "📊 Model Metrics", "📈 Visualizations"])
    
    with tab1:
        transaction_df = simulate_transaction()
        if transaction_df is not None:
            display_prediction(transaction_df, selected_model, selected_model_name)
    
    with tab2:
        metrics = load_metrics()
        display_metrics(metrics)
    
    with tab3:
        display_plots()
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit • Powered by scikit-learn, XGBoost & SHAP*")

if __name__ == "__main__":
    main()
