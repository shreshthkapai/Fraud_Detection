import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_shap_explanations(model, X_train: pd.DataFrame, X_test: pd.DataFrame, sample_size: int = 1000):
    """Generate SHAP explanations for XGBoost model."""
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Sample data for faster computation
    X_train_sample = X_train.sample(n=min(sample_size, len(X_train)), random_state=42)
    X_test_sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)
    
    # Calculate SHAP values
    shap_values_train = explainer.shap_values(X_train_sample)
    shap_values_test = explainer.shap_values(X_test_sample)
    
    return explainer, shap_values_train, shap_values_test, X_train_sample, X_test_sample

def plot_global_shap_summary(shap_values, X_sample: pd.DataFrame):
    """Generate and save global SHAP summary plot."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    
    os.makedirs('outputs/plots', exist_ok=True)
    plt.savefig('outputs/plots/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_local_shap_explanation(explainer, X_sample: pd.DataFrame, instance_idx: int = 0):
    """Generate and save local SHAP explanation for a single prediction."""
    shap_values_single = explainer.shap_values(X_sample.iloc[instance_idx:instance_idx+1])
    
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_single[0],
            base_values=explainer.expected_value,
            data=X_sample.iloc[instance_idx].values,
            feature_names=X_sample.columns.tolist()
        ),
        show=False
    )
    
    os.makedirs('outputs/plots', exist_ok=True)
    plt.savefig('outputs/plots/shap_local.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(shap_values, X_sample: pd.DataFrame):
    """Generate and save feature importance plot based on SHAP values."""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    
    os.makedirs('outputs/plots', exist_ok=True)
    plt.savefig('outputs/plots/shap_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def explain_model(model, X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Complete SHAP explanation pipeline."""
    print("Generating SHAP explanations...")
    
    # Generate SHAP values
    explainer, shap_values_train, shap_values_test, X_train_sample, X_test_sample = generate_shap_explanations(
        model, X_train, X_test
    )
    
    # Generate plots
    print("Creating global SHAP summary plot...")
    plot_global_shap_summary(shap_values_test, X_test_sample)
    
    print("Creating local SHAP explanation...")
    plot_local_shap_explanation(explainer, X_test_sample, instance_idx=0)
    
    print("Creating feature importance plot...")
    plot_feature_importance(shap_values_test, X_test_sample)
    
    print("SHAP explanations saved to outputs/plots/")
    
    return explainer, shap_values_test, X_test_sample
