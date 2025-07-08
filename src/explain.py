import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from typing import Dict, Any, List, Tuple, Optional

def load_trained_model(model_path: str):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded model object
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    return joblib.load(model_path)

def initialize_shap_explainer(model, X_train: pd.DataFrame, model_type: str = 'tree') -> shap.Explainer:
    """
    Initialize SHAP explainer for the given model type.
    
    Args:
        model: Trained model
        X_train: Training features for background data
        model_type: Type of explainer ('tree', 'linear', 'kernel')
        
    Returns:
        shap.Explainer: Initialized SHAP explainer
    """
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    elif model_type == 'linear':
        explainer = shap.LinearExplainer(model, X_train)
    elif model_type == 'kernel':
        # Use a sample of training data for kernel explainer (more efficient)
        background_sample = shap.sample(X_train, 100)
        explainer = shap.KernelExplainer(model.predict_proba, background_sample)
    else:
        raise ValueError("model_type must be 'tree', 'linear', or 'kernel'")
    
    return explainer

def calculate_shap_values(explainer: shap.Explainer, X_test: pd.DataFrame, 
                         model_type: str = 'tree') -> np.ndarray:
    """
    Calculate SHAP values for test data.
    
    Args:
        explainer: SHAP explainer object
        X_test: Test features
        model_type: Type of explainer used
        
    Returns:
        np.ndarray: SHAP values
    """
    if model_type == 'tree':
        shap_values = explainer.shap_values(X_test)
        # For binary classification, return values for positive class
        if isinstance(shap_values, list):
            return shap_values[1]  # Fraud class
        return shap_values
    else:
        return explainer.shap_values(X_test)

def create_global_summary_plot(shap_values: np.ndarray, X_test: pd.DataFrame, 
                              save_path: str = 'outputs/plots/') -> None:
    """
    Create global SHAP summary plot showing feature importance.
    
    Args:
        shap_values: SHAP values for test data
        X_test: Test features
        save_path: Directory to save plot
    """
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
    plt.title('SHAP Summary Plot - Global Feature Importance', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(f'{save_path}shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_global_bar_plot(shap_values: np.ndarray, X_test: pd.DataFrame,
                          save_path: str = 'outputs/plots/') -> None:
    """
    Create global SHAP bar plot showing mean absolute feature importance.
    
    Args:
        shap_values: SHAP values for test data
        X_test: Test features
        save_path: Directory to save plot
    """
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title('SHAP Bar Plot - Mean Absolute Feature Importance', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(f'{save_path}shap_bar_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_local_explanation_plot(shap_values: np.ndarray, X_test: pd.DataFrame, 
                                 instance_idx: int, save_path: str = 'outputs/plots/') -> None:
    """
    Create local SHAP explanation plot for a specific instance.
    
    Args:
        shap_values: SHAP values for test data
        X_test: Test features
        instance_idx: Index of instance to explain
        save_path: Directory to save plot
    """
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(shap.Explanation(values=shap_values[instance_idx], 
                                       base_values=shap_values.mean(axis=0), 
                                       data=X_test.iloc[instance_idx]))
    plt.title(f'SHAP Waterfall Plot - Instance {instance_idx}', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(f'{save_path}shap_waterfall_{instance_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_force_plot(explainer: shap.Explainer, shap_values: np.ndarray, 
                     X_test: pd.DataFrame, instance_idx: int, 
                     save_path: str = 'outputs/plots/') -> None:
    """
    Create SHAP force plot for a specific instance.
    
    Args:
        explainer: SHAP explainer object
        shap_values: SHAP values for test data
        X_test: Test features
        instance_idx: Index of instance to explain
        save_path: Directory to save plot
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Generate force plot
    force_plot = shap.force_plot(explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
                                shap_values[instance_idx],
                                X_test.iloc[instance_idx],
                                matplotlib=True,
                                show=False)
    
    plt.savefig(f'{save_path}shap_force_{instance_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()

def get_feature_importance_ranking(shap_values: np.ndarray, X_test: pd.DataFrame) -> pd.DataFrame:
    """
    Get feature importance ranking based on mean absolute SHAP values.
    
    Args:
        shap_values: SHAP values for test data
        X_test: Test features
        
    Returns:
        pd.DataFrame: Feature importance ranking
    """
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    return feature_importance

def create_feature_importance_plot(feature_importance: pd.DataFrame, top_n: int = 20,
                                  save_path: str = 'outputs/plots/') -> None:
    """
    Create feature importance plot from SHAP values.
    
    Args:
        feature_importance: Feature importance DataFrame
        top_n: Number of top features to show
        save_path: Directory to save plot
    """
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(top_n)
    
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.title(f'Top {top_n} Features by Mean Absolute SHAP Value', fontsize=16)
    plt.xlabel('Mean Absolute SHAP Value')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(f'{save_path}feature_importance_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_fraud_patterns(shap_values: np.ndarray, X_test: pd.DataFrame, 
                          y_test: pd.Series) -> Dict[str, Any]:
    """
    Analyze SHAP patterns for fraud vs normal transactions.
    
    Args:
        shap_values: SHAP values for test data
        X_test: Test features
        y_test: Test target
        
    Returns:
        dict: Analysis results
    """
    fraud_mask = y_test == 1
    normal_mask = y_test == 0
    
    fraud_shap_mean = shap_values[fraud_mask].mean(axis=0)
    normal_shap_mean = shap_values[normal_mask].mean(axis=0)
    
    # Calculate difference in SHAP contributions
    shap_diff = fraud_shap_mean - normal_shap_mean
    
    # Get top features that distinguish fraud from normal
    feature_diff = pd.DataFrame({
        'feature': X_test.columns,
        'fraud_shap_mean': fraud_shap_mean,
        'normal_shap_mean': normal_shap_mean,
        'difference': shap_diff
    }).sort_values('difference', key=abs, ascending=False)
    
    analysis = {
        'fraud_patterns': feature_diff.head(10),
        'top_fraud_drivers': feature_diff.head(5),
        'top_normal_drivers': feature_diff.tail(5)
    }
    
    return analysis

def create_fraud_pattern_plot(analysis: Dict[str, Any], save_path: str = 'outputs/plots/') -> None:
    """
    Create plot showing different SHAP patterns for fraud vs normal transactions.
    
    Args:
        analysis: Analysis results from analyze_fraud_patterns
        save_path: Directory to save plot
    """
    os.makedirs(save_path, exist_ok=True)
    
    fraud_patterns = analysis['fraud_patterns'].head(15)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Fraud vs Normal SHAP contributions
    x = np.arange(len(fraud_patterns))
    width = 0.35
    
    ax1.bar(x - width/2, fraud_patterns['fraud_shap_mean'], width, 
            label='Fraud Transactions', color='red', alpha=0.7)
    ax1.bar(x + width/2, fraud_patterns['normal_shap_mean'], width,
            label='Normal Transactions', color='blue', alpha=0.7)
    
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Mean SHAP Value')
    ax1.set_title('Mean SHAP Values: Fraud vs Normal Transactions')
    ax1.set_xticks(x)
    ax1.set_xticklabels(fraud_patterns['feature'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Difference in SHAP contributions
    colors = ['red' if x > 0 else 'blue' for x in fraud_patterns['difference']]
    ax2.bar(x, fraud_patterns['difference'], color=colors, alpha=0.7)
    ax2.set_xlabel('Features')
    ax2.set_ylabel('SHAP Difference (Fraud - Normal)')
    ax2.set_title('Features Most Discriminative for Fraud Detection')
    ax2.set_xticks(x)
    ax2.set_xticklabels(fraud_patterns['feature'], rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}fraud_pattern_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def find_interesting_instances(shap_values: np.ndarray, X_test: pd.DataFrame, 
                              y_test: pd.Series, y_pred_proba: np.ndarray) -> Dict[str, int]:
    """
    Find interesting instances for local explanation.
    
    Args:
        shap_values: SHAP values for test data
        X_test: Test features
        y_test: Test target
        y_pred_proba: Predicted probabilities
        
    Returns:
        dict: Indices of interesting instances
    """
    # High confidence correct fraud prediction
    fraud_mask = y_test == 1
    high_conf_fraud = np.where(fraud_mask & (y_pred_proba > 0.9))[0]
    
    # High confidence correct normal prediction
    normal_mask = y_test == 0
    high_conf_normal = np.where(normal_mask & (y_pred_proba < 0.1))[0]
    
    # False positive (predicted fraud, actually normal)
    false_positive = np.where(normal_mask & (y_pred_proba > 0.5))[0]
    
    # False negative (predicted normal, actually fraud)
    false_negative = np.where(fraud_mask & (y_pred_proba < 0.5))[0]
    
    # Edge cases (predictions around 0.5)
    edge_cases = np.where((y_pred_proba > 0.4) & (y_pred_proba < 0.6))[0]
    
    interesting_instances = {
        'high_conf_fraud': high_conf_fraud[0] if len(high_conf_fraud) > 0 else None,
        'high_conf_normal': high_conf_normal[0] if len(high_conf_normal) > 0 else None,
        'false_positive': false_positive[0] if len(false_positive) > 0 else None,
        'false_negative': false_negative[0] if len(false_negative) > 0 else None,
        'edge_case': edge_cases[0] if len(edge_cases) > 0 else None
    }
    
    return interesting_instances

def create_all_local_explanations(explainer: shap.Explainer, shap_values: np.ndarray,
                                 X_test: pd.DataFrame, y_test: pd.Series,
                                 y_pred_proba: np.ndarray, save_path: str = 'outputs/plots/') -> None:
    """
    Create local explanations for interesting instances.
    
    Args:
        explainer: SHAP explainer object
        shap_values: SHAP values for test data
        X_test: Test features
        y_test: Test target
        y_pred_proba: Predicted probabilities
        save_path: Directory to save plots
    """
    interesting_instances = find_interesting_instances(shap_values, X_test, y_test, y_pred_proba)
    
    for instance_type, idx in interesting_instances.items():
        if idx is not None:
            create_local_explanation_plot(shap_values, X_test, idx, save_path)
            
            # Add context information
            actual_label = "Fraud" if y_test.iloc[idx] == 1 else "Normal"
            predicted_prob = y_pred_proba[idx]
            
            print(f"{instance_type}: Instance {idx}")
            print(f"  Actual: {actual_label}")
            print(f"  Predicted Probability: {predicted_prob:.3f}")
            print(f"  Plot saved: shap_waterfall_{idx}.png")

def generate_shap_summary_report(feature_importance: pd.DataFrame, 
                               analysis: Dict[str, Any], 
                               save_path: str = 'outputs/') -> None:
    """
    Generate text summary report of SHAP analysis.
    
    Args:
        feature_importance: Feature importance DataFrame
        analysis: Fraud pattern analysis results
        save_path: Directory to save report
    """
    os.makedirs(save_path, exist_ok=True)
    
    report = f"""
# SHAP Explainability Report - Fraud Detection Model

## Top 10 Most Important Features (Global)
{feature_importance.head(10).to_string(index=False)}

## Top 5 Features that Drive Fraud Predictions
{analysis['top_fraud_drivers'].to_string(index=False)}

## Top 5 Features that Drive Normal Predictions  
{analysis['top_normal_drivers'].to_string(index=False)}

## Key Insights
1. **Most Discriminative Feature**: {analysis['fraud_patterns'].iloc[0]['feature']} 
   - Fraud Impact: {analysis['fraud_patterns'].iloc[0]['fraud_shap_mean']:.4f}
   - Normal Impact: {analysis['fraud_patterns'].iloc[0]['normal_shap_mean']:.4f}

2. **Strongest Fraud Indicator**: {analysis['top_fraud_drivers'].iloc[0]['feature']}
   - Difference: {analysis['top_fraud_drivers'].iloc[0]['difference']:.4f}

3. **Strongest Normal Indicator**: {analysis['top_normal_drivers'].iloc[0]['feature']}
   - Difference: {analysis['top_normal_drivers'].iloc[0]['difference']:.4f}

## Files Generated
- shap_summary_plot.png: Global feature importance
- shap_bar_plot.png: Mean absolute importance
- feature_importance_ranking.png: Top features ranked
- fraud_pattern_analysis.png: Fraud vs normal patterns
- shap_waterfall_*.png: Local explanations for interesting cases
"""
    
    with open(f'{save_path}shap_report.txt', 'w') as f:
        f.write(report)
    
    print(f"SHAP summary report saved to {save_path}shap_report.txt")

def explain_model(model, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                 y_test: pd.Series, y_pred_proba: np.ndarray,
                 model_type: str = 'tree') -> Dict[str, Any]:
    """
    Main function to generate comprehensive model explanations.
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        y_test: Test target
        y_pred_proba: Predicted probabilities
        model_type: Type of model ('tree', 'linear', 'kernel')
        
    Returns:
        dict: SHAP analysis results
    """
    print("Initializing SHAP explainer...")
    explainer = initialize_shap_explainer(model, X_train, model_type)
    
    print("Calculating SHAP values...")
    shap_values = calculate_shap_values(explainer, X_test, model_type)
    
    print("Creating global explanations...")
    create_global_summary_plot(shap_values, X_test)
    create_global_bar_plot(shap_values, X_test)
    
    print("Analyzing feature importance...")
    feature_importance = get_feature_importance_ranking(shap_values, X_test)
    create_feature_importance_plot(feature_importance)
    
    print("Analyzing fraud patterns...")
    analysis = analyze_fraud_patterns(shap_values, X_test, y_test)
    create_fraud_pattern_plot(analysis)
    
    print("Creating local explanations...")
    create_all_local_explanations(explainer, shap_values, X_test, y_test, y_pred_proba)
    
    print("Generating summary report...")
    generate_shap_summary_report(feature_importance, analysis)
    
    print("✅ SHAP analysis complete! Check outputs/plots/ for visualizations.")
    
    return {
        'shap_values': shap_values,
        'feature_importance': feature_importance,
        'fraud_analysis': analysis,
        'explainer': explainer
    }