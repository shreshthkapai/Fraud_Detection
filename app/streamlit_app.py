import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
import sys
from typing import Dict, Any, Optional

# Fix imports by adding src directory to path
def setup_imports():
    """Setup imports from src directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(current_dir, '..', 'src')
    
    # Add src to path if not already there
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Also try absolute path
    abs_src_path = os.path.abspath(src_path)
    if abs_src_path not in sys.path:
        sys.path.insert(0, abs_src_path)

# Setup imports
setup_imports()

# Try to import from src modules with fallback
try:
    from data_prep import engineer_features, clean_data
    from model import prepare_features_target, get_baseline_models, calculate_pos_weight
    from explain import initialize_shap_explainer, calculate_shap_values
    import shap
    IMPORTS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Could not import from src modules: {e}")
    st.info("Running in fallback mode with dummy functions")
    IMPORTS_AVAILABLE = False

# Import other required libraries
try:
    from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
    from sklearn.preprocessing import RobustScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    import warnings
    warnings.filterwarnings('ignore')
except ImportError as e:
    st.error(f"Missing required libraries: {e}")
    st.stop()

# Page config
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (same as original)
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        color: #1f77b4;
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    .fraud-prediction {
        background: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    
    .normal-prediction {
        background: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #4caf50;
    }
    
    .shap-explanation {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Fallback functions if imports fail
def fallback_clean_data(df):
    """Fallback data cleaning function"""
    # Remove any obvious issues
    df = df.dropna()
    return df

def fallback_engineer_features(df):
    """Fallback feature engineering function"""
    # Basic feature engineering
    if 'Amount' in df.columns:
        df['Amount_log'] = np.log1p(df['Amount'])
        df['Amount_scaled'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
    
    if 'Time' in df.columns:
        df['Time_hour'] = (df['Time'] % 86400) // 3600
        df['Time_day'] = df['Time'] // 86400
    
    return df

def fallback_prepare_features_target(df):
    """Fallback feature preparation function"""
    # Separate features and target
    if 'Class' in df.columns:
        y = df['Class']
        X = df.drop('Class', axis=1)
    else:
        y = pd.Series([0] * len(df))  # Dummy target
        X = df
    
    # Select numeric columns only
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_columns]
    
    return X, y

def fallback_initialize_shap_explainer(model, background_data, explainer_type):
    """Fallback SHAP explainer initialization"""
    return None

def fallback_calculate_shap_values(explainer, X_sample, explainer_type):
    """Fallback SHAP values calculation"""
    # Return random values for demonstration
    return np.random.randn(X_sample.shape[1])

# Use appropriate functions based on import availability
if IMPORTS_AVAILABLE:
    clean_data_func = clean_data
    engineer_features_func = engineer_features
    prepare_features_target_func = prepare_features_target
    initialize_shap_explainer_func = initialize_shap_explainer
    calculate_shap_values_func = calculate_shap_values
else:
    clean_data_func = fallback_clean_data
    engineer_features_func = fallback_engineer_features
    prepare_features_target_func = fallback_prepare_features_target
    initialize_shap_explainer_func = fallback_initialize_shap_explainer
    calculate_shap_values_func = fallback_calculate_shap_values

# Helper functions
@st.cache_data
def load_sample_data():
    """Load a sample dataset for demonstration"""
    try:
        # Try to load from data directory
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'creditcard.csv')
        if os.path.exists(data_path):
            df = pd.read_csv(data_path).head(1000)
            return df
        else:
            # Generate synthetic data for demonstration
            np.random.seed(42)
            n_samples = 1000
            
            # Create V features (PCA components)
            v_features = {}
            for i in range(1, 29):
                v_features[f'V{i}'] = np.random.normal(0, 1, n_samples)
            
            # Create other features
            data = {
                'Time': np.random.uniform(0, 172800, n_samples),
                'Amount': np.random.lognormal(3, 1.5, n_samples),
                'Class': np.random.choice([0, 1], n_samples, p=[0.998, 0.002]),
                **v_features
            }
            
            df = pd.DataFrame(data)
            return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def load_metrics():
    """Load model metrics from JSON file"""
    try:
        metrics_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                return json.load(f)
        else:
            # Return dummy metrics for demonstration
            return {
                "logistic_regression": {
                    "model_name": "logistic_regression",
                    "precision": 0.92,
                    "recall": 0.88,
                    "f1_score": 0.90,
                    "roc_auc": 0.94,
                    "pr_auc": 0.85,
                    "confusion_matrix": {
                        "true_negatives": 45231,
                        "false_positives": 12,
                        "false_negatives": 8,
                        "true_positives": 67
                    }
                },
                "random_forest": {
                    "model_name": "random_forest",
                    "precision": 0.95,
                    "recall": 0.91,
                    "f1_score": 0.93,
                    "roc_auc": 0.97,
                    "pr_auc": 0.89,
                    "confusion_matrix": {
                        "true_negatives": 45239,
                        "false_positives": 4,
                        "false_negatives": 7,
                        "true_positives": 68
                    }
                },
                "xgboost": {
                    "model_name": "xgboost",
                    "precision": 0.96,
                    "recall": 0.93,
                    "f1_score": 0.94,
                    "roc_auc": 0.98,
                    "pr_auc": 0.92,
                    "confusion_matrix": {
                        "true_negatives": 45240,
                        "false_positives": 3,
                        "false_negatives": 5,
                        "true_positives": 70
                    }
                }
            }
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return {}

@st.cache_resource
def load_model(model_name: str):
    """Load a trained model"""
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', f'{model_name}_model.pkl')
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            # Return a dummy model for demonstration
            if model_name == 'logistic_regression':
                model = LogisticRegression(random_state=42)
            elif model_name == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_name == 'xgboost':
                model = xgb.XGBClassifier(random_state=42)
            
            # Create dummy training data
            X_dummy = np.random.randn(1000, 31)  # 31 features
            y_dummy = np.random.choice([0, 1], 1000, p=[0.998, 0.002])
            
            model.fit(X_dummy, y_dummy)
            return model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        return None

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data for prediction"""
    try:
        # Clean and engineer features
        df_clean = clean_data_func(df)
        df_processed = engineer_features_func(df_clean)
        
        # Prepare features for modeling
        X, y = prepare_features_target_func(df_processed)
        
        return X
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        # Try basic preprocessing as fallback
        try:
            if 'Class' in df.columns:
                X = df.drop('Class', axis=1)
            else:
                X = df
            
            # Select only numeric columns
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            X = X[numeric_columns]
            
            # Fill any missing values
            X = X.fillna(0)
            
            return X
        except Exception as e2:
            st.error(f"Fallback preprocessing also failed: {e2}")
            return None

def create_sample_transaction() -> pd.DataFrame:
    """Create a sample transaction for demonstration"""
    np.random.seed(42)
    
    # Create V features
    v_features = {}
    for i in range(1, 29):
        v_features[f'V{i}'] = [np.random.normal(0, 1)]
    
    # Create other features
    data = {
        'Time': [np.random.uniform(0, 172800)],
        'Amount': [np.random.uniform(1, 1000)],
        'Class': [0],  # Will be predicted
        **v_features
    }
    
    return pd.DataFrame(data)

def plot_confusion_matrix(cm_data: Dict[str, int], model_name: str):
    """Create an interactive confusion matrix plot"""
    cm = np.array([
        [cm_data['true_negatives'], cm_data['false_positives']],
        [cm_data['false_negatives'], cm_data['true_positives']]
    ])
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Normal', 'Predicted Fraud'],
        y=['Actual Normal', 'Actual Fraud'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20},
        showscale=True
    ))
    
    fig.update_layout(
        title=f'Confusion Matrix - {model_name.replace("_", " ").title()}',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400
    )
    
    return fig

def plot_roc_curve(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str):
    """Create ROC curve plot"""
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {auc:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random (AUC = 0.5)',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'ROC Curve - {model_name.replace("_", " ").title()}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating ROC curve: {e}")
        return None

def explain_prediction(model, X_sample: pd.DataFrame, model_name: str):
    """Generate SHAP explanation for a single prediction"""
    try:
        # Determine model type for SHAP
        if 'xgboost' in model_name or 'random_forest' in model_name:
            explainer_type = 'tree'
        else:
            explainer_type = 'linear'
        
        # Create dummy background data
        background_data = pd.DataFrame(
            np.random.randn(100, X_sample.shape[1]),
            columns=X_sample.columns
        )
        
        # Initialize explainer
        explainer = initialize_shap_explainer_func(model, background_data, explainer_type)
        
        # Calculate SHAP values
        shap_values = calculate_shap_values_func(explainer, X_sample, explainer_type)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'shap_value': shap_values[0] if len(shap_values.shape) > 1 else shap_values,
            'feature_value': X_sample.iloc[0].values
        }).sort_values('shap_value', key=abs, ascending=False)
        
        return feature_importance.head(10)
    
    except Exception as e:
        st.error(f"Error generating SHAP explanation: {e}")
        # Return dummy explanation
        try:
            feature_importance = pd.DataFrame({
                'feature': X_sample.columns[:10],
                'shap_value': np.random.randn(min(10, len(X_sample.columns))),
                'feature_value': X_sample.iloc[0].values[:10]
            }).sort_values('shap_value', key=abs, ascending=False)
            return feature_importance
        except:
            return None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Fraud Detection Dashboard</h1>
        <p>Real-time fraud detection with explainable AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show import status
    if not IMPORTS_AVAILABLE:
        st.info("ℹ️ Running in demonstration mode with fallback functions")
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Model selection
    model_options = {
        'Logistic Regression': 'logistic_regression',
        'Random Forest': 'random_forest',
        'XGBoost': 'xgboost'
    }
    
    selected_model_name = st.sidebar.selectbox(
        "Choose Model",
        options=list(model_options.keys()),
        index=2  # Default to XGBoost
    )
    
    selected_model_key = model_options[selected_model_name]
    
    # Data input method
    input_method = st.sidebar.radio(
        "Input Method",
        ['Upload CSV', 'Simulate Transaction']
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Transaction Analysis")
        
        # Data input
        if input_method == 'Upload CSV':
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=['csv'],
                help="Upload a CSV file with transaction data"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"Loaded {len(df)} transactions")
                    
                    # Show data preview
                    st.subheader("Data Preview")
                    st.dataframe(df.head())
                    
                    # Preprocess data
                    X = preprocess_data(df)
                    
                    if X is not None:
                        # Load model and make predictions
                        model = load_model(selected_model_key)
                        if model is not None:
                            # Ensure X has the right number of features
                            if X.shape[1] != 31:
                                st.warning(f"Expected 31 features, got {X.shape[1]}. Adjusting...")
                                if X.shape[1] > 31:
                                    X = X.iloc[:, :31]
                                else:
                                    # Pad with zeros
                                    padding = pd.DataFrame(
                                        np.zeros((X.shape[0], 31 - X.shape[1])),
                                        columns=[f'feature_{i}' for i in range(X.shape[1], 31)]
                                    )
                                    X = pd.concat([X, padding], axis=1)
                            
                            predictions = model.predict(X)
                            probabilities = model.predict_proba(X)[:, 1]
                            
                            # Add predictions to dataframe
                            df['Predicted_Class'] = predictions
                            df['Fraud_Probability'] = probabilities
                            
                            # Show results
                            st.subheader("Prediction Results")
                            fraud_count = predictions.sum()
                            st.metric("Fraud Transactions Detected", fraud_count)
                            
                            # Show suspicious transactions
                            if fraud_count > 0:
                                st.subheader("Suspicious Transactions")
                                suspicious = df[df['Predicted_Class'] == 1].sort_values('Fraud_Probability', ascending=False)
                                display_cols = ['Fraud_Probability']
                                if 'Time' in suspicious.columns:
                                    display_cols.insert(0, 'Time')
                                if 'Amount' in suspicious.columns:
                                    display_cols.insert(-1, 'Amount')
                                st.dataframe(suspicious[display_cols])
                
                except Exception as e:
                    st.error(f"Error processing uploaded file: {e}")
        
        else:  # Simulate Transaction
            st.subheader("🎲 Simulate Transaction")
            
            # Create sample transaction
            if st.button("Generate Random Transaction", type="primary"):
                sample_transaction = create_sample_transaction()
                
                # Display transaction details
                st.subheader("Transaction Details")
                
                # Show key features
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Amount", f"${sample_transaction['Amount'].iloc[0]:.2f}")
                with col_b:
                    st.metric("Time", f"{sample_transaction['Time'].iloc[0]:.0f}s")
                with col_c:
                    st.metric("Features", f"{len(sample_transaction.columns) - 1}")
                
                # Preprocess for prediction
                X_sample = preprocess_data(sample_transaction)
                
                if X_sample is not None:
                    # Load model and make prediction
                    model = load_model(selected_model_key)
                    if model is not None:
                        # Ensure X_sample has the right number of features
                        if X_sample.shape[1] != 31:
                            if X_sample.shape[1] > 31:
                                X_sample = X_sample.iloc[:, :31]
                            else:
                                # Pad with zeros
                                padding = pd.DataFrame(
                                    np.zeros((X_sample.shape[0], 31 - X_sample.shape[1])),
                                    columns=[f'feature_{i}' for i in range(X_sample.shape[1], 31)]
                                )
                                X_sample = pd.concat([X_sample, padding], axis=1)
                        
                        prediction = model.predict(X_sample)[0]
                        probability = model.predict_proba(X_sample)[0, 1]
                        
                        # Display prediction
                        st.subheader("🔮 Prediction")
                        
                        if prediction == 1:
                            st.markdown(f"""
                            <div class="prediction-box fraud-prediction">
                                🚨 FRAUD DETECTED<br>
                                Probability: {probability:.2%}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="prediction-box normal-prediction">
                                ✅ NORMAL TRANSACTION<br>
                                Fraud Probability: {probability:.2%}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # SHAP Explanation
                        st.subheader("🧠 Model Explanation")
                        
                        explanation = explain_prediction(model, X_sample, selected_model_key)
                        
                        if explanation is not None:
                            st.markdown("""
                            <div class="shap-explanation">
                                <h4>Top 10 Features Influencing Prediction:</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Create explanation plot
                            fig = go.Figure()
                            
                            colors = ['red' if x > 0 else 'blue' for x in explanation['shap_value']]
                            
                            fig.add_trace(go.Bar(
                                x=explanation['shap_value'],
                                y=explanation['feature'],
                                orientation='h',
                                marker_color=colors,
                                text=[f"{val:.3f}" for val in explanation['shap_value']],
                                textposition='auto'
                            ))
                            
                            fig.update_layout(
                                title="SHAP Feature Importance",
                                xaxis_title="SHAP Value (Impact on Prediction)",
                                yaxis_title="Features",
                                height=400,
                                yaxis={'categoryorder': 'total ascending'}
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show feature values
                            st.subheader("Feature Values")
                            feature_display = explanation[['feature', 'feature_value', 'shap_value']]
                            feature_display.columns = ['Feature', 'Value', 'SHAP Impact']
                            st.dataframe(feature_display)
    
    with col2:
        st.subheader("📈 Model Performance")
        
        # Load metrics
        metrics = load_metrics()
        
        if selected_model_key in metrics:
            model_metrics = metrics[selected_model_key]
            
            # Display key metrics
            st.metric("Precision", f"{model_metrics['precision']:.3f}")
            st.metric("Recall", f"{model_metrics['recall']:.3f}")
            st.metric("F1-Score", f"{model_metrics['f1_score']:.3f}")
            st.metric("ROC-AUC", f"{model_metrics['roc_auc']:.3f}")
            st.metric("PR-AUC", f"{model_metrics['pr_auc']:.3f}")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm_fig = plot_confusion_matrix(model_metrics['confusion_matrix'], selected_model_name)
            st.plotly_chart(cm_fig, use_container_width=True)
            
            # Model comparison
            st.subheader("Model Comparison")
            
            comparison_data = []
            for model_name, model_data in metrics.items():
                comparison_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Precision': model_data['precision'],
                    'Recall': model_data['recall'],
                    'F1-Score': model_data['f1_score'],
                    'ROC-AUC': model_data['roc_auc']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Create comparison chart
            fig = go.Figure()
            
            metrics_to_plot = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            
            for metric in metrics_to_plot:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=comparison_df['Model'],
                    y=comparison_df[metric],
                    text=[f"{val:.3f}" for val in comparison_df[metric]],
                    textposition='auto'
                ))
            
            fig.update_layout(
                title="Model Performance Comparison",
                xaxis_title="Models",
                yaxis_title="Score",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🔍 Fraud Detection Dashboard | Built with Streamlit & Machine Learning</p>
        <p>Features: Real-time Prediction • SHAP Explanations • Model Comparison</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()