import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

# Import your custom modules
from src.data_prep import load_data, clean_data, engineer_features, handle_outliers, normalize_features
from src.model import train_tuned_models, evaluate_models, stratified_split

# Page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .fraud-alert {
        background-color: #ffebee;
        border: 2px solid #f44336;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .safe-alert {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_models():
    """Load all trained models."""
    models = {}
    model_files = {
        'Logistic Regression': 'outputs/logistic_model.pkl',
        'Random Forest': 'outputs/random_forest_model.pkl',
        'XGBoost': 'outputs/xgboost_model.pkl'
    }
    
    for name, filepath in model_files.items():
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                models[name] = pickle.load(f)
    
    return models

@st.cache_data
def load_metrics():
    """Load model evaluation metrics."""
    if os.path.exists('outputs/metrics.json'):
        with open('outputs/metrics.json', 'r') as f:
            return json.load(f)
    return {}

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the dataset."""
    if os.path.exists('data/raw.csv'):
        df = load_data('data/raw.csv')
        df = clean_data(df)
        df = engineer_features(df)
        df = handle_outliers(df)
        return df
    return None

def get_feature_names(df):
    """Get the feature names used for training."""
    # This should match the feature selection logic in main.py
    feature_cols = [col for col in df.columns if col not in ['Class', 'Day', 'Hour', 'Trans_per_hour', 'Amount_log']]
    return feature_cols

def prepare_single_transaction(transaction_data, feature_names):
    """Prepare a single transaction for prediction."""
    # Create a DataFrame with the correct feature names
    df = pd.DataFrame([transaction_data], columns=feature_names)
    
    # Normalize Amount if it's in the features
    if 'Amount' in df.columns:
        # Use the same normalization as in training
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        # Note: In production, you'd load the fitted scaler from training
        df['Amount'] = scaler.fit_transform(df[['Amount']])
    
    return df

def display_prediction(transaction_df, model, model_name):
    """Display prediction results."""
    try:
        # Make prediction
        prediction = model.predict(transaction_df)[0]
        probability = model.predict_proba(transaction_df)[0]
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.markdown("""
                <div class="fraud-alert">
                    <h3>⚠️ FRAUD DETECTED</h3>
                    <p>This transaction is flagged as potentially fraudulent.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="safe-alert">
                    <h3>✅ TRANSACTION SAFE</h3>
                    <p>This transaction appears to be legitimate.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.metric(
                "Fraud Probability",
                f"{probability[1]:.2%}",
                delta=f"Model: {model_name}"
            )
            
            # Probability visualization
            fig = go.Figure(go.Bar(
                x=['Legitimate', 'Fraud'],
                y=[probability[0], probability[1]],
                marker_color=['green', 'red']
            ))
            fig.update_layout(
                title="Prediction Probabilities",
                yaxis_title="Probability",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.error("Please check that the feature names match those used during training.")

def main():
    st.markdown('<h1 class="main-header">🔍 Credit Card Fraud Detection</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "📊 Dataset Overview", "🔮 Make Prediction", "📈 Model Performance", "🎯 Batch Prediction"]
    )
    
    if page == "🏠 Home":
        st.markdown("""
        ## Welcome to the Credit Card Fraud Detection System
        
        This application uses machine learning to detect fraudulent credit card transactions.
        
        ### Features:
        - **Real-time Prediction**: Analyze individual transactions
        - **Multiple Models**: Compare Logistic Regression, Random Forest, and XGBoost
        - **Batch Processing**: Analyze multiple transactions at once
        - **Model Explanations**: SHAP values for interpretability
        
        ### How to Use:
        1. Navigate to **Make Prediction** to analyze a single transaction
        2. Check **Model Performance** to see how well our models work
        3. Use **Batch Prediction** for multiple transactions
        
        ### Dataset:
        The models are trained on credit card transaction data with 284,807 transactions and 492 fraud cases.
        """)
        
        # Load and display basic stats
        df = load_and_prepare_data()
        if df is not None:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Transactions", f"{len(df):,}")
            with col2:
                st.metric("Fraud Cases", f"{df['Class'].sum():,}")
            with col3:
                st.metric("Fraud Rate", f"{df['Class'].mean()*100:.3f}%")
            with col4:
                st.metric("Features", f"{len(df.columns)-1}")
    
    elif page == "📊 Dataset Overview":
        st.header("Dataset Overview")
        
        df = load_and_prepare_data()
        if df is not None:
            # Basic statistics
            st.subheader("Dataset Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Class Distribution**")
                class_counts = df['Class'].value_counts()
                fig = px.pie(values=class_counts.values, names=['Legitimate', 'Fraud'], 
                           title="Transaction Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Amount Distribution**")
                fig = px.histogram(df, x='Amount', nbins=50, title="Transaction Amount Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature correlation
            st.subheader("Feature Correlation with Fraud")
            numeric_features = df.select_dtypes(include=[np.number]).columns
            correlations = df[numeric_features].corr()['Class'].sort_values(key=abs, ascending=False)
            
            fig = px.bar(
                x=correlations.index[:20], 
                y=correlations.values[:20],
                title="Top 20 Features Correlated with Fraud"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("Dataset not found. Please ensure 'data/raw.csv' exists.")
    
    elif page == "🔮 Make Prediction":
        st.header("Make a Prediction")
        
        # Load models
        models = load_models()
        if not models:
            st.error("No trained models found. Please run the training pipeline first.")
            return
        
        # Load data to get feature names
        df = load_and_prepare_data()
        if df is None:
            st.error("Dataset not found. Please ensure 'data/raw.csv' exists.")
            return
        
        feature_names = get_feature_names(df)
        
        st.subheader("Enter Transaction Details")
        
        # Model selection
        model_name = st.selectbox("Select Model", list(models.keys()))
        selected_model = models[model_name]
        
        # Create input form
        with st.form("transaction_form"):
            col1, col2 = st.columns(2)
            
            transaction_data = {}
            
            with col1:
                st.subheader("Basic Information")
                transaction_data['Time'] = st.number_input("Time (seconds)", min_value=0, value=0)
                transaction_data['Amount'] = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=0.01)
            
            with col2:
                st.subheader("PCA Features")
                # For PCA features V1-V28, provide reasonable defaults
                for i in range(1, 29):
                    if f'V{i}' in feature_names:
                        transaction_data[f'V{i}'] = st.number_input(
                            f"V{i}", 
                            value=0.0, 
                            step=0.01, 
                            format="%.6f"
                        )
            
            submitted = st.form_submit_button("🔍 Analyze Transaction")
            
            if submitted:
                # Prepare transaction data
                transaction_df = prepare_single_transaction(transaction_data, feature_names)
                
                # Make prediction
                display_prediction(transaction_df, selected_model, model_name)
    
    elif page == "📈 Model Performance":
        st.header("Model Performance")
        
        # Load metrics
        metrics = load_metrics()
        if not metrics:
            st.error("No metrics found. Please run the training pipeline first.")
            return
        
        # Display metrics comparison
        st.subheader("Model Comparison")
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, model_metrics in metrics.items():
            comparison_data.append({
                'Model': model_name.title(),
                'Precision': model_metrics['precision'],
                'Recall': model_metrics['recall'],
                'F1-Score': model_metrics['f1_score'],
                'ROC-AUC': model_metrics['roc_auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display as table
        st.dataframe(comparison_df.round(4), use_container_width=True)
        
        # Visual comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Metrics comparison chart
            fig = px.bar(
                comparison_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score'),
                x='Model', y='Score', color='Metric',
                title="Model Performance Comparison",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # ROC-AUC comparison
            fig = px.bar(
                comparison_df, x='Model', y='ROC-AUC',
                title="ROC-AUC Comparison",
                color='ROC-AUC',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrices
        st.subheader("Confusion Matrices")
        cols = st.columns(len(metrics))
        
        for i, (model_name, model_metrics) in enumerate(metrics.items()):
            with cols[i]:
                cm = np.array(model_metrics['confusion_matrix'])
                fig = px.imshow(
                    cm, 
                    text_auto=True,
                    title=f"{model_name.title()} Confusion Matrix",
                    labels=dict(x="Predicted", y="Actual"),
                    x=['Legitimate', 'Fraud'],
                    y=['Legitimate', 'Fraud']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Display plots if they exist
        if os.path.exists('outputs/plots/roc_curves.png'):
            st.subheader("ROC Curves")
            st.image('outputs/plots/roc_curves.png', caption="ROC Curves Comparison")
    
    elif page == "🎯 Batch Prediction":
        st.header("Batch Prediction")
        
        # Load models
        models = load_models()
        if not models:
            st.error("No trained models found. Please run the training pipeline first.")
            return
        
        st.subheader("Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Load the uploaded data
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(batch_df)} transactions")
                
                # Display first few rows
                st.subheader("Data Preview")
                st.dataframe(batch_df.head())
                
                # Model selection
                model_name = st.selectbox("Select Model for Batch Prediction", list(models.keys()))
                selected_model = models[model_name]
                
                if st.button("🔍 Run Batch Prediction"):
                    # Prepare data (assuming it has the same structure as training data)
                    try:
                        # Remove Class column if it exists
                        if 'Class' in batch_df.columns:
                            X_batch = batch_df.drop('Class', axis=1)
                            y_true = batch_df['Class']
                        else:
                            X_batch = batch_df
                            y_true = None
                        
                        # Make predictions
                        predictions = selected_model.predict(X_batch)
                        probabilities = selected_model.predict_proba(X_batch)[:, 1]
                        
                        # Create results DataFrame
                        results_df = batch_df.copy()
                        results_df['Fraud_Prediction'] = predictions
                        results_df['Fraud_Probability'] = probabilities
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Transactions", len(results_df))
                        with col2:
                            st.metric("Predicted Fraud", int(predictions.sum()))
                        with col3:
                            st.metric("Fraud Rate", f"{predictions.mean()*100:.2f}%")
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.dataframe(results_df)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="fraud_predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Fraud probability distribution
                        st.subheader("Fraud Probability Distribution")
                        fig = px.histogram(
                            results_df, x='Fraud_Probability', 
                            title="Distribution of Fraud Probabilities"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error processing batch prediction: {str(e)}")
                        st.error("Please ensure the CSV file has the correct format.")
                        
            except Exception as e:
                st.error(f"Error loading CSV file: {str(e)}")

if __name__ == "__main__":
    main()
