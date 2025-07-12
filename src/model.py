import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import optuna
import pickle
import json
import matplotlib.pyplot as plt
import os

def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """Split data with stratification to preserve class balance."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def handle_imbalance(X_train: pd.DataFrame, y_train: pd.Series, method: str = 'smote') -> tuple:
    """Handle class imbalance using different sampling techniques."""
    
    if method == 'smote':
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    elif method == 'undersample':
        undersampler = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
    
    elif method == 'none':
        X_resampled, y_resampled = X_train, y_train
    
    return X_resampled, y_resampled

def train_with_class_weights(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Train models with balanced class weights."""
    models = {}
    
    # Logistic Regression with balanced class weights
    models['logistic'] = LogisticRegression(class_weight='balanced', random_state=42)
    models['logistic'].fit(X_train, y_train)
    
    # Random Forest with balanced class weights
    models['random_forest'] = RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced', 
        random_state=42
    )
    models['random_forest'].fit(X_train, y_train)
    
    return models

def tune_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Tune logistic regression with proper scaling and convergence."""
    
    # Scale features for better convergence
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],            
        'solver': ['liblinear', 'lbfgs'],  # Added liblinear as alternative
        'max_iter': [5000]  # Increased iterations
    }
    
    lr = LogisticRegression(
        class_weight='balanced', 
        random_state=42
    )

    grid_search = GridSearchCV(lr, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    # Store scaler with the model
    best_model = grid_search.best_estimator_
    best_model.scaler = scaler
    
    return best_model

def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Tune random forest with optimized parameters for faster execution."""
    param_grid = {
        'n_estimators': [50, 100],         
        'max_depth': [10, 15],             
        'min_samples_split': [2, 5]         
    }
    
    rf = RandomForestClassifier(
        class_weight='balanced', 
        random_state=42,
        n_jobs=-1
    )
    
    grid_search = GridSearchCV(
        rf, param_grid, 
        cv=3,
        scoring='roc_auc', 
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_

def tune_xgboost_optuna(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 20) -> xgb.XGBClassifier:
    """Tune XGBoost with Optuna - optimized for faster execution."""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_train)[:, 1]
        
        return roc_auc_score(y_train, y_pred_proba)
    
    # Suppress optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_params['scale_pos_weight'] = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    best_params['random_state'] = 42
    best_params['eval_metric'] = 'logloss'
    
    return xgb.XGBClassifier(**best_params)

def train_tuned_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Train and tune all models."""
    models = {}
    
    print("Tuning Logistic Regression...")
    models['logistic'] = tune_logistic_regression(X_train, y_train)
    
    print("Tuning Random Forest...")
    models['random_forest'] = tune_random_forest(X_train, y_train)
    
    print("Tuning XGBoost...")
    models['xgboost'] = tune_xgboost_optuna(X_train, y_train)
    models['xgboost'].fit(X_train, y_train)
    
    return models

def evaluate_models(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate models and return comprehensive metrics."""
    results = {}
    
    for name, model in models.items():
        # Handle scaled logistic regression
        if name == 'logistic' and hasattr(model, 'scaler'):
            X_test_scaled = model.scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Get classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results[name] = {
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
    
    return results

def plot_roc_curves(models: dict, X_test: pd.DataFrame, y_test: pd.Series):
    """Plot and save ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        # Handle scaled logistic regression
        if name == 'logistic' and hasattr(model, 'scaler'):
            X_test_scaled = model.scaler.transform(X_test)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Fraud Detection Models')
    plt.legend()
    plt.grid(True)
    
    os.makedirs('outputs/plots', exist_ok=True)
    plt.savefig('outputs/plots/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_metrics(results: dict):
    """Save evaluation metrics to JSON file."""
    os.makedirs('outputs', exist_ok=True)
    
    with open('outputs/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

def save_model(model, filepath: str):
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
