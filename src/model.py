import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           precision_recall_curve, roc_curve, average_precision_score,
                           precision_score, recall_score, f1_score, make_scorer)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import xgboost as xgb
import optuna
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any, Tuple

def prepare_features_target(df: pd.DataFrame) -> tuple:
    """
    Separate features and target variable.
    
    Args:
        df: Processed dataset
        
    Returns:
        tuple: (X, y) features and target
    """
    # Drop original columns we don't need for modeling
    feature_cols = [col for col in df.columns if col not in ['Class', 'Time']]
    
    X = df[feature_cols]
    y = df['Class']
    
    return X, y

def create_stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Create stratified train/test split preserving fraud ratio.
    
    Args:
        X: Features
        y: Target
        test_size: Test set proportion
        random_state: Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Preserve fraud ratio
    )
    
    return X_train, X_test, y_train, y_test

def apply_smote_sampling(X_train: pd.DataFrame, y_train: pd.Series, strategy: str = 'moderate') -> tuple:
    """
    Apply SMOTE oversampling with different strategies.
    
    Args:
        X_train: Training features
        y_train: Training target
        strategy: 'conservative', 'moderate', or 'aggressive'
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    if strategy == 'conservative':
        # Only oversample to 10% fraud ratio
        sampling_strategy = 0.1
    elif strategy == 'moderate':
        # Oversample to 20% fraud ratio
        sampling_strategy = 0.2
    elif strategy == 'aggressive':
        # Oversample to 50% fraud ratio
        sampling_strategy = 0.5
    else:
        raise ValueError("Strategy must be 'conservative', 'moderate', or 'aggressive'")
    
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled

def apply_combined_sampling(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """
    Apply SMOTEENN (SMOTE + Edited Nearest Neighbors) for cleaner boundaries.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    smote_enn = SMOTEENN(sampling_strategy=0.3, random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled

def apply_undersampling(X_train: pd.DataFrame, y_train: pd.Series, ratio: float = 0.1) -> tuple:
    """
    Apply random undersampling to majority class.
    
    Args:
        X_train: Training features
        y_train: Training target
        ratio: Fraud to normal ratio after undersampling
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    undersampler = RandomUnderSampler(sampling_strategy=ratio, random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled

def get_class_weights(y: pd.Series) -> dict:
    """
    Calculate balanced class weights for sklearn models.
    
    Args:
        y: Target variable
        
    Returns:
        dict: Class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    
    return dict(zip(classes, class_weights))

def prepare_datasets(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Prepare multiple versions of training data with different sampling strategies.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        dict: Different versions of training data
    """
    datasets = {}
    
    # Original imbalanced data
    datasets['original'] = (X_train, y_train)
    
    # SMOTE variants
    datasets['smote_conservative'] = apply_smote_sampling(X_train, y_train, 'conservative')
    datasets['smote_moderate'] = apply_smote_sampling(X_train, y_train, 'moderate')
    
    # Combined sampling
    datasets['smote_enn'] = apply_combined_sampling(X_train, y_train)
    
    # Undersampling
    datasets['undersampled'] = apply_undersampling(X_train, y_train, ratio=0.1)
    
    # Class weights (for original data)
    class_weights = get_class_weights(y_train)
    datasets['class_weights'] = class_weights
    
    return datasets

def get_baseline_models() -> dict:
    """
    Initialize baseline models with fraud-optimized configurations.
    
    Returns:
        dict: Initialized models
    """
    models = {
        'logistic_regression': LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'xgboost': xgb.XGBClassifier(
            random_state=42,
            eval_metric='aucpr',  # Better for imbalanced data
            scale_pos_weight=1,  # Will be adjusted based on class ratio
            n_jobs=-1
        )
    }
    
    return models

def calculate_pos_weight(y: pd.Series) -> float:
    """
    Calculate positive class weight for XGBoost.
    
    Args:
        y: Target variable
        
    Returns:
        float: Positive class weight
    """
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    
    return neg_count / pos_count

def evaluate_sampling_strategy(X_train: pd.DataFrame, y_train: pd.Series, 
                             X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Quick evaluation of different sampling strategies using Logistic Regression.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        dict: Performance metrics for each strategy
    """
    datasets = prepare_datasets(X_train, y_train)
    results = {}
    
    for strategy_name, (X_sample, y_sample) in datasets.items():
        if strategy_name == 'class_weights':
            continue
            
        # Quick logistic regression evaluation
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_sample, y_sample)
        
        y_pred_proba = lr.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results[strategy_name] = {
            'auc_score': auc_score,
            'training_fraud_ratio': y_sample.mean(),
            'training_size': len(y_sample)
        }
    
    return results

def print_sampling_analysis(results: dict) -> None:
    """
    Print analysis of different sampling strategies.
    
    Args:
        results: Results from evaluate_sampling_strategy
    """
    print("=== SAMPLING STRATEGY ANALYSIS ===")
    print(f"{'Strategy':<20} {'AUC':<8} {'Fraud%':<10} {'Size':<10}")
    print("-" * 50)
    
    for strategy, metrics in results.items():
        print(f"{strategy:<20} {metrics['auc_score']:.4f}   "
              f"{metrics['training_fraud_ratio']:.2%}    "
              f"{metrics['training_size']:,}")
    
    best_strategy = max(results.items(), key=lambda x: x[1]['auc_score'])
    print(f"\nBest strategy: {best_strategy[0]} (AUC: {best_strategy[1]['auc_score']:.4f})")

def tune_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    """
    Tune Logistic Regression using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        dict: Best model and parameters
    """
    # Parameter grid for logistic regression
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'class_weight': ['balanced', {0: 1, 1: 10}, {0: 1, 1: 50}]
    }
    
    # Use PR-AUC as scoring metric (better for imbalanced data)
    scorer = make_scorer(roc_auc_score, greater_is_better=True)
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    
    # 3-fold CV to save time while maintaining reliability
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        lr, param_grid, 
        cv=cv, 
        scoring=scorer, 
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    return {
        'model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_
    }

def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    """
    Tune Random Forest using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        dict: Best model and parameters
    """
    # Focused parameter grid for fraud detection
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    scorer = make_scorer(roc_auc_score, greater_is_better=True)
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        rf, param_grid, 
        cv=cv, 
        scoring=scorer, 
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    return {
        'model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_
    }

def tune_xgboost_optuna(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 50) -> Dict[str, Any]:
    """
    Tune XGBoost using Optuna for more efficient hyperparameter search.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_trials: Number of Optuna trials
        
    Returns:
        dict: Best model and parameters
    """
    def objective(trial):
        # Sample hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 100),
            'random_state': 42,
            'eval_metric': 'aucpr'
        }
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr)
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred_proba)
            scores.append(score)
        
        return np.mean(scores)
    
    # Run Optuna optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Train best model on full dataset
    best_model = xgb.XGBClassifier(**study.best_params)
    best_model.fit(X_train, y_train)
    
    return {
        'model': best_model,
        'best_params': study.best_params,
        'best_score': study.best_value
    }

def train_all_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Dict[str, Any]]:
    """
    Train and tune all three models.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        dict: All trained models with their results
    """
    models = {}
    
    print("=== TRAINING LOGISTIC REGRESSION ===")
    models['logistic_regression'] = tune_logistic_regression(X_train, y_train)
    
    print("\n=== TRAINING RANDOM FOREST ===")
    models['random_forest'] = tune_random_forest(X_train, y_train)
    
    print("\n=== TRAINING XGBOOST ===")
    models['xgboost'] = tune_xgboost_optuna(X_train, y_train)
    
    return models

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a trained model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model for reporting
        
    Returns:
        dict: Comprehensive evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Core metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Business metrics for fraud detection
    specificity = tn / (tn + fp)  # True negative rate
    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (fn + tp)
    
    # Cost-based metrics (assuming fraud costs 100x more than false alarm)
    fraud_cost = 100
    false_alarm_cost = 1
    total_cost = (fn * fraud_cost) + (fp * false_alarm_cost)
    
    evaluation = {
        'model_name': model_name,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'specificity': specificity,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp), 
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'business_metrics': {
            'total_cost': total_cost,
            'fraud_cost_per_miss': fraud_cost,
            'false_alarm_cost': false_alarm_cost,
            'fraud_caught': int(tp),
            'fraud_missed': int(fn),
            'fraud_catch_rate': recall
        },
        'predictions': {
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist()
        }
    }
    
    return evaluation

def plot_confusion_matrix(cm: np.ndarray, model_name: str, save_path: str = 'outputs/plots/') -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix
        model_name: Name of the model
        save_path: Directory to save plot
    """
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fraud'], 
                yticklabels=['Normal', 'Fraud'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'{save_path}{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_test: pd.Series, y_pred_proba: np.ndarray, model_name: str, 
                   save_path: str = 'outputs/plots/') -> None:
    """
    Plot and save ROC curve.
    
    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model
        save_path: Directory to save plot
    """
    os.makedirs(save_path, exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}{model_name}_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curve(y_test: pd.Series, y_pred_proba: np.ndarray, model_name: str,
                               save_path: str = 'outputs/plots/') -> None:
    """
    Plot and save Precision-Recall curve (more informative for imbalanced data).
    
    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model
        save_path: Directory to save plot
    """
    os.makedirs(save_path, exist_ok=True)
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.axhline(y=y_test.mean(), color='navy', linestyle='--', label=f'Baseline ({y_test.mean():.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}{model_name}_pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_model_comparison_plot(all_evaluations: Dict[str, Dict], save_path: str = 'outputs/plots/') -> None:
    """
    Create comparison plot of all models.
    
    Args:
        all_evaluations: Dictionary of all model evaluations
        save_path: Directory to save plot
    """
    os.makedirs(save_path, exist_ok=True)
    
    models = list(all_evaluations.keys())
    metrics = ['precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
    
    # Create comparison dataframe
    comparison_data = []
    for model in models:
        for metric in metrics:
            comparison_data.append({
                'Model': model.replace('_', ' ').title(),
                'Metric': metric.replace('_', ' ').upper(),
                'Score': all_evaluations[model][metric]
            })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Create grouped bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_comparison, x='Metric', y='Score', hue='Model')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{save_path}model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_evaluation_summary(evaluation: Dict[str, Any]) -> None:
    """
    Print formatted evaluation summary.
    
    Args:
        evaluation: Model evaluation dictionary
    """
    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY - {evaluation['model_name'].upper()}")
    print(f"{'='*50}")
    
    print(f"🎯 CORE METRICS:")
    print(f"   Precision: {evaluation['precision']:.4f}")
    print(f"   Recall:    {evaluation['recall']:.4f}")
    print(f"   F1-Score:  {evaluation['f1_score']:.4f}")
    print(f"   ROC-AUC:   {evaluation['roc_auc']:.4f}")
    print(f"   PR-AUC:    {evaluation['pr_auc']:.4f}")
    
    print(f"\n💼 BUSINESS IMPACT:")
    print(f"   Fraud Caught: {evaluation['business_metrics']['fraud_caught']}")
    print(f"   Fraud Missed: {evaluation['business_metrics']['fraud_missed']}")
    print(f"   Catch Rate:   {evaluation['business_metrics']['fraud_catch_rate']:.2%}")
    print(f"   Total Cost:   ${evaluation['business_metrics']['total_cost']:,.0f}")
    
    cm = evaluation['confusion_matrix']
    print(f"\n📊 CONFUSION MATRIX:")
    print(f"   True Negatives:  {cm['true_negatives']:,}")
    print(f"   False Positives: {cm['false_positives']:,}")
    print(f"   False Negatives: {cm['false_negatives']:,}")
    print(f"   True Positives:  {cm['true_positives']:,}")

def save_metrics_to_json(all_evaluations: Dict[str, Dict], filepath: str = 'outputs/metrics.json') -> None:
    """
    Save all evaluation metrics to JSON file.
    
    Args:
        all_evaluations: Dictionary of all model evaluations
        filepath: Path to save JSON file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Remove predictions from saved metrics (too large)
    clean_evaluations = {}
    for model_name, evaluation in all_evaluations.items():
        clean_eval = evaluation.copy()
        clean_eval.pop('predictions', None)  # Remove predictions to save space
        clean_evaluations[model_name] = clean_eval
    
    with open(filepath, 'w') as f:
        json.dump(clean_evaluations, f, indent=2)
    
    print(f"Metrics saved to {filepath}")

def evaluate_all_models(models: Dict[str, Dict], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
    """
    Evaluate all trained models and create comprehensive report.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test target
        
    Returns:
        dict: All model evaluations
    """
    all_evaluations = {}
    
    for model_name, model_data in models.items():
        print(f"\n{'='*60}")
        print(f"EVALUATING {model_name.upper()}")
        print(f"{'='*60}")
        
        # Evaluate model
        evaluation = evaluate_model(model_data['model'], X_test, y_test, model_name)
        all_evaluations[model_name] = evaluation
        
        # Print summary
        print_evaluation_summary(evaluation)
        
        # Create visualizations
        cm = confusion_matrix(y_test, model_data['model'].predict(X_test))
        y_pred_proba = model_data['model'].predict_proba(X_test)[:, 1]
        
        plot_confusion_matrix(cm, model_name)
        plot_roc_curve(y_test, y_pred_proba, model_name)
        plot_precision_recall_curve(y_test, y_pred_proba, model_name)
    
    # Create comparison plot
    create_model_comparison_plot(all_evaluations)
    
    # Save metrics
    save_metrics_to_json(all_evaluations)
    
    # Find best model
    best_model = max(all_evaluations.items(), key=lambda x: x[1]['pr_auc'])
    print(f"\n🏆 BEST MODEL: {best_model[0]} (PR-AUC: {best_model[1]['pr_auc']:.4f})")
    
    return all_evaluations

def save_models(models: Dict[str, Dict[str, Any]], filepath: str = 'outputs/') -> None:
    """
    Save trained models to disk.
    
    Args:
        models: Dictionary of trained models
        filepath: Directory to save models
    """
    os.makedirs(filepath, exist_ok=True)
    
    for model_name, model_data in models.items():
        # Save model
        model_path = f"{filepath}{model_name}_model.pkl"
        joblib.dump(model_data['model'], model_path)
        
        # Save parameters
        params_path = f"{filepath}{model_name}_params.json"
        with open(params_path, 'w') as f:
            json.dump({
                'best_params': model_data['best_params'],
                'best_score': model_data['best_score']
            }, f, indent=2)
    
    print(f"Models saved to {filepath}")

def get_train_test_data(df: pd.DataFrame) -> tuple:
    """
    Main function to prepare train/test data with optimal sampling.
    
    Args:
        df: Processed dataset from data_prep.py
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, best_sampling_strategy)
    """
    # Prepare features and target
    X, y = prepare_features_target(df)
    
    # Create stratified split
    X_train, X_test, y_train, y_test = create_stratified_split(X, y)
    
    # Evaluate sampling strategies
    sampling_results = evaluate_sampling_strategy(X_train, y_train, X_test, y_test)
    print_sampling_analysis(sampling_results)
    
    # Get best sampling strategy
    best_strategy = max(sampling_results.items(), key=lambda x: x[1]['auc_score'])[0]
    
    return X_train, X_test, y_train, y_test, best_strategy