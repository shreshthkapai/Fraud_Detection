import pandas as pd
import numpy as np
from src.data_prep import load_data, clean_data, engineer_features, handle_outliers, normalize_features
from src.model import stratified_split, train_tuned_models, evaluate_models, plot_roc_curves, save_metrics, save_model
from src.explain import explain_model

def main():
    print("🔍 Starting Fraud Detection Pipeline...")
    
    # Step 1: Load and prepare data
    print("\n📊 Loading data...")
    df = load_data("data/raw.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
    
    # Step 2: Clean and engineer features
    print("\n🔧 Cleaning and engineering features...")
    df = clean_data(df)
    df = engineer_features(df)
    df = handle_outliers(df)
    
    # Step 3: Prepare features and target
    print("\n🎯 Preparing features and target...")
    # Select features (excluding engineered time features for simplicity)
    feature_cols = [col for col in df.columns if col not in ['Class', 'Day', 'Hour', 'Trans_per_hour', 'Amount_log']]
    X = df[feature_cols]
    y = df['Class']
    
    # Normalize Amount feature
    X, scaler = normalize_features(X, ['Amount'])
    
    # Step 4: Split data
    print("\n✂️ Splitting data...")
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 5: Train models
    print("\n🤖 Training models...")
    models = train_tuned_models(X_train, y_train)
    
    # Step 6: Evaluate models
    print("\n📈 Evaluating models...")
    results = evaluate_models(models, X_test, y_test)
    
    # Step 7: Generate visualizations
    print("\n📊 Generating visualizations...")
    plot_roc_curves(models, X_test, y_test)
    
    # Step 8: Save everything
    print("\n💾 Saving results...")
    save_metrics(results)
    
    # Save individual models
    for name, model in models.items():
        model_name = name.replace(' ', '_').lower()
        save_model(model, f"outputs/{model_name}_model.pkl")
    
    # Step 9: Generate SHAP explanations (for best model)
    print("\n🔍 Generating SHAP explanations...")
    best_model = models['xgboost']  # Assuming XGBoost is best
    explain_model(best_model, X_train, X_test)
    
    # Step 10: Print results summary
    print("\n📋 Results Summary:")
    print("=" * 50)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  F1-Score: {metrics['f1_score']:.3f}")
        print(f"  ROC-AUC:  {metrics['roc_auc']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:   {metrics['recall']:.3f}")
    
    print("\n✅ Pipeline completed successfully!")
    print("📁 Check the 'outputs/' directory for results")

if __name__ == "__main__":
    main()
