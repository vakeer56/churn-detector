import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from src.data_preprocessing import load_data, prepare_data

def evaluate_model():
    model_path = Path('models/churn_model.pkl')
    data_path = Path('data/Telco-Customer-Churn.csv')
    
    if not model_path.exists() or not data_path.exists():
        raise FileNotFoundError("Model or dataset not found. Ensure training has completed.")
        
    print("Loading data and model...")
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, numeric_features, categorical_features = prepare_data(df)
    
    pipeline = joblib.load(model_path)
    model = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    
    print("Making predictions on the test set...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance (if applicable)
    if hasattr(model, 'feature_importances_'):
        print("\n--- Feature Importance ---")
        
        # Extract feature names after one-hot encoding
        numeric_names = numeric_features
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        categorical_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
        feature_names = numeric_names + categorical_names
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("Top 10 most important features:")
        top_features = []
        top_importances = []
        for i in range(min(10, len(feature_names))):
            name = feature_names[indices[i]]
            val = importances[indices[i]]
            top_features.append(name)
            top_importances.append(val)
            print(f"{i+1}. {name} ({val:.4f})")
            
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_importances, y=top_features, hue=top_features, legend=False, palette='viridis')
        plt.title('Top 10 Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        figures_dir = Path('results/figures')
        figures_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(figures_dir / 'feature_importance.png')
        print(f"\nFeature importance plot saved to {figures_dir / 'feature_importance.png'}")
        
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(figures_dir / 'confusion_matrix.png')
    print(f"Confusion matrix plot saved to {figures_dir / 'confusion_matrix.png'}")

if __name__ == "__main__":
    evaluate_model()
