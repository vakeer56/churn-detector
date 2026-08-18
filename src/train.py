import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from src.data_preprocessing import load_data, get_preprocessing_pipeline, prepare_data

def train_and_tune():
    data_path = Path('data/Telco-Customer-Churn.csv')
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please download it first.")
        
    print("Loading and preparing data...")
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, numeric_features, categorical_features = prepare_data(df)
    
    print("Building preprocessing pipeline...")
    preprocessor = get_preprocessing_pipeline(numeric_features, categorical_features)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    if XGB_AVAILABLE:
        models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        
    trained_pipelines = {}
    
    # Base training
    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        score = pipeline.score(X_test, y_test)
        print(f"{name} Base Accuracy: {score:.4f}")
        
    print("\nStarting Hyperparameter Tuning for Random Forest...")
    # Hyperparameter tuning for Random Forest
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
    
    grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    print(f"Best Random Forest params: {grid_search.best_params_}")
    print(f"Tuned Random Forest Test Accuracy: {best_rf.score(X_test, y_test):.4f}")
    
    # Save the best model (using tuned Random Forest as our final model)
    model_save_path = Path('models/churn_model.pkl')
    model_save_path.parent.mkdir(exist_ok=True)
    joblib.dump(best_rf, model_save_path)
    print(f"\nBest model saved to {model_save_path}")

if __name__ == "__main__":
    train_and_tune()
