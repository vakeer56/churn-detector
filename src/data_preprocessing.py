import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(filepath):
    """Load the dataset and perform basic cleaning on target and features."""
    df = pd.read_csv(filepath)
    
    # Drop customerID as it's just an identifier
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    # TotalCharges is object type because it contains ' ' for new customers
    # Replace ' ' with NaN and convert to float
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    
    # Map target variable Churn
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df

def get_preprocessing_pipeline(numeric_features, categorical_features):
    """Build and return a scikit-learn preprocessing pipeline."""
    # Numerical pipeline: impute missing with median, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: impute missing with most frequent, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
    ])
    
    # Combine using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def prepare_data(df, target_col='Churn', test_size=0.2, random_state=42):
    """Split data into train and test sets, separating features and target."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Stratified split to maintain churn ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, numeric_features, categorical_features
