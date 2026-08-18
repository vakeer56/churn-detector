import pandas as pd
import joblib
from pathlib import Path

def predict_churn(customer_data):
    """
    Predict churn for a single customer.
    customer_data: dict containing feature names and values.
    """
    model_path = Path('models/churn_model.pkl')
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Please train the model first.")
        
    pipeline = joblib.load(model_path)
    
    # Convert dict to DataFrame (1 row)
    try:
        df = pd.DataFrame([customer_data])
    except Exception as e:
        raise ValueError(f"Invalid input data format: {e}")
    
    # Preprocess and predict
    # (The pipeline will handle scaling and encoding automatically, 
    # but we must ensure we have the required columns)
    try:
        prediction = pipeline.predict(df)[0]
        probability = pipeline.predict_proba(df)[0, 1]
    except Exception as e:
        raise ValueError(f"Error making prediction, possibly missing columns: {e}")
        
    result = "Customer likely to churn" if prediction == 1 else "Customer likely to stay"
    
    print("\n--- Prediction Results ---")
    print(f"Outcome: {result}")
    print(f"Churn probability: {probability * 100:.2f}%\n")
    
    return prediction, probability

if __name__ == "__main__":
    # Example realistic sample input
    sample_customer = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 24,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 85.50,
        'TotalCharges': 2052.00
    }
    
    print("Testing with a sample customer profile:")
    for key, value in sample_customer.items():
        print(f"  {key}: {value}")
        
    predict_churn(sample_customer)
