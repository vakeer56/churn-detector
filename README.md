# Customer Churn Prediction

## Problem Statement
Businesses in the telecommunications sector often suffer from high customer turnover (churn). Identifying customers at risk of leaving allows companies to implement proactive retention strategies.

## Objective
The objective of this project is to develop an end-to-end machine learning pipeline that predicts customer churn based on historical data. The project involves exploratory data analysis (EDA), data cleaning, feature engineering, model training, evaluation, and prediction.

## Dataset
We use the **IBM Telco Customer Churn** dataset.
The dataset contains information about customers, including their demographics, account information (tenure, contract type, payment method), and subscribed services. The target variable is `Churn`, which indicates whether the customer left within the last month.

## Features
- **Categorical:** Gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod.
- **Numerical:** Tenure, MonthlyCharges, TotalCharges.
- **Target:** Churn (Yes/No).

## Project Structure
```
customer-churn-prediction/
│
├── data/
│   ├── README.md
│   └── Telco-Customer-Churn.csv (after downloading)
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── churn_model.pkl (generated after training)
│
├── results/
│   └── figures/
│       ├── feature_importance.png
│       └── confusion_matrix.png
│
├── requirements.txt
├── README.md
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## How to Run

### 1. Download Dataset
You can manually download the dataset to `data/Telco-Customer-Churn.csv`, or run:
```bash
wget -O data/Telco-Customer-Churn.csv https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

### 2. Exploratory Data Analysis
Open the Jupyter Notebook to view the analysis:
```bash
jupyter notebook notebooks/churn_analysis.ipynb
```

### 3. Data Preprocessing & Model Training
To train the models (Logistic Regression, Decision Tree, Random Forest, XGBoost) and perform hyperparameter tuning:
```bash
python src/train.py
```
This saves the best model pipeline to `models/churn_model.pkl`.

### 4. Evaluate Models
Evaluate the saved model on test data to see Precision, Recall, F1, ROC-AUC, and feature importances:
```bash
python src/evaluate.py
```
Visualizations will be saved in `results/figures/`.

### 5. Launch Web Application Frontend
Run the interactive Streamlit web dashboard:
```bash
streamlit run app.py
```
This opens a modern web dashboard in your browser where you can input customer parameters, receive real-time churn risk predictions (with color-coded badges and risk percentages), view strategic retention recommendations, and analyze dataset insights & model performance.

### 6. Make a Prediction via CLI Script
Test the CLI prediction script with a sample customer:
```bash
PYTHONPATH=. python src/predict.py
```

## Models Used
1. **Logistic Regression:** Serves as a baseline model due to its interpretability.
2. **Decision Tree:** Can capture non-linear relationships but is prone to overfitting.
3. **Random Forest:** An ensemble method that reduces overfitting and improves generalization.
4. **XGBoost:** A gradient boosting algorithm known for high performance on tabular data.

## Evaluation Metrics
Because churn prediction datasets are often imbalanced (more people stay than leave), Accuracy alone is misleading. We focus on:
- **Recall:** How many actual churners were identified? Missing a churner is expensive.
- **Precision:** Of those predicted to churn, how many actually did? High precision prevents wasting retention budgets.
- **F1-Score:** The harmonic mean of precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between classes at various thresholds.

## Results
(Execute `src/evaluate.py` to view actual results).
Typically, Random Forest or XGBoost performs best on this dataset.

## Feature Importance
We extract feature importances from the best tree-based model (e.g., Random Forest). Usually, features like `Contract type`, `Tenure`, and `MonthlyCharges` heavily influence whether a customer will churn.

## Example Prediction
Run `python src/predict.py` to see the model infer whether a single hypothetical customer profile is likely to churn, alongside its churn probability.

## Limitations
- We don't have detailed usage behavior (e.g. daily call duration).
- Predictions assume past trends continue exactly into the future.
- Hyperparameter tuning is restricted to a small grid to ensure it runs quickly on personal laptops.

## Future Improvements
- Expand hyperparameter search grid.
- Apply SMOTE or class weighting for class imbalance.
- Implement an interactive web dashboard (e.g., Streamlit) for non-technical stakeholders to input customer data and see churn risk.
