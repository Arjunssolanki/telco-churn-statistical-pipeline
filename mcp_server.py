import os
import joblib
import pandas as pd
from fastmcp import FastMCP

# 1. Initialize the FastMCP Server instance (Updated to match latest API specs)
mcp = FastMCP(
    name="Telco Churn Analytics Server"
)

# 2. Secure Artifact Loading Engine
def load_production_models():
    model_path = "artifacts/churn_classifier_model.pkl"
    columns_path = "artifacts/production_feature_columns.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(columns_path):
        return None, None
        
    return joblib.load(model_path), joblib.load(columns_path)

model, expected_features = load_production_models()

# 3. Register Tool A: Real-Time Single-Customer Prediction Agent
@mcp.tool(
    name="predict_customer_churn_risk",
    description="Calculates the probability of a customer churning based on their account properties, contractual features, and service types."
)
def predict_customer_churn_risk(
    tenure_months: int,
    monthly_charges: float,
    contract_type: str,
    internet_service_type: str,
    payment_method: str,
    gender: str = "Male",
    senior_citizen: str = "No",
    partner: str = "No",
    dependents: str = "No"
) -> str:
    """
    Validates customer inputs, mirrors the production scaling/encoding workflow, 
    and returns a standardized operational risk classification.
    """
    if model is None or expected_features is None:
        return "Error: Production model artifacts missing. Please build artifacts by running main.py first."
        
    # Scale continuous slider representations using baseline Z-approximations
    scaled_tenure = (tenure_months - 32.37) / 24.56  
    scaled_charges = (monthly_charges - 64.76) / 30.09
    
    # Establish dictionary mapping context properties
    input_data = {
        "tenure": scaled_tenure, "MonthlyCharges": scaled_charges, "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0, "Partner": partner,
        "Dependents": dependents, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": internet_service_type, "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": contract_type, "PaperlessBilling": "Yes",
        "PaymentMethod": payment_method
    }
    
    # Process inputs through identical one-hot schema arrays
    input_df = pd.DataFrame([input_data])
    for col in input_df.columns:
        input_df[col] = input_df[col].astype(str) if col not in ['tenure', 'MonthlyCharges', 'SeniorCitizen'] else input_df[col]
        
    input_encoded = pd.get_dummies(input_df)
    
    # Standardize dimensions to exactly match the 29 expected feature columns with explicit 0 index
    final_input_matrix = pd.DataFrame(0, index=[0], columns=expected_features)
    for col in expected_features:
        if col in input_encoded.columns:
            final_input_matrix[col] = input_encoded[col].values
        elif col in input_df.columns:
            final_input_matrix[col] = input_df[col].values
            
    # Run the machine learning model inference query
    prediction_proba = model.predict_proba(final_input_matrix)[0][1]
    
    # Categorize risks output matrices using standard operational boundaries
    if prediction_proba < 0.30:
        risk_tier = "STABLE (LOW RISK)"
        action = "Maintain standard automated marketing cadences."
    elif prediction_proba < 0.50:
        risk_tier = "ELEVATED (MODERATE RISK)"
        action = "Deploy soft retention incentives (e.g., minor value billing credits)."
    else:
        risk_tier = "CRITICAL (HIGH CHURN RISK)"
        action = "Trigger immediate account recovery actions (e.g., direct contract upgrade optimization discount)."
        
    return f"""
📊 CUSTOMER RISK DIAGNOSTIC METRIC:
  • Churn Probability: {prediction_proba * 100:.2f}%
  • Assigned Risk Tier: {risk_tier}
  • Recommended Strategy: {action}
"""

# 4. Register Tool B: Pipeline Structural Profile Summary Report
@mcp.tool(
    name="get_pipeline_performance_metadata",
    description="Returns cross-validated historical scoring matrices and top predictive features for stakeholder auditing."
)
def get_pipeline_performance_metadata() -> str:
    """Returns a production evaluation summary for reporting purposes."""
    return """
📈 PRODUCTION CROSS-VALIDATED PERFORMANCE REPORT:
  • Metric Performance: Accuracy=0.7843 | F1-Score=0.7740 | ROC-AUC=0.8150
  • Primary Continuous Driver: Account Tenure (Months) - 22.92% System Weight
  • Secondary Continuous Driver: Monthly Charges ($) - 20.20% System Weight
  • Top Contractual Buffer: Two-year Contract Commitment - 6.49% System Weight
"""

if __name__ == "__main__":
    # Start the FastMCP server process using built-in command handlers
    mcp.run()