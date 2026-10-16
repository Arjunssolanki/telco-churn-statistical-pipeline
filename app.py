import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# 1. Initialize High-Density Page Configurations
st.set_page_config(
    page_title="Telco Churn Analytics Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .block-container {padding-top: 1.0rem; padding-bottom: 0rem; max-width: 96%;}
    <h1> {margin-bottom: 0.2rem; font-size: 2.0rem !important;}
    <h2> {margin-top: 0.2rem; font-size: 1.3rem !important; margin-bottom: 0.2rem;}
    <h3> {font-size: 1.0rem !important; margin-bottom: 0.2rem;}
    .stSlider {margin-bottom: -15px;}
    .stSelectbox {margin-bottom: -15px;}
    .stRadio {margin-bottom: -15px;}
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Telco Customer Churn Predictive Hub")

# 2. Secure Artifacts Loading Layer
@st.cache_resource
def load_production_artifacts():
    model_path = "artifacts/churn_classifier_model.pkl"
    columns_path = "artifacts/production_feature_columns.pkl"
    if not os.path.exists(model_path) or not os.path.exists(columns_path):
        return None, None
    return joblib.load(model_path), joblib.load(columns_path)

model, expected_features = load_production_artifacts()

if model is None or expected_features is None:
    st.error("🛑 Missing Deployment Artifacts! Please run 'python main.py' to generate your model assets first.")
    st.stop()

# 3. Primary Workspace Split
col_inputs, col_results = st.columns([1.2, 0.8], gap="medium")

with col_inputs:
    st.subheader("👤 Customer Profile Configuration")
    
    tenure = st.slider("Account Tenure (Months)", min_value=0, max_value=72, value=15, key="slider_tenure")
    monthly_charges = st.slider("Monthly Charges ($)", min_value=18, max_value=120, value=75, key="slider_charges")
    
    st.markdown("---")
    
    sub_col1, sub_col2 = st.columns(2, gap="small")
    
    with sub_col1:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"])
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True, index=1)
        partner = st.radio("Has Partner?", ["No", "Yes"], horizontal=True, index=1)

    with sub_col2:
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=3)
        senior_citizen = st.selectbox("Senior Citizen Status", ["No", "Yes"], index=0)
        dependents = st.radio("Has Dependents?", ["No", "Yes"], horizontal=True)

    phone_service, multiple_lines, online_security = "Yes", "No", "No"
    online_backup, device_protection, tech_support = "No", "No", "No"
    streaming_tv, streaming_movies, paperless_billing = "No", "No", "Yes"

with col_results:
    st.subheader("🔮 Retention Risk Analysis")
    
    # 4. Statistical Normalization Mappings
    scaled_tenure = (tenure - 32.37) / 24.56  
    scaled_charges = (monthly_charges - 64.76) / 30.09
    
    input_data = {
        "tenure": scaled_tenure, "MonthlyCharges": scaled_charges, "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0, "Partner": partner,
        "Dependents": dependents, "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "InternetService": internet_service, "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies, "Contract": contract, "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method
    }
    
    input_df = pd.DataFrame([input_data])
    categorical_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", 
                        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
                        "TechSupport", "StreamingTV", "StreamingMovies", "Contract", 
                        "PaperlessBilling", "PaymentMethod"]
    
    for col in categorical_cols:
        input_df[col] = input_df[col].astype(str)
        
    input_encoded = pd.get_dummies(input_df)
    
    final_input_matrix = pd.DataFrame(0, index=[0], columns=expected_features)
    for col in expected_features:
        if col in input_encoded.columns:
            final_input_matrix[col] = input_encoded[col].values
        elif col in input_df.columns:
            final_input_matrix[col] = input_df[col].values
            
    # 5. Extract Multi-Class Inference Probabilities
    prediction_proba = float(model.predict_proba(final_input_matrix)[0][1])
    
    st.markdown("### Risk Diagnostic")
    
    # Updated Core Routing Logic Gates for Three-Tier Support
    if prediction_proba >= 0.50:
        st.error("### ⚠️ HIGH CHURN RISK")
        playbook_status = "high"
    elif 0.30 <= prediction_proba < 0.50:
        st.warning("### ⚡ MODERATE CHURN RISK")
        playbook_status = "moderate"
    else:
        st.success("### ✅ STABLE CUSTOMER")
        playbook_status = "low"
        
    st.metric(label="Calculated Retention Churn Probability", value=f"{prediction_proba * 100:.2f}%")
    st.progress(prediction_proba)
    
    st.markdown("---")
    st.subheader("💡 Strategic Playbook")
    
    if playbook_status == "high":
        st.error("**Urgent Intervention Required:** Account displays critical cancellation markers.")
        if contract == "Month-to-month":
            st.markdown("*   **Retention Offer:** Push automated upgrade to a **One/Two-year contract** with a 15% discount incentive.")
    elif playbook_status == "moderate":
        st.warning("**Proactive Engagement Recommended:** Account shows early signs of instability.")
        st.markdown("*   **Value Optimization Campaign:** Send targeted feature guides highlighting unused platform tools to build product value.")
        if contract == "Month-to-month":
            st.markdown("*   **Contract Review:** Prompt account managers to offer soft incentives for moving off month-to-month billing terms.")
    else:
        st.info("Account metrics remain stable. Maintain standard automated retention campaigns.")
