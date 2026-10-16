import os
import yaml
import pandas as pd
import numpy as np

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Safely loads pipeline configuration rules and metadata parameters."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file missing at expected path: {config_path}")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def ingest_raw_data(data_path: str, config: dict) -> pd.DataFrame:
    """
    Loads raw CSV data, enforces strict typing, handles missing values,
    and guarantees the target variable is locked and preserved in the schema.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Target raw dataset missing at path: {data_path}")
        
    print(f"📦 Commencing ingestion pipeline for: {data_path}")
    df = pd.read_csv(data_path)
    
    target_var = config["data"]["target_variable"]
    continuous_cols = config["data"]["continuous_features"]
    
    # Ensure target column is present right at the start
    if target_var not in df.columns:
        raise KeyError(f"Critical Error: Target column '{target_var}' missing from raw input source data fields!")

    # 1. Type Coercion: Force numeric conversion (blank spaces turn into NaNs)
    for col in continuous_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 2. Apply Business Domain Logic (Tenure = 0 implies TotalCharges = 0)
    if 'tenure' in df.columns and 'TotalCharges' in df.columns:
        zero_tenure_mask = (df['tenure'] == 0) & (df['TotalCharges'].isna())
        num_business_imputed = zero_tenure_mask.sum()
        if num_business_imputed > 0:
            df.loc[zero_tenure_mask, 'TotalCharges'] = 0.0
            print(f"🔧 Imputed {num_business_imputed} rows in 'TotalCharges' with 0.0 based on zero-tenure business logic.")

    # 3. Standardize Target Variable to Integer early
    # This prevents Arrow extensions from shifting characters down the road
    df[target_var] = df[target_var].astype(str).str.strip()
    target_mapping = {'Yes': 1, '1': 1, '1.0': 1, 'No': 0, '0': 0, '0.0': 0}
    df[target_var] = df[target_var].map(target_mapping).astype(int)
    print(f"🔄 Standardised target variable '{target_var}' to binary integers (0/1).")
        
    print(f"✅ Ingestion successfully completed. Total rows preserved: {len(df)}")
    return df

if __name__ == "__main__":
    try:
        pipeline_config = load_config()
        sample_df = ingest_raw_data("telco_churn_raw.csv", pipeline_config)
        print(sample_df[[pipeline_config["data"]["target_variable"], 'tenure', 'MonthlyCharges']].head(2))
    except Exception as e:
        print(f"🛑 Ingestion test failed: {str(e)}")
