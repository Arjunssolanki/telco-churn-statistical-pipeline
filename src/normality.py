import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PowerTransformer, StandardScaler

def test_and_transform_features(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """
    Evaluates continuous features for normality using Shapiro-Wilk. 
    Applies Power Transformation or Standard Scaling while explicitly 
    preserving all target and categorical fields in the returned DataFrame.
    """
    continuous_cols = config["data"]["continuous_features"]
    alpha = config["statistical_thresholds"]["alpha_normality"]
    max_sample = config["pipeline_routing_rules"]["shapiro_max_sample_size"]
    transform_method = config["pipeline_routing_rules"]["non_normal_transform"]
    
    # Create a full, deep copy to preserve all columns (target and categorical)
    transformed_df = df.copy()
    normality_registry = {}
    
    for col in continuous_cols:
        if col not in df.columns:
            continue
            
        series_data = df[col].dropna().values
        sample_data = series_data[:max_sample] if len(series_data) > max_sample else series_data
        
        # Run Shapiro-Wilk test
        shapiro_stat, p_value = stats.shapiro(sample_data)
        is_normal = p_value >= alpha
        normality_registry[col] = is_normal
        
        feature_vector = series_data.reshape(-1, 1)
        
        # Route to appropriate transformation layer
        if not is_normal:
            transformer = PowerTransformer(method=transform_method, standardize=True)
            transformed_vector = transformer.fit_transform(feature_vector)
            transformed_df[col] = transformed_vector.flatten()
        else:
            scaler = StandardScaler()
            scaled_vector = scaler.fit_transform(feature_vector)
            transformed_df[col] = scaled_vector.flatten()
            
    return transformed_df, normality_registry

if __name__ == "__main__":
    from src.ingestion import load_config, ingest_raw_data
    try:
        cfg = load_config()
        raw_df = ingest_raw_data("telco_churn_raw.csv", cfg)
        processed_df, registry = test_and_transform_features(raw_df, cfg)
        print("\nDistribution Normality Status Result Registry:")
        print(registry)
        print(f"\nVerification - Columns preserved in output: {list(processed_df.columns)[:5]}...")
    except Exception as e:
        print(f"Execution Error: {str(e)}")
