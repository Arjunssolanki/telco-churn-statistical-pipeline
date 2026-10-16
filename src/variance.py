import numpy as np
import pandas as pd
from scipy import stats

def evaluate_feature_variance(df: pd.DataFrame, normality_registry: dict, config: dict) -> dict:
    continuous_cols = config["data"]["continuous_features"]
    target_var = config["data"]["target_variable"]
    alpha = config["statistical_thresholds"]["alpha_variance"]
    center_metric = config["pipeline_routing_rules"]["variance_center_metric"]
    
    variance_registry = {}
    
    # Inspect the values present in the target variable column to adaptively extract groups
    unique_targets = df[target_var].unique()
    print(f"🔍 Debug: Target column unique values currently found in data: {unique_targets}")
    
    for col in continuous_cols:
        # Dynamically determine group slices by checking what format the target values use
        if 0 in unique_targets or 1 in unique_targets:
            group_0 = df[df[target_var] == 0][col].dropna().values
            group_1 = df[df[target_var] == 1][col].dropna().values
        elif '0' in unique_targets or '1' in unique_targets:
            group_0 = df[df[target_var] == '0'][col].dropna().values
            group_1 = df[df[target_var] == '1'][col].dropna().values
        else:
            # Fallback directly to the original strings if mapping hasn't been applied yet
            group_0 = df[df[target_var].isin(['No', 'no', 'NO'])][col].dropna().values
            group_1 = df[df[target_var].isin(['Yes', 'yes', 'YES'])][col].dropna().values
            
        if len(group_0) == 0 or len(group_1) == 0:
            print(f"🛑 Error: Empty slice extracted for feature '{col}'. Group 0 size: {len(group_0)}, Group 1 size: {len(group_1)}")
            variance_registry[col] = {"equal_variance": False, "p_value": np.nan, "test_executed": "None"}
            continue
            
        is_normal = normality_registry.get(col, False)
        
        if is_normal:
            stat, p_value = stats.bartlett(group_0, group_1)
            test_used = "Bartlett"
        else:
            stat, p_value = stats.levene(group_0, group_1, center=center_metric)
            test_used = "Levene"
            
        has_equal_variance = p_value >= alpha
        variance_registry[col] = {
            "equal_variance": has_equal_variance,
            "p_value": float(p_value),
            "test_executed": test_used
        }
        
        print(f"📊 [{col}] Variance Check via {test_used} Test | p-value: {p_value:.5f} | Equal Variance: {has_equal_variance}")
        
    return variance_registry

if __name__ == "__main__":
    from src.ingestion import load_config, ingest_raw_data
    from src.normality import test_and_transform_features
    try:
        cfg = load_config()
        raw_df = ingest_raw_data("telco_churn_raw.csv", cfg)
        processed_df, norm_reg = test_and_transform_features(raw_df, cfg)
        
        print("\n🚀 Commencing Variance Evaluation...")
        var_reg = evaluate_feature_variance(processed_df, norm_reg, cfg)
    except Exception as e:
        print(f"Execution Error: {str(e)}")
