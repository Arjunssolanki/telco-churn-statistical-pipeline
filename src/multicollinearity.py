import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

def prune_features_by_vif(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list]:
    """
    Iteratively identifies and removes highly multicollinear continuous features 
    using VIF thresholds, while preserving the target and categorical columns intact.
    """
    features_to_check = list(config["data"]["continuous_features"])
    vif_threshold = config["statistical_thresholds"]["vif_max"]
    
    print("\n🔍 Starting Multicollinearity Assessment via VIF Loop...")
    
    dropped_features = []
    
    while True:
        # Isolate the current continuous feature matrix subset
        X_subset = df[features_to_check].dropna()
        
        # Add an intercept constant column required for statsmodels VIF logic
        X_matrix = X_subset.assign(intercept=1.0)
        
        vif_data = []
        for i in range(len(features_to_check)):
            vif_val = variance_inflation_factor(X_matrix.values, i)
            vif_data.append({"Feature": features_to_check[i], "VIF": vif_val})
            
        vif_df = pd.DataFrame(vif_data).sort_values(by="VIF", ascending=False)
        
        if vif_df.empty:
            break
            
        highest_vif = vif_df.iloc[0]["VIF"]
        highest_feature = vif_df.iloc[0]["Feature"]
        
        # Check if the feature exceeds our structural threshold
        if highest_vif > vif_threshold:
            print(f"⚠️ High Multicollinearity Flagged: '{highest_feature}' has a VIF of {highest_vif:.4f}")
            print(f"❌ Dropping '{highest_feature}' from the machine learning pipeline.")
            features_to_check.remove(highest_feature)
            dropped_features.append(highest_feature)
        else:
            print("✅ All remaining continuous features are stable (VIF < 5.0).")
            print(vif_df.to_string(index=False))
            break
            
    # Drop only the highly collinear features from the main dataframe
    # This keeps 'Churn' and all categorical columns completely safe
    pruned_df = df.drop(columns=dropped_features)
            
    return pruned_df, features_to_check

if __name__ == "__main__":
    from src.ingestion import load_config, ingest_raw_data
    from src.normality import test_and_transform_features
    try:
        cfg = load_config()
        raw_df = ingest_raw_data("telco_churn_raw.csv", cfg)
        processed_df, _ = test_and_transform_features(raw_df, cfg)
        
        pruned_df, final_features = prune_features_by_vif(processed_df, cfg)
        print(f"\n🎯 Final continuous features passed: {final_features}")
        print(f"🔒 Target column check ('Churn' exists): {cfg['data']['target_variable'] in pruned_df.columns}")
    except Exception as e:
        print(f"Execution Error: {str(e)}")
