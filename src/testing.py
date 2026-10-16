import numpy as np
import pandas as pd
from scipy import stats

def execute_hypothesis_routing(df: pd.DataFrame, normality_registry: dict, variance_registry: dict, config: dict) -> pd.DataFrame:
    continuous_cols = config["data"]["continuous_features"]
    target_var = config["data"]["target_variable"]
    alpha = config["statistical_thresholds"]["alpha_hypothesis"]
    
    test_results = []
    
    for col in continuous_cols:
        # Fallback support to capture both Arrow string metrics or mapped numeric indicators
        unique_targets = df[target_var].unique()
        if 0 in unique_targets or 1 in unique_targets:
            g0 = df[df[target_var] == 0][col].dropna().values
            g1 = df[df[target_var] == 1][col].dropna().values
        elif '0' in unique_targets or '1' in unique_targets:
            g0 = df[df[target_var] == '0'][col].dropna().values
            g1 = df[df[target_var] == '1'][col].dropna().values
        else:
            g0 = df[df[target_var].isin(['No', 'no', 'NO'])][col].dropna().values
            g1 = df[df[target_var].isin(['Yes', 'yes', 'YES'])][col].dropna().values

        is_normal = normality_registry.get(col, False)
        has_equal_var = variance_registry.get(col, {}).get("equal_variance", False)
        
        # Inferences routing engine logic matrix
        if is_normal and has_equal_var:
            stat, p_value = stats.ttest_ind(g0, g1, equal_var=True)
            test_name = "Independent t-Test"
        elif is_normal and not has_equal_var:
            stat, p_value = stats.ttest_ind(g0, g1, equal_var=False)
            test_name = "Welch's t-Test"
        else:
            stat, p_value = stats.mannwhitneyu(g0, g1)
            test_name = "Mann-Whitney U"
            
        # Calculate Cohen's d effect size for clinical tracking
        mean_diff = np.mean(g0) - np.mean(g1)
        pooled_std = np.sqrt((np.var(g0, ddof=1) + np.var(g1, ddof=1)) / 2)
        cohens_d = abs(mean_diff / pooled_std) if pooled_std != 0 else 0
        
        passed_gate = p_value < alpha
        
        result = {
            "Feature": col,
            "Test Executed": test_name,
            "p-value": float(p_value),
            "Cohen's d": float(cohens_d),
            "Statistically Significant": passed_gate
        }
        test_results.append(result)
        
        print(f"🔬 [{col}] Routed to -> {test_name} | p-value: {p_value:.5f} | Effect Size (d): {cohens_d:.4f} | Pass Gate: {passed_gate}")
        
    return pd.DataFrame(test_results)

if __name__ == "__main__":
    from src.ingestion import load_config, ingest_raw_data
    from src.normality import test_and_transform_features
    from src.variance import evaluate_feature_variance
    try:
        cfg = load_config()
        raw_df = ingest_raw_data("telco_churn_raw.csv", cfg)
        processed_df, norm_reg = test_and_transform_features(raw_df, cfg)
        var_reg = evaluate_feature_variance(processed_df, norm_reg, cfg)
        
        print("\n🚀 Initiating Inference Routing System...")
        results_df = execute_hypothesis_routing(processed_df, norm_reg, var_reg, cfg)
    except Exception as e:
        print(f"Execution Error: {str(e)}")
