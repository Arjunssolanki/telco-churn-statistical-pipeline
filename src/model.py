import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier

def execute_production_ml_workflow(df: pd.DataFrame, continuous_features: list, config: dict, transformers_registry: dict):
    """
    Executes cross-validation for performance metrics verification,
    extracts feature importances, trains the final production model,
    and serializes all deployment assets to binary .pkl files.
    """
    target_var = config["data"]["target_variable"]
    categorical_features = config["data"]["categorical_features"]
    seed = config["model_parameters"]["random_state"]
    estimators = config["model_parameters"]["rf_estimators"]
    cls_weight = config["model_parameters"]["class_weight"]
    cv_splits = config["model_parameters"]["cv_splits"]
    
    # 1. Bulletproof Target Extraction & Mapping Layer
    if target_var not in df.columns:
        raise KeyError(f"Target column '{target_var}' is completely missing from the processed dataframe fields.")
        
    # Convert series to standard string/object format to clear any PyArrow custom types
    target_series = df[target_var].astype(str).str.strip()
    
    # Map target columns values deterministically
    y = target_series.map({'Yes': 1, '1': 1, '1.0': 1, 'No': 0, '0': 0, '0.0': 0})
    
    # Catch any unmapped elements to avoid silent model corruption
    if y.isna().any():
        # Fallback: check if the data is already stored as pure integers/floats
        try:
            y = pd.to_numeric(df[target_var]).astype(int)
        except Exception:
            raise ValueError(f"Target mapping failed. Unique values present in column: {df[target_var].unique()}")
    else:
        y = y.astype(int)
        
    # 2. Categorical dummy encoding configuration
    X_categorical = df[categorical_features].copy()
    for col in categorical_features:
        X_categorical[col] = X_categorical[col].astype(str)
        
    X_encoded_cat = pd.get_dummies(X_categorical, drop_first=True)
    X_continuous = df[continuous_features]
    
    X = pd.concat([X_continuous, X_encoded_cat], axis=1)
    
    # Ensure all data components are aligned
    X = X.loc[y.index]
    
    # 3. Cross-Validation Phase for Performance Auditing
    model_cv = RandomForestClassifier(n_estimators=estimators, class_weight=cls_weight, random_state=seed, n_jobs=-1)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
    
    print(f"\n🚀 Running Stratified {cv_splits}-Fold Cross-Validation...")
    cv_results = cross_validate(model_cv, X, y, cv=cv, scoring=scoring_metrics, return_estimator=True)
    
    # Render validation metrics directly to the console terminal
    print("\n📈 Production Cross-Validated Performance Matrix:")
    print(f"  • Accuracy (Mean):        {np.mean(cv_results['test_accuracy']):.4f}")
    print(f"  • Precision (Weighted):   {np.mean(cv_results['test_precision_weighted']):.4f}")
    print(f"  • Recall (Weighted):      {np.mean(cv_results['test_recall_weighted']):.4f}")
    
    f1_key = 'test_1f1_weighted' if 'test_1f1_weighted' in cv_results else 'test_f1_weighted'
    print(f"  • F1-Score (Weighted):    {np.mean(cv_results[f1_key]):.4f}")
    print(f"  • ROC-AUC (Mean):         {np.mean(cv_results['test_roc_auc']):.4f}")
    
    # Extract mean structural importances across validation folds
    fold_importances = [est.feature_importances_ for est in cv_results['estimator']]
    mean_importances = np.mean(fold_importances, axis=0)
    
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": mean_importances
    }).sort_values(by="Importance", ascending=False)
    
    print("\n🎯 Top 5 Key Structural Predictive Drivers:")
    print(importance_df.head(5).to_string(index=False))
    
    # 4. Final Production Fit & Serialization Phase
    print(f"\n📦 Training final production estimator on all {X.shape} rows...")
    final_model = RandomForestClassifier(n_estimators=estimators, class_weight=cls_weight, random_state=seed, n_jobs=-1)
    final_model.fit(X, y)
    
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(final_model, "artifacts/churn_classifier_model.pkl")
    joblib.dump(list(X.columns), "artifacts/production_feature_columns.pkl")
    joblib.dump(transformers_registry, "artifacts/normality_transformers.pkl")
    
    print("\n💾 Successfully exported production deployment artifacts to 'artifacts/' directory:")
    print("  └── churn_classifier_model.pkl")
    print("  └── production_feature_columns.pkl")
    print("  └── normality_transformers.pkl")
    
    return final_model

if __name__ == "__main__":
    from src.ingestion import load_config, ingest_raw_data
    from src.normality import test_and_transform_features
    from src.multicollinearity import prune_features_by_vif
    try:
        cfg = load_config()
        raw_df = ingest_raw_data("telco_churn_raw.csv", cfg)
        processed_df, norm_reg = test_and_transform_features(raw_df, cfg)
        _, final_cont_features = prune_features_by_vif(processed_df, cfg)
        
        execute_production_ml_workflow(processed_df, final_cont_features, cfg, norm_reg)
    except Exception as e:
        print(f"Execution Error: {str(e)}")
