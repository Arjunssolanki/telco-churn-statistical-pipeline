import os
import sys
from src.ingestion import load_config, ingest_raw_data
from src.normality import test_and_transform_features
from src.multicollinearity import prune_features_by_vif
from src.model import execute_production_ml_workflow

def run_pipeline():
    print("==================================================")
    print("🚀 RUNNING PRODUCTION STATISTICAL ML APPLICATION ENGINE")
    print("==================================================")
    
    try:
        # Load centralized configuration setup
        config = load_config("config/config.yaml")
        raw_data_path = "telco_churn_raw.csv"
        
        # Hard constraint check for environment files
        if not os.path.exists(raw_data_path):
            print(f"🛑 Critical Error: Runtime execution halted. Raw file missing at: {raw_data_path}")
            sys.exit(1)
            
        # 1. Ingest raw tabular assets
        df_raw = ingest_raw_data(raw_data_path, config)
        
        # 2. Execute distribution normality testing gates
        print("\n⚡ Running Distribution Normality Gates...")
        df_transformed, normality_registry = test_and_transform_features(df_raw, config)
        
        # 3. Prune overlapping multicollinear parameters
        print("\n⚡ Running Multicollinearity VIF Gates...")
        df_pruned, final_continuous_features = prune_features_by_vif(df_transformed, config)
        
        # 4. Trigger training phase and pickle serialization
        print("\n⚡ Running Performance Assessment & Asset Serialization...")
        execute_production_ml_workflow(df_pruned, final_continuous_features, config, normality_registry)
        
        print("\n==================================================")
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
        print("==================================================")
        
    except Exception as e:
        print(f"\n🛑 Pipeline Core Runtime Failure: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()
