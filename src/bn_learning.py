import pandas as pd
from pgmpy.estimators import BayesianEstimator
from src.bn_structure import get_failnet_model
import os
def learn_parameters(data_path="data/processed/bn_evidence_table.csv", output_path="failnet_model.bif"):
    
    """
    Learn the Conditional Probability Tables (CPDs) for the FailNet BN using Bayesian Estimation (K2 prior).
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    if "label_provenance" in df.columns:
        df = df.drop(columns=["label_provenance"])
        
    model = get_failnet_model()    
    print("Fitting model with BayesianEstimator and K2 prior...")
    estimator = BayesianEstimator(model, df)

    cpds = estimator.get_parameters(prior_type="K2", equivalent_sample_size=0.1)
    model.add_cpds(*cpds)
    print("Checking model consistency...")
    assert model.check_model()
    print("Model is consistent! All CPDs are valid and normalized.")
    return model

if __name__ == "__main__":
    if not os.path.exists("data/processed/bn_evidence_table.csv"):
        print("Data not found. Run data_ingestion.py first.")
    else:
        model = learn_parameters()
        print("Model learning completed successfully.")