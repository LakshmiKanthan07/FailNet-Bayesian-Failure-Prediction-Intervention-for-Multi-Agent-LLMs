import pandas as pd
import numpy as np
import os
import warnings
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from sklearn.model_selection import KFold
from src.bn_structure import get_failnet_model

warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.metrics import log_loss
def evaluate_estimators_cv(model, df, k=5):
    """
    Performs K-Fold Cross Validation to compare Maximum Likelihood vs BDeu Priors.
    Evaluates based on Log-Loss for predicting Task_Failure.
    Returns the name of the best estimator and its parameters.
    """
    from src.inference import FailNetInference
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    estimators = {
        "MLE": {"type": "MLE"},
        "BDeu_1": {"type": "BDeu", "ess": 1},
        "BDeu_5": {"type": "BDeu", "ess": 5},
        "BDeu_10": {"type": "BDeu", "ess": 10},
    }
    
    scores = {name: [] for name in estimators}
    
    # We do a simplified evaluation: fitting on train, scoring on val
    for train_idx, val_idx in kf.split(df):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        
        # Prepare validation targets
        y_true = (val_df["Task_Failure"] == "Yes").astype(int)
        
        for name, config in estimators.items():
            fold_model = model.copy()
            if config["type"] == "MLE":
                est = MaximumLikelihoodEstimator(fold_model, train_df)
                fold_model.add_cpds(*est.get_parameters())
            else:
                est = BayesianEstimator(fold_model, train_df)
                fold_model.add_cpds(*est.get_parameters(prior_type="BDeu", equivalent_sample_size=config["ess"]))
            
            try:
                inference = FailNetInference(fold_model)
                y_preds = []
                for _, row in val_df.iterrows():
                    # Predict using evidence (all nodes except target and severity)
                    ev = {c: row[c] for c in df.columns if c not in ["Task_Failure", "Failure_Severity"] and not pd.isna(row[c])}
                    res = inference.posterior_failure(ev)
                    prob_yes = res.values[res.state_names['Task_Failure'].index('Yes')]
                    y_preds.append(prob_yes)
                
                score = log_loss(y_true, y_preds)
                scores[name].append(score)
            except Exception:
                scores[name].append(np.inf) # High penalty
                
    # Average the log-loss (lower is better)
    avg_scores = {name: np.mean(val) for name, val in scores.items()}
    best_estimator = min(avg_scores, key=avg_scores.get)
    print(f"CV Log-Loss Scores (Lower is Better): {avg_scores}")
    print(f"Selected Optimal Estimator: {best_estimator}")
    
    return estimators[best_estimator]

def learn_parameters(data_path="data/processed/bn_evidence_table.csv", output_path="failnet_model.bif", optimize=True):
    """
    Learn the Conditional Probability Tables (CPDs) for the FailNet BN.
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    if "label_provenance" in df.columns:
        df = df.drop(columns=["label_provenance"])
        
    model = get_failnet_model(df)    
    
    if optimize:
        print("Running K-Fold CV to select the optimal parameter estimator...")
        best_config = evaluate_estimators_cv(model, df)
    else:
        best_config = {"type": "BDeu", "ess": 10}
        
    print(f"Fitting final model using {best_config}...")
    if best_config["type"] == "MLE":
        estimator = MaximumLikelihoodEstimator(model, df)
        cpds = estimator.get_parameters()
    else:
        estimator = BayesianEstimator(model, df)
        cpds = estimator.get_parameters(prior_type="BDeu", equivalent_sample_size=best_config["ess"])
        
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