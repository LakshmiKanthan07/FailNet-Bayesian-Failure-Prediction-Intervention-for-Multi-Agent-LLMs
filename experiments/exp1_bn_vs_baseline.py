from src.bn_learning import learn_parameters
from src.inference import FailNetInference
from src.baselines import evaluate_baselines
from sklearn.metrics import f1_score, log_loss, brier_score_loss, classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os

def run_experiment_1():
    print("=== Experiment 1: BN vs Baselines ===")
    
    import time
    import matplotlib.pyplot as plt
    
    # 1. Train and evaluate Baselines
    print("\nTraining Logistic Regression and XGBoost...")
    baseline_results = evaluate_baselines()
    
    # 2. Train and evaluate BN
    print("\nTraining Bayesian Network...")
    model = learn_parameters()
    inference = FailNetInference(model)
    
    # Load test data to evaluate BN
    df = pd.read_csv("data/processed/bn_evidence_table.csv")
    if "label_provenance" in df.columns:
        df = df.drop(columns=["label_provenance"])
    if "Failure_Severity" in df.columns:
        df = df.drop(columns=["Failure_Severity"])
        
    y = df["Task_Failure"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Task_Failure"])
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("\nEvaluating Bayesian Network on Test Set...")
    bn_probs = []
    bn_preds = []
    
    # Evaluate point by point (inefficient but exact for BN inference)
    start_time = time.time()
    for idx, row in X_test.iterrows():
        # Convert row to evidence dict
        evidence = row.to_dict()
        
        try:
            res = inference.posterior_failure(evidence)
            prob_fail = res.values[res.state_names['Task_Failure'].index('Yes')]
        except Exception as e:
            # Fallback if inference fails due to zero probability evidence
            prob_fail = 0.5
            
        bn_probs.append(prob_fail)
        bn_preds.append(1 if prob_fail >= 0.5 else 0)
    bn_time = time.time() - start_time
        
    print("=== Bayesian Network ===")
    print(classification_report(y_test, bn_preds))
    print("Log Loss:", log_loss(y_test, bn_probs))
    print("Brier Score:", brier_score_loss(y_test, bn_probs))
    print(f"Inference Time on Test Set: {bn_time:.4f}s")
    
    # --- Plotting Comparisons ---
    os.makedirs("report/figures", exist_ok=True)
    
    models = ['Logistic Regression', 'XGBoost', 'Bayesian Network']
    times = [baseline_results['lr_time'], baseline_results['xgb_time'], bn_time]
    
    bn_f1 = classification_report(y_test, bn_preds, output_dict=True)["1"]["f1-score"]
    f1_scores = [baseline_results['lr_f1'], baseline_results['xgb_f1'], bn_f1]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Inference Time (s)', color=color)
    bars1 = ax1.bar([x - 0.2 for x in range(len(models))], times, width=0.4, color=color, label='Inference Time (s)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('F1-Score (Failure=Yes)', color=color)  
    bars2 = ax2.bar([x + 0.2 for x in range(len(models))], f1_scores, width=0.4, color=color, label='F1-Score')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Comparison of Inference Time and F1-Score')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models)
    
    fig.tight_layout()  
    plt.savefig("report/figures/exp1_comparison.png")
    print("\nComparison plot saved to report/figures/exp1_comparison.png")

if __name__ == "__main__":
    run_experiment_1()
