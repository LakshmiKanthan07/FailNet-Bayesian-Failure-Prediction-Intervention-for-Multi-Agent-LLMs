"""
exp8_decision_boundary.py
--------------------------
Experiment 8: Decision Boundary & Utility Sensitivity Analysis

Generates:
  1. The Expected Utility vs. P(Failure) decision boundary curve.
  2. A table of analytical crossover thresholds for all action pairs.
  3. K-Fold cross-validation results for the Bayesian Network.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations

from src.bn_learning import learn_parameters
from src.inference import FailNetInference
from src.decision_network import MockDecisionNetwork, UTILITY_TABLE


def run_experiment_8():
    print("=" * 60)
    print("Experiment 8: Decision Boundary & K-Fold Cross-Validation")
    print("=" * 60)

    model = learn_parameters()
    inference = FailNetInference(model)
    dn = MockDecisionNetwork(inference)
    os.makedirs("report/figures", exist_ok=True)

    # ─── Part A: Decision Boundary Plot ───────────────────────────────
    print("\n[Part A] Generating Decision Boundary Curve...")
    dn.decision_boundary_curve(save_path="report/figures/exp8_decision_boundary.png")

    # ─── Part B: Analytical Crossover Table ───────────────────────────
    print("\n[Part B] Analytical Decision Boundaries (all action pairs):")
    actions = list(UTILITY_TABLE.keys())
    rows = []
    for a, b in combinations(actions, 2):
        p_star = dn.get_decision_boundary(a, b)
        rows.append({
            "Action A":   a,
            "Action B":   b,
            "P* (crossover)": f"{p_star:.4f}" if not np.isnan(p_star) else "No crossing (parallel)",
            "Interpretation": (
                f"Below p*={p_star:.2f}: prefer '{a}'. Above: prefer '{b}'."
                if not np.isnan(p_star) and 0 < p_star < 1
                else "One action always dominates."
            )
        })
    boundary_df = pd.DataFrame(rows)
    print(boundary_df.to_string(index=False))

    # Save to CSV
    boundary_df.to_csv("report/exp8_decision_boundaries.csv", index=False)
    print("\n  -> Saved: report/exp8_decision_boundaries.csv")

    # --- Part C: K-Fold Cross-Validation for BN ---
    print("\n[Part C] K-Fold Cross-Validation for Bayesian Network (k=5)...")
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        accuracy_score, f1_score, brier_score_loss, log_loss
    )
    from src.bn_structure import get_failnet_model
    from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

    df = pd.read_csv("data/processed/bn_evidence_table.csv")
    if "label_provenance" in df.columns:
        df = df.drop(columns=["label_provenance"])

    y = (df["Task_Failure"] == "Yes").astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = {
        "Accuracy": [], "F1": [], "Brier": [], "LogLoss": []
    }

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(df, y)):
        print(f"  Fold {fold_i + 1}/5 ...", end=" ", flush=True)
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)

        fold_model = get_failnet_model()
        try:
            est = BayesianEstimator(fold_model, train_df)
            fold_model.add_cpds(*est.get_parameters(prior_type="BDeu", equivalent_sample_size=5))
        except Exception:
            est = MaximumLikelihoodEstimator(fold_model, train_df)
            fold_model.add_cpds(*est.get_parameters())

        fold_inference = FailNetInference(fold_model)
        y_true_fold = y.iloc[val_idx].values
        y_probs = []

        for _, row in val_df.iterrows():
            ev = {
                c: row[c]
                for c in val_df.columns
                if c not in ["Task_Failure", "Failure_Severity"] and not pd.isna(row[c])
            }
            try:
                res = fold_inference.posterior_failure(ev)
                p = float(res.values[res.state_names["Task_Failure"].index("Yes")])
            except Exception:
                p = 0.5
            y_probs.append(p)

        y_preds = [1 if p >= 0.5 else 0 for p in y_probs]
        cv_results["Accuracy"].append(accuracy_score(y_true_fold, y_preds))
        cv_results["F1"].append(f1_score(y_true_fold, y_preds, zero_division=0))
        cv_results["Brier"].append(brier_score_loss(y_true_fold, y_probs))
        cv_results["LogLoss"].append(log_loss(y_true_fold, y_probs))
        print(f"Acc={cv_results['Accuracy'][-1]:.3f}, F1={cv_results['F1'][-1]:.3f}")

    print("\n  === K-Fold CV Summary (Bayesian Network, k=5) ===")
    summary_rows = []
    for metric, vals in cv_results.items():
        mean_val = np.mean(vals)
        std_val  = np.std(vals)
        summary_rows.append({
            "Metric": metric,
            "Mean":   round(mean_val, 4),
            "Std Dev": round(std_val,  4),
            "Min":    round(min(vals), 4),
            "Max":    round(max(vals), 4),
        })
        print(f"  {metric:10s}: {mean_val:.4f} +/- {std_val:.4f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("report/exp8_kfold_cv_results.csv", index=False)
    print("\n  -> Saved: report/exp8_kfold_cv_results.csv")

    # ─── Part D: CV Results Bar Chart ─────────────────────────────────
    metrics_to_plot = ["Accuracy", "F1", "Brier", "LogLoss"]
    means = [np.mean(cv_results[m]) for m in metrics_to_plot]
    stds  = [np.std(cv_results[m])  for m in metrics_to_plot]

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]
    bars = ax.bar(metrics_to_plot, means, yerr=stds, capsize=6,
                  color=bar_colors, alpha=0.85, edgecolor="white", linewidth=1.2)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Bayesian Network — 5-Fold Cross-Validation Results\n(Error bars = +/- 1 std)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("report/figures/exp8_kfold_cv.png", dpi=150)
    plt.close()
    print("  -> Saved: report/figures/exp8_kfold_cv.png")

    print("\nExperiment 8 Completed.")


if __name__ == "__main__":
    run_experiment_8()
