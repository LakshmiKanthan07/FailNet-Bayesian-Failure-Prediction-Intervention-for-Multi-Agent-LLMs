"""
evaluation.py
-------------
Evaluates the FailNet Bayesian Network on held-out data.

Provides:
  - cross_validate_bn():  5-fold CV giving Accuracy, F1, AUC-ROC, Brier Score
  - partial_evidence_degradation(): accuracy as evidence is progressively removed
  - Both functions return DataFrames suitable for dashboard display.
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, brier_score_loss,
    confusion_matrix,
)
from pgmpy.estimators import BayesianEstimator

from src.bn_structure import get_failnet_model

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────

def _predict_proba(model, test_df: pd.DataFrame) -> np.ndarray:
    """
    For each row in test_df, run Variable Elimination with all columns
    except Task_Failure as evidence. Returns P(Task_Failure=Yes) per row.
    """
    from pgmpy.inference import VariableElimination
    ve = VariableElimination(model)

    feature_cols = [c for c in test_df.columns if c != "Task_Failure"]
    probs = []
    for _, row in test_df.iterrows():
        evidence = {col: row[col] for col in feature_cols
                    if pd.notna(row[col]) and row[col] not in ("", None)}
        try:
            res = ve.query(
                variables=["Task_Failure"],
                evidence=evidence,
                show_progress=False,
            )
            p = float(res.values[res.state_names["Task_Failure"].index("Yes")])
        except Exception:
            p = 0.5
        probs.append(p)
    return np.array(probs)


# ─────────────────────────────────────────────────────────────────
#  5-Fold Cross-Validation
# ─────────────────────────────────────────────────────────────────

def cross_validate_bn(
    data_path: str = "data/processed/bn_evidence_table.csv",
    n_splits: int = 5,
    ess: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Performs n_splits-Fold Stratified Cross-Validation of the FailNet BN.

    Returns a DataFrame with columns:
        Fold, Accuracy, F1, AUC-ROC, Brier Score
    plus a summary row with Mean ± Std.
    """
    df = pd.read_csv(data_path)
    if "label_provenance" in df.columns:
        df = df.drop(columns=["label_provenance"])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_col = "Task_Failure"

    records = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, df[y_col])):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df  = df.iloc[test_idx].reset_index(drop=True)

        # Learn parameters on training fold
        model = get_failnet_model()
        est = BayesianEstimator(model, train_df)
        cpds = est.get_parameters(prior_type="BDeu", equivalent_sample_size=ess)
        model.add_cpds(*cpds)
        if not model.check_model():
            continue

        # Predict on test fold
        probs = _predict_proba(model, test_df)
        preds = (probs >= 0.5).astype(int)
        truth = (test_df[y_col] == "Yes").astype(int).values

        records.append({
            "Fold":        fold_idx + 1,
            "Accuracy":    round(accuracy_score(truth, preds), 4),
            "F1":          round(f1_score(truth, preds, zero_division=0), 4),
            "AUC-ROC":     round(roc_auc_score(truth, probs), 4),
            "Brier Score": round(brier_score_loss(truth, probs), 4),
        })
        print(f"  Fold {fold_idx+1}: Acc={records[-1]['Accuracy']:.3f}  "
              f"F1={records[-1]['F1']:.3f}  "
              f"AUC={records[-1]['AUC-ROC']:.3f}  "
              f"Brier={records[-1]['Brier Score']:.3f}")

    results_df = pd.DataFrame(records)

    # Summary row
    summary = {"Fold": "Mean ± Std"}
    for col in ["Accuracy", "F1", "AUC-ROC", "Brier Score"]:
        m = results_df[col].mean()
        s = results_df[col].std()
        summary[col] = f"{m:.4f} ± {s:.4f}"
    results_df = pd.concat([results_df, pd.DataFrame([summary])], ignore_index=True)

    return results_df


# ─────────────────────────────────────────────────────────────────
#  Partial Evidence Degradation Study
# ─────────────────────────────────────────────────────────────────

def partial_evidence_degradation(
    data_path: str = "data/processed/bn_evidence_table.csv",
    n_samples: int = 60,
    ess: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Shows how BN accuracy degrades (or stays robust) as evidence is
    progressively REMOVED — simulating partial observability.

    For each step level k = [all features, 18, 14, 10, 6, 3, 0]:
      - For each test row, randomly mask (k) features as unobserved
      - Run VE with remaining evidence
      - Measure accuracy

    Returns a DataFrame with columns: Features_Observed, Accuracy, F1, AUC-ROC
    """
    df = pd.read_csv(data_path)
    if "label_provenance" in df.columns:
        df = df.drop(columns=["label_provenance"])

    # Learn model on full data (this is an illustration, not held-out eval)
    model = get_failnet_model()
    est = BayesianEstimator(model, df)
    cpds = est.get_parameters(prior_type="BDeu", equivalent_sample_size=ess)
    model.add_cpds(*cpds)
    model.check_model()

    from pgmpy.inference import VariableElimination
    ve = VariableElimination(model)

    feature_cols = [c for c in df.columns if c not in ("Task_Failure", "Failure_Severity")]
    y_col = "Task_Failure"

    # Sample rows for evaluation
    rng = np.random.default_rng(random_state)
    sample_df = df.sample(min(n_samples, len(df)), random_state=random_state).reset_index(drop=True)
    truth = (sample_df[y_col] == "Yes").astype(int).values

    n_features = len(feature_cols)
    steps = [n_features, 18, 14, 10, 6, 3, 0]
    steps = sorted(set(min(s, n_features) for s in steps), reverse=True)

    records = []
    for k in steps:
        probs = []
        for _, row in sample_df.iterrows():
            if k == 0:
                evidence = {}
            else:
                chosen = rng.choice(feature_cols, size=min(k, len(feature_cols)), replace=False)
                evidence = {col: row[col] for col in chosen
                            if pd.notna(row[col]) and row[col] not in ("", None)}
            try:
                res = ve.query(variables=["Task_Failure"], evidence=evidence, show_progress=False)
                p = float(res.values[res.state_names["Task_Failure"].index("Yes")])
            except Exception:
                p = 0.5
            probs.append(p)

        probs = np.array(probs)
        preds = (probs >= 0.5).astype(int)
        try:
            auc = roc_auc_score(truth, probs)
        except Exception:
            auc = float("nan")

        records.append({
            "Features_Observed": k,
            "Accuracy":          round(accuracy_score(truth, preds), 4),
            "F1":                round(f1_score(truth, preds, zero_division=0), 4),
            "AUC-ROC":           round(auc, 4),
        })
        print(f"  Observed={k:2d} features -> Acc={records[-1]['Accuracy']:.3f}  "
              f"F1={records[-1]['F1']:.3f}  AUC={records[-1]['AUC-ROC']:.3f}")


    return pd.DataFrame(records)


if __name__ == "__main__":
    DATA = "data/processed/bn_evidence_table.csv"
    if not os.path.exists(DATA):
        print("Dataset not found. Run python run_pipeline.py first.")
    else:
        print("=" * 60)
        print("5-Fold Cross-Validation")
        print("=" * 60)
        cv_df = cross_validate_bn(DATA)
        print(cv_df.to_string(index=False))

        print("\n" + "=" * 60)
        print("Partial Evidence Degradation Study")
        print("=" * 60)
        pe_df = partial_evidence_degradation(DATA)
        print(pe_df.to_string(index=False))
