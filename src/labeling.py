"""
labeling.py
-----------
Converts raw task outcome signals from the Who&When dataset into
discrete class labels for the FailNet Bayesian Network target node.

Also provides utilities for encoding/decoding categorical columns
to/from integer representations for sklearn-compatible evaluators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────
#  Primary Label Derivation
# ─────────────────────────────────────────────

def derive_task_failure_label(is_correct: bool) -> str:
    """
    Task_Failure: Yes | No
    The primary binary outcome label for the FailNet BN.

    Args:
        is_correct: Boolean from the Who&When 'is_correct' field.
                    True  → the agent completed the task correctly → No failure.
                    False → the agent failed the task             → Yes failure.

    Returns:
        "No" if is_correct is True, else "Yes".
    """
    return "No" if is_correct else "Yes"


def derive_failure_severity_label(task_fail: str) -> str:
    """
    Failure_Severity: Critical | None
    A downstream consequence node derived from Task_Failure.

    Args:
        task_fail: "Yes" or "No" from derive_task_failure_label.

    Returns:
        "Critical" if task_fail is "Yes", else "None".
    """
    return "Critical" if task_fail == "Yes" else "None"


# ─────────────────────────────────────────────
#  Provenance Tagging
# ─────────────────────────────────────────────

def get_provenance_tag(source: str = "who_and_when_heuristic_parser") -> str:
    """
    Returns a string tag indicating the origin of a data row.
    Used to track which rows are real vs. synthetic vs. augmented.

    Args:
        source: One of:
            - "who_and_when_heuristic_parser"  (real extracted data)
            - "synthetic_success"               (generated balanced samples)
            - "augmented"                       (bootstrapped rows)

    Returns:
        The source string as-is for storage in the 'label_provenance' column.
    """
    valid_sources = {
        "who_and_when_heuristic_parser",
        "synthetic_success",
        "augmented",
    }
    if source not in valid_sources:
        raise ValueError(f"Unknown provenance source '{source}'. Must be one of {valid_sources}")
    return source


# ─────────────────────────────────────────────
#  Categorical Encoding for sklearn Baselines
# ─────────────────────────────────────────────

def encode_dataframe(df: pd.DataFrame, target_col: str = "Task_Failure") -> Tuple[pd.DataFrame, pd.Series]:
    """
    One-hot encodes all non-target categorical columns and binary-encodes the target.
    Used exclusively to prepare data for sklearn-based baseline models (Logistic Regression, XGBoost).
    The Bayesian Network works directly on the original string-valued DataFrame.

    Args:
        df:         The processed evidence table DataFrame.
        target_col: Name of the target column.

    Returns:
        (X_encoded, y): Feature matrix (one-hot encoded) and binary target Series.
    """
    df = df.copy()

    # Drop columns not used in BN evaluation
    drop_cols = [c for c in ["label_provenance", "Failure_Severity"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    y = df[target_col].map({"Yes": 1, "No": 0}).astype(int)
    X = df.drop(columns=[target_col])
    X_encoded = pd.get_dummies(X)

    return X_encoded, y


def get_label_distribution(df: pd.DataFrame, target_col: str = "Task_Failure") -> Dict[str, float]:
    """
    Returns the class distribution of the target column as percentage strings.

    Args:
        df:         The evidence table DataFrame.
        target_col: Name of the target column.

    Returns:
        Dict mapping label value -> percentage (0.0 to 1.0).
    """
    counts = df[target_col].value_counts(normalize=True)
    return counts.to_dict()


def summarize_labels(df: pd.DataFrame, target_col: str = "Task_Failure") -> None:
    """
    Prints a human-readable summary of label distribution and provenance.
    """
    print(f"\n{'='*45}")
    print(f"Label Distribution for '{target_col}'")
    print(f"{'='*45}")
    dist = get_label_distribution(df, target_col)
    for label, pct in dist.items():
        bar = "#" * int(pct * 30)
        print(f"  {label:8s}: {pct*100:5.1f}%  {bar}")

    if "label_provenance" in df.columns:
        print(f"\nProvenance Breakdown:")
        prov_counts = df["label_provenance"].value_counts()
        for prov, cnt in prov_counts.items():
            print(f"  {prov}: {cnt} rows")
    print(f"{'='*45}\n")
