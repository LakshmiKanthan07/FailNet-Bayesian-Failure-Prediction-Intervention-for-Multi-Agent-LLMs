"""
exp9_partial_evidence.py
------------------------
Experiment 9 — Partial Evidence Robustness Study

Key claim of FailNet: "The Bayesian Network handles partial evidence
gracefully — it marginalizes out unobserved variables. Unlike XGBoost
or Logistic Regression, which crash or require imputation."

This experiment PROVES that claim quantitatively:
  - For varying numbers of observed features (from all 22 down to 0)
  - Run inference on a test set
  - Plot Accuracy, F1, AUC-ROC vs. number of features observed

Expected result: BN accuracy degrades gracefully, staying > chance
even with only 3–5 features observed.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.evaluation import partial_evidence_degradation

OUTPUT_DIR = os.path.join(ROOT, "report", "figures")
CSV_DIR    = os.path.join(ROOT, "report")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

DATA_PATH = os.path.join(ROOT, "data", "processed", "bn_evidence_table.csv")


def run():
    print("=" * 60)
    print("Experiment 9 -- Partial Evidence Robustness Study")
    print("=" * 60)


    if not os.path.exists(DATA_PATH):
        print("Dataset not found. Run python run_pipeline.py first.")
        return

    pe_df = partial_evidence_degradation(DATA_PATH, n_samples=80)

    # ── Save CSV ────────────────────────────────────────────────────
    csv_path = os.path.join(CSV_DIR, "exp9_partial_evidence.csv")
    pe_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    print(pe_df.to_string(index=False))

    # ── Plot ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    colors = {"Accuracy": "#22c55e", "F1": "#818cf8", "AUC-ROC": "#f59e0b"}
    for metric, color in colors.items():
        ax.plot(
            pe_df["Features_Observed"], pe_df[metric],
            marker="o", linewidth=2.5, markersize=8,
            color=color, label=metric,
        )
        # Annotate each point
        for _, row in pe_df.iterrows():
            ax.annotate(
                f"{row[metric]:.2f}",
                xy=(row["Features_Observed"], row[metric]),
                xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=8.5, color=color,
            )

    # Chance baseline (balanced = 0.5)
    ax.axhline(0.5, color="#475569", linestyle="--", linewidth=1.2, alpha=0.7,
               label="Chance baseline (0.50)")

    ax.set_xlabel("Number of Features Observed (out of 22)", fontsize=12, color="#e2e8f0")
    ax.set_ylabel("Score", fontsize=12, color="#e2e8f0")
    ax.set_title(
        "FailNet — Partial Evidence Robustness Study\n"
        "BN accuracy degrades gracefully as fewer variables are observed",
        fontsize=13, fontweight="bold", color="#f1f5f9",
    )
    ax.legend(fontsize=10, facecolor="#0f172a", labelcolor="#e2e8f0",
              edgecolor="#334155", loc="lower left")
    ax.set_ylim(0.3, 1.05)
    ax.set_xlim(-0.5, pe_df["Features_Observed"].max() + 0.5)
    ax.tick_params(colors="#94a3b8")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.grid(True, color="#334155", alpha=0.5, linestyle="--")

    fig.tight_layout()
    img_path = os.path.join(OUTPUT_DIR, "exp9_partial_evidence.png")
    fig.savefig(img_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Plot saved to:    {img_path}")

    return pe_df


if __name__ == "__main__":
    run()
