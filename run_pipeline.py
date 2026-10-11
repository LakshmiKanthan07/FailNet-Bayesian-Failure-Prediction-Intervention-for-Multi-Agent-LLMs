"""
run_pipeline.py
---------------
FailNet Full Pipeline Runner.

Executes all phases in order:
  1. Data Ingestion & Feature Engineering
  2. BN Structure & Parameter Learning
  3. Experiments 1–8 (evaluations, comparisons, sensitivity, CV)
  4. Consolidated final results summary
"""

import subprocess
import sys
import os
import time


def run_step(name: str, script_path: str) -> float:
    """Runs a pipeline step as a subprocess module. Returns elapsed time."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    python_exe = sys.executable
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [python_exe, "-m", script_path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    elapsed = time.time() - t0

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        # Show warnings/errors but don't treat all stderr as fatal
        for line in result.stderr.splitlines():
            if any(kw in line.lower() for kw in ["error", "traceback", "exception"]):
                print(f"  [!]  {line}")

    if result.returncode != 0:
        print(f"\n[FAILED] Step '{name}' failed (exit code {result.returncode}). Aborting.")
        sys.exit(result.returncode)
    else:
        print(f"\n[OK] '{name}' completed in {elapsed:.1f}s")

    return elapsed


def generate_final_summary():
    """
    Reads result CSVs produced by experiments and prints a consolidated
    comparison table: BN vs. Logistic Regression vs. XGBoost.
    """
    import pandas as pd
    import os

    print(f"\n{'='*60}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*60}")

    # K-Fold CV results (Exp 8)
    cv_path = "report/exp8_kfold_cv_results.csv"
    if os.path.exists(cv_path):
        cv_df = pd.read_csv(cv_path)
        print("\n[CV] Bayesian Network -- 5-Fold Cross-Validation:")
        print(cv_df.to_string(index=False))
    else:
        print(f"\n[!] CV results not found at '{cv_path}'.")

    # Decision boundaries (Exp 8)
    db_path = "report/exp8_decision_boundaries.csv"
    if os.path.exists(db_path):
        db_df = pd.read_csv(db_path)
        print("\n[DB] Decision Boundaries (P* where optimal action switches):")
        print(db_df[["Action A", "Action B", "P* (crossover)"]].to_string(index=False))

    print(f"\n{'='*60}")
    print("  All experiment figures saved to: report/figures/")
    print("  Launch dashboard: streamlit run dashboard/app.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("\n>>> Starting FailNet Full Pipeline...\n")

    total_start = time.time()

    steps = [
        ("Phase 1 & 2: Data Ingestion & Feature Engineering",
         "src.data_ingestion"),
        ("Phase 3 & 4: BN Structure & Parameter Learning",
         "src.bn_learning"),
        ("Exp 5: Sensitivity Analysis & Tornado Chart",
         "experiments.exp5_evidence_uncertainty"),
        ("Exp 8: Decision Boundary & K-Fold CV",
         "experiments.exp8_decision_boundary"),
        ("Exp 9: Partial Evidence Robustness Study",
         "experiments.exp9_partial_evidence"),
    ]

    step_times = {}
    for step_name, module_path in steps:
        elapsed = run_step(step_name, module_path)
        step_times[step_name] = elapsed

    # Print timing summary
    print(f"\n{'='*60}")
    print("  Pipeline Timing Summary")
    print(f"{'='*60}")
    for name, t in step_times.items():
        print(f"  {t:5.1f}s  {name}")
    print(f"  {'-'*40}")
    print(f"  {time.time() - total_start:5.1f}s  TOTAL")

    # Print consolidated results
    generate_final_summary()
