import subprocess
import sys
import os

def run_step(name, script_path):
    print(f"\n{'='*50}")
    print(f"Executing: {name}")
    print(f"{'='*50}")
    
    python_exe = sys.executable
    result = subprocess.run([python_exe, "-m", script_path], capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("ERRORS/WARNINGS:")
        print(result.stderr)
        
    if result.returncode != 0:
        print(f"Step '{name}' failed with return code {result.returncode}. Aborting pipeline.")
        sys.exit(result.returncode)
    else:
        print(f"Step '{name}' completed successfully.\n")

if __name__ == "__main__":
    print("Starting FailNet Full Pipeline...")
    
    steps = [
        ("Phase 1 & 2: Data Ingestion (Synthetic)", "src.data_ingestion"),
        ("Phase 3 & 4: BN Structure & Learning", "src.bn_learning"),
        ("Phase 7 & 8: Baseline vs BN Experiment", "experiments.exp1_bn_vs_baseline"),
        ("Experiment 2: Exact vs Approx Inference", "experiments.exp2_exact_vs_approx"),
        ("Experiment 3: MAP Diagnosis", "experiments.exp3_map_diagnosis"),
        ("Experiment 4: Inference Time vs Evidence Size", "experiments.exp4_network_size"),
        ("Experiment 5: Evidence Uncertainty", "experiments.exp5_evidence_uncertainty"),
    ]
    
    for step_name, module_path in steps:
        run_step(step_name, module_path)
        
    print("Pipeline finished successfully! You can now start the dashboard:")
    print("streamlit run dashboard/app.py")
