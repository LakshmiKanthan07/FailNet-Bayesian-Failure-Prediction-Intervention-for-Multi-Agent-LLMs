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
    ]
    
    for step_name, module_path in steps:
        run_step(step_name, module_path)
        
    print("Pipeline finished successfully! You can now start the dashboard:")
    print("streamlit run dashboard/app.py")
