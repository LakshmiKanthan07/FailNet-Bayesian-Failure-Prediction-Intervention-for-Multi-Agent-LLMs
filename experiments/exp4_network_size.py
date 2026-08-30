from src.bn_learning import learn_parameters
from src.inference import FailNetInference
import time
import matplotlib.pyplot as plt
import os

def run_experiment_4():
    print("=== Experiment 4: Inference Time vs Evidence Size ===")
    
    model = learn_parameters()
    inference = FailNetInference(model)
    
    # Evidence sequence to add one by one
    evidence_nodes = [
        ("Task_Complexity", "High"),
        ("Base_Model_Capability", "Low"),
        ("Instruction_Ambiguity", "High"),
        ("Environment_Volatility", "High"),
        ("Tool_Reliability", "Low"),
        ("Human_Oversight_Present", "No"),
        ("Multi_Agent_Coordination_Required", "Yes")
    ]
    
    current_evidence = {}
    sizes = []
    times = []
    
    for k, v in evidence_nodes:
        current_evidence[k] = v
        
        start = time.time()
        res = inference.posterior_failure(current_evidence)
        elapsed = time.time() - start
        
        sizes.append(len(current_evidence))
        times.append(elapsed)
        print(f"Evidence Size: {len(current_evidence)} | Time: {elapsed:.4f}s")
        
    # Plotting
    os.makedirs("report/figures", exist_ok=True)
    plt.figure()
    plt.plot(sizes, times, marker='o')
    plt.title("Exact Inference Time vs Evidence Size")
    plt.xlabel("Number of Evidence Variables")
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    plt.savefig("report/figures/exp4_inference_time.png")
    print("\nPlot saved to report/figures/exp4_inference_time.png")

if __name__ == "__main__":
    run_experiment_4()
