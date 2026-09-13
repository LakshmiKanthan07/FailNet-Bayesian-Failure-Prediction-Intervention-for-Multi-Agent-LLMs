from src.bn_learning import learn_parameters
from src.inference import FailNetInference
import matplotlib.pyplot as plt
import os

def run_experiment_5():
    print("=== Experiment 5: Evidence Uncertainty (Streaming Belief Update) ===")
    
    model = learn_parameters()
    inference = FailNetInference(model)
    
    # Simulate an agent execution trace
    trace = [
        {"Task_Complexity": "High", "Base_Model_Capability": "High"}, # Step 1: Initial context
        {"Plan_Quality": "Flawed"}, # Step 2: Planning phase
        {"Tool_Selection_Correctness": "Incorrect"}, # Step 3: Tool use
        {"Execution_Error_Occurred": "Yes"}, # Step 4: Execution result
        {"Error_Recovery_Success": "No"} # Step 5: Recovery result
    ]
    
    print("\nStreaming Evidence:")
    beliefs = inference.streaming_belief_update(trace)
    
    prob_fail = []
    cumulative_evidence = {}
    for i, (step, belief) in enumerate(zip(trace, beliefs)):
        cumulative_evidence.update(step)
        p = belief.values[belief.state_names['Task_Failure'].index('Yes')]
        prob_fail.append(p)
        print(f"Step {i+1}: Evidence added = {step} -> P(Task_Failure=Yes) = {p:.4f}")
        
    # Plotting
    os.makedirs("report/figures", exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(trace)+1), prob_fail, marker='s', color='r')
    plt.title("Streaming Belief Update for Task Failure")
    plt.xlabel("Execution Step")
    plt.ylabel("P(Task_Failure = Yes)")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig("report/figures/exp5_streaming_belief.png")
    print("\nPlot saved to report/figures/exp5_streaming_belief.png")

if __name__ == "__main__":
    run_experiment_5()
