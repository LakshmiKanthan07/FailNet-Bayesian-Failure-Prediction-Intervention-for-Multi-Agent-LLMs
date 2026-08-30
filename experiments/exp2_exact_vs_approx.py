from src.bn_learning import learn_parameters
from src.inference import FailNetInference
import time
import pandas as pd

def run_experiment_2():
    print("=== Experiment 2: Exact vs Approximate Inference ===")
    
    model = learn_parameters()
    inference = FailNetInference(model)
    
    # Test cases
    test_cases = [
        {"name": "No Evidence", "evidence": {}},
        {"name": "Single Evidence", "evidence": {"Execution_Error_Occurred": "Yes"}},
        {"name": "Multiple Evidence", "evidence": {"Execution_Error_Occurred": "Yes", "Task_Complexity": "High", "Base_Model_Capability": "Low"}}
    ]
    
    results = []
    
    for tc in test_cases:
        evidence = tc["evidence"]
        
        # Exact (Variable Elimination)
        start_time = time.time()
        res_exact = inference.posterior_failure(evidence)
        prob_exact = res_exact.values[res_exact.state_names['Task_Failure'].index('Yes')]
        time_exact = time.time() - start_time
        
        # Approximate (Gibbs Sampling)
        start_time = time.time()
        try:
            res_approx = inference.approximate_posterior(evidence, n_samples=1000)
            prob_approx = res_approx.get('Yes', 0.0)
        except Exception as e:
            print(f"Gibbs sampling failed for {tc['name']}: {e}")
            prob_approx = None
        time_approx = time.time() - start_time
        
        results.append({
            "Scenario": tc["name"],
            "Exact_Prob": prob_exact,
            "Exact_Time(s)": time_exact,
            "Approx_Prob": prob_approx,
            "Approx_Time(s)": time_approx
        })
        
    df = pd.DataFrame(results)
    print("\nResults:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_experiment_2()
