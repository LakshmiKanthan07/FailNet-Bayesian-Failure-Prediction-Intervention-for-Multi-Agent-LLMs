from src.bn_learning import learn_parameters
from src.inference import FailNetInference
from src.decision_network import MockDecisionNetwork
from src.voi import compute_voi

def run_experiment_6():
    print("=== Experiment 6: Value of Information (VoI) for Active Interventions ===")
    
    model = learn_parameters()
    inference = FailNetInference(model)
    dn = MockDecisionNetwork(inference)
    
    scenarios = [
        {"name": "No Evidence (High Baseline Uncertainty)", "evidence": {}},
        {"name": "Task Complexity is High", "evidence": {"Task_Complexity": "High"}},
        {"name": "Execution Error (Certain Failure)", "evidence": {"Execution_Error_Occurred": "Yes"}},
    ]
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"Evidence: {scenario['evidence']}")
        
        best_action, expected_utilities, best_diagnostic, best_voi_value = dn.recommend_active_intervention(scenario['evidence'])
        
        print(f"Current Optimal Action: {best_action}")
        
        if best_diagnostic:
            print(f"Active Recommendation: Query '{best_diagnostic}'")
            print(f"Expected Utility Gain (VoI): {best_voi_value:.4f}")
        else:
            print("Active Recommendation: None (No query will significantly alter the optimal decision)")

if __name__ == "__main__":
    run_experiment_6()
