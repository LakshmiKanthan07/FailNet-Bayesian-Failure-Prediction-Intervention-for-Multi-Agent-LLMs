from src.bn_learning import learn_parameters
from src.inference import FailNetInference

def run_experiment_3():
    print("=== Experiment 3: MAP Diagnosis ===")
    
    model = learn_parameters()
    inference = FailNetInference(model)
    
    scenarios = [
        {"name": "Task Failed", "evidence": {"Task_Failure": "Yes"}},
        {"name": "Task Failed, Good Plan", "evidence": {"Task_Failure": "Yes", "Plan_Quality": "Good"}},
        {"name": "Task Failed, High Complexity", "evidence": {"Task_Failure": "Yes", "Task_Complexity": "High"}},
    ]
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"Evidence: {scenario['evidence']}")
        
        try:
            map_res = inference.map_diagnosis(scenario['evidence'])
            print("MAP Diagnosis (Most Likely Causes):")
            for k, v in map_res.items():
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"Failed to compute MAP diagnosis: {e}")

if __name__ == "__main__":
    run_experiment_3()