from src.inference import FailNetInference
import numpy as np

def compute_voi(inference_engine, decision_network, evidence: dict, target_var: str = "Task_Failure"):
    """
    Computes the Value of Information (VoI) for all unobserved variables in the network.
    VoI = Expected Utility (with information) - Expected Utility (without information)
    """
    unobserved_vars = [node for node in inference_engine.model.nodes() if node not in evidence and node != target_var and node not in ["Failure_Severity"]]
    
    # 1. Base Expected Utility (without additional information)
    base_action, base_eus = decision_network.get_optimal_policy(evidence)
    eu_base = base_eus[base_action]
    
    voi_results = {}
    
    for var in unobserved_vars:
        expected_utility_with_info = 0.0
        
        # Get prior probability of this variable given current evidence
        try:
            var_posterior = inference_engine.ve.query(variables=[var], evidence=evidence, show_progress=False)
        except Exception:
            continue # Variable might be conditionally independent or throw error
            
        for state in var_posterior.state_names[var]:
            prob_state = var_posterior.values[var_posterior.state_names[var].index(state)]
            
            # Formulate new evidence
            new_evidence = evidence.copy()
            new_evidence[var] = state
            
            # Compute EU for optimal action under this specific new evidence
            _, conditional_eus = decision_network.get_optimal_policy(new_evidence)
            best_eu_given_state = max(conditional_eus.values())
            
            expected_utility_with_info += prob_state * best_eu_given_state
            
        voi = expected_utility_with_info - eu_base
        if voi > 0.001:  # Only recommend if there is a meaningful information gain
            voi_results[var] = voi
        
    # Sort descending
    sorted_voi = dict(sorted(voi_results.items(), key=lambda item: item[1], reverse=True))
    return sorted_voi
