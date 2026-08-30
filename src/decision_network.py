from src.inference import FailNetInference
from src.bn_learning import learn_parameters
import numpy as np

class MockDecisionNetwork:
    """
    Since pycid couldn't be installed on the generic Windows environment due to compilation issues,
    this class implements the utility logic manually. It computes expected utility for each action
    given the posterior probability of failure from the pgmpy model.
    """
    def __init__(self, inference_engine):
        self.inference = inference_engine
        self.actions = ["Continue", "Retry_Step", "Escalate_to_Human", "Halt"]

        self.utility = {
            "Continue": {"No": 10, "Yes": -20},
            "Retry_Step": {"No": 5, "Yes": -5},
            "Escalate_to_Human": {"No": -2, "Yes": -2},
            "Halt": {"No": 0, "Yes": 5}
        }

    def get_optimal_policy(self, evidence: dict):
        # 1. Get posterior of failure
        res = self.inference.posterior_failure(evidence)
        prob_fail = res.values[res.state_names['Task_Failure'].index('Yes')]
        prob_no_fail = 1.0 - prob_fail
        
        # 2. Compute Expected Utility (EU) for each action
        expected_utilities = {}
        for action in self.actions:
            eu = (self.utility[action]["Yes"] * prob_fail) + (self.utility[action]["No"] * prob_no_fail)
            expected_utilities[action] = eu
            
        # 3. Choose max EU
        best_action = max(expected_utilities, key=expected_utilities.get)
        return best_action, expected_utilities

if __name__ == "__main__":
    model = learn_parameters()
    inference = FailNetInference(model)
    dn = MockDecisionNetwork(inference)
    
    # Test case 1: No evidence (low prob fail)
    action, eus = dn.get_optimal_policy({})
    print("No evidence - Optimal Action:", action)
    print("Expected Utilities:", eus)
    
    # Test case 2: Execution error occurred (higher prob fail)
    action, eus = dn.get_optimal_policy({"Execution_Error_Occurred": "Yes"})
    print("\nExecution Error - Optimal Action:", action)
    print("Expected Utilities:", eus)
