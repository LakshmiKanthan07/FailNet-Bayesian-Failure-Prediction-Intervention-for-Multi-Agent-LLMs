from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.sampling import GibbsSampling
from src.bn_learning import learn_parameters
from src.bn_structure import get_failnet_model
import pandas as pd

class FailNetInference:
    def __init__(self, model):
        self.model = model
        self.ve = VariableElimination(self.model)
        try:
            self.bp = BeliefPropagation(self.model)
        except Exception as e:
            print("BeliefPropagation init failed (could be an issue with graph structure):", e)
            self.bp = None

    def posterior_failure(self, evidence: dict):
        """Exact marginal P(Task_Failure | evidence) via Variable Elimination."""
        res = self.ve.query(variables=["Task_Failure"], evidence=evidence, show_progress=False)
        return res

    def map_diagnosis(self, evidence: dict):
        """Most probable causal configuration given an observed failure."""
        vars_to_query = ["Plan_Quality", "Tool_Selection_Correctness", "Hallucination_Risk",
                         "Intermediate_Verification_Quality"]
        vars_to_query = [v for v in vars_to_query if v not in evidence]
        
        return self.ve.map_query(
            variables=vars_to_query,
            evidence=evidence, show_progress=False,
        )

    def approximate_posterior(self, evidence: dict, n_samples: int = 5000):
        """Likelihood Weighting approximate posterior — fast & robust for evidence queries."""
        from pgmpy.sampling import BayesianModelSampling
        sampler = BayesianModelSampling(self.model)
        if evidence:
            ev_tuples = [(k, v) for k, v in evidence.items()]
            samples = sampler.likelihood_weighted_sample(evidence=ev_tuples, size=n_samples, show_progress=False)
            grouped = samples.groupby("Task_Failure")["_weight"].sum()
            total_wt = grouped.sum()
            if total_wt > 0:
                return (grouped / total_wt).to_dict()
            return samples["Task_Failure"].value_counts(normalize=True).to_dict()
        else:
            samples = sampler.forward_sample(size=n_samples, show_progress=False)
            return samples["Task_Failure"].value_counts(normalize=True).to_dict()

    def streaming_belief_update(self, evidence_sequence: list):
        """Belief propagation re-run as each new evidence item arrives — the monitoring use case."""
        beliefs = []
        accumulated = {}
        for step_evidence in evidence_sequence:
            accumulated.update(step_evidence)
            if self.bp:
                beliefs.append(self.bp.query(variables=["Task_Failure"], evidence=accumulated, show_progress=False))
            else:
                beliefs.append(self.ve.query(variables=["Task_Failure"], evidence=accumulated, show_progress=False))
        return beliefs

if __name__ == "__main__":
    # Test inference
    model = learn_parameters()
    inference = FailNetInference(model)
    
    # Test 1: Prior marginal
    print("Prior Marginal for Task_Failure:")
    print(inference.posterior_failure({}))
    
    # Test 2: Posterior given Execution_Error_Occurred = Yes
    print("\nPosterior given Execution_Error_Occurred=Yes:")
    print(inference.posterior_failure({"Execution_Error_Occurred": "Yes"}))
    
    # Test 3: MAP diagnosis
    print("\nMAP Diagnosis given Task_Failure=Yes:")
    print(inference.map_diagnosis({"Task_Failure": "Yes"}))
