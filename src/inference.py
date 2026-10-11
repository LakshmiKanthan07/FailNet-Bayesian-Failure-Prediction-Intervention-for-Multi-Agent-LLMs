from pgmpy.inference import VariableElimination
from src.bn_learning import learn_parameters
from src.bn_structure import get_failnet_model
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class FailNetInference:
    def __init__(self, model):
        self.model = model
        self.ve = VariableElimination(self.model)

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



    def streaming_belief_update(self, evidence_sequence: list):
        """Update beliefs iteratively as new evidence items arrive (monitoring)."""
        beliefs = []
        accumulated = {}
        for step_evidence in evidence_sequence:
            accumulated.update(step_evidence)
            beliefs.append(self.ve.query(variables=["Task_Failure"], evidence=accumulated, show_progress=False))
        return beliefs

    # ─────────────────────────────────────────────────────────────────
    #  NEW: Markov Blanket
    # ─────────────────────────────────────────────────────────────────
    def get_markov_blanket(self, target: str = "Task_Failure") -> List[str]:
        """
        Returns the Markov Blanket of the target node.
        
        The Markov Blanket of a node X in a Bayesian Network consists of:
          - Its direct parents
          - Its direct children
          - The other parents of its children (co-parents)
        
        The Markov Blanket is the minimal set of nodes that, when observed,
        renders the target conditionally independent of ALL other nodes.
        This means these are the ONLY variables you need to observe to
        perfectly predict the target.
        
        Args:
            target: The node to compute the blanket for. Defaults to "Task_Failure".
            
        Returns:
            Sorted list of node names in the Markov Blanket.
        """
        blanket = self.model.get_markov_blanket(target)
        return sorted(blanket)

    # ─────────────────────────────────────────────────────────────────
    #  NEW: Sensitivity Analysis (Tornado Chart Data)
    # ─────────────────────────────────────────────────────────────────
    def sensitivity_analysis(
        self,
        target: str = "Task_Failure",
        target_state: str = "Yes",
        base_evidence: Optional[dict] = None,
        exclude_nodes: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Computes sensitivity of P(target=target_state) to each unobserved variable.
        
        For each variable X with domain [x1, x2, ...]:
          - swing = max_state P(target | X=xi) - min_state P(target | X=xi)
          - best_val = state xi that MINIMISES P(target=Yes)
          - worst_val = state xi that MAXIMISES P(target=Yes)
        
        Returns a DataFrame sorted by swing (descending), suitable for a
        Tornado Chart: the wider the bar, the more influence that variable has.
        
        Args:
            target:        Query node name.
            target_state:  State of target to query ("Yes" for failure probability).
            base_evidence: Optional fixed evidence to condition on.
            exclude_nodes: Nodes to exclude from sensitivity sweep.
        
        Returns:
            DataFrame with columns:
              Variable, BestState, BestProb, WorstState, WorstProb, Swing
        """
        if base_evidence is None:
            base_evidence = {}
        if exclude_nodes is None:
            exclude_nodes = []

        # Nodes to exclude: target itself, Failure_Severity (child of target), already-observed
        skip = set(exclude_nodes) | set(base_evidence.keys()) | {target, "Failure_Severity"}

        # Get base probability (no additional evidence)
        base_res = self.ve.query(variables=[target], evidence=base_evidence, show_progress=False)
        base_prob = float(base_res.values[base_res.state_names[target].index(target_state)])

        records = []
        for node in self.model.nodes():
            if node in skip:
                continue
            
            node_cpd = self.model.get_cpds(node)
            if node_cpd is None:
                continue
            states = node_cpd.state_names[node]
            
            probs_per_state = {}
            for state in states:
                ev = {**base_evidence, node: state}
                try:
                    res = self.ve.query(variables=[target], evidence=ev, show_progress=False)
                    p = float(res.values[res.state_names[target].index(target_state)])
                    probs_per_state[state] = p
                except Exception:
                    continue
            
            if len(probs_per_state) < 2:
                continue

            best_state  = min(probs_per_state, key=probs_per_state.get)
            worst_state = max(probs_per_state, key=probs_per_state.get)
            best_prob   = probs_per_state[best_state]
            worst_prob  = probs_per_state[worst_state]
            swing       = worst_prob - best_prob

            records.append({
                "Variable":   node,
                "BestState":  best_state,
                "BestProb":   round(best_prob, 4),
                "WorstState": worst_state,
                "WorstProb":  round(worst_prob, 4),
                "Swing":      round(swing, 4),
                "BaseProb":   round(base_prob, 4),
            })

        df = pd.DataFrame(records).sort_values("Swing", ascending=False).reset_index(drop=True)
        return df


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
    
    # Test 4: Markov Blanket
    print("\nMarkov Blanket of Task_Failure:")
    print(inference.get_markov_blanket("Task_Failure"))
    
    # Test 5: Sensitivity Analysis (top 5)
    print("\nSensitivity Analysis (Top 5 by Swing):")
    sa = inference.sensitivity_analysis()
    print(sa.head(5).to_string(index=False))
