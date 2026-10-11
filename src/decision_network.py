"""
decision_network.py
-------------------
Implements a Mock Decision Network (MDN) over the FailNet Bayesian Network.

Since pycid couldn't be installed on the generic Windows environment due to
compilation issues, this class implements the utility logic manually. It computes
Expected Utility (EU) for each action given the posterior probability of failure
from the pgmpy model.

New in this version:
  - Justified utility table with documented rationale.
  - sensitivity_to_utility(): shows the decision boundary for each action pair.
  - decision_boundary_curve(): generates a full plot of EU vs. P(Fail) for all actions.
"""

from src.inference import FailNetInference
from src.bn_learning import learn_parameters
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


# ─────────────────────────────────────────────────────────────
#  Utility Table — Rationale
# ─────────────────────────────────────────────────────────────
# Utility[action][outcome] = points awarded
#
# "Continue" (no intervention):
#   - If no failure: agent completes task → high reward (+50)
#   - If failure:    silent failure detected late → moderate penalty (-20)
#
# "Retry_Step" (soft intervention):
#   - If no failure: slight wasted compute → small reward (+15)
#   - If failure:    retrying at least avoids silent failure → small penalty (-5)
#
# "Escalate_to_Human" (interrupt):
#   - Cost of human time is constant regardless of outcome → (-2 / -2)
#   - This action always costs something but caps downside loss.
#
# "Halt" (stop everything):
#   - If no failure: unnecessary halt → zero (wasted run but no damage) (0)
#   - If failure:    correct halt → small reward (+5, prevented bad outcome)
# ─────────────────────────────────────────────────────────────

UTILITY_TABLE = {
    "Continue":           {"No": 50, "Yes": -20},
    "Retry_Step":         {"No": 15, "Yes":  -5},
    "Escalate_to_Human":  {"No": -2, "Yes":  -2},
    "Halt":               {"No":  0, "Yes":   5},
}

# Human-readable labels for the dashboard
ACTION_LABELS = {
    "Continue":           "Continue",
    "Retry_Step":         "Retry Step",
    "Escalate_to_Human":  "Escalate to Human",
    "Halt":               "Halt",
}


class MockDecisionNetwork:
    def __init__(self, inference_engine: FailNetInference):
        self.inference = inference_engine
        self.actions = list(UTILITY_TABLE.keys())
        self.utility = UTILITY_TABLE

    def _compute_eu(self, prob_fail: float) -> dict:
        """Compute Expected Utility for all actions given P(Task_Failure=Yes)."""
        prob_no_fail = 1.0 - prob_fail
        return {
            action: (
                self.utility[action]["Yes"] * prob_fail
                + self.utility[action]["No"]  * prob_no_fail
            )
            for action in self.actions
        }

    def get_optimal_policy(self, evidence: dict):
        """Returns (best_action, expected_utilities_dict) given evidence."""
        res = self.inference.posterior_failure(evidence)
        prob_fail = res.values[res.state_names["Task_Failure"].index("Yes")]
        prob_fail = float(prob_fail)

        expected_utilities = self._compute_eu(prob_fail)
        best_action = max(expected_utilities, key=expected_utilities.get)
        return best_action, expected_utilities

    def recommend_active_intervention(self, evidence: dict):
        """Returns optimal action, expected utilities, and best VoI recommendation."""
        best_action, expected_utilities = self.get_optimal_policy(evidence)

        from src.voi import compute_voi
        voi_results = compute_voi(self.inference, self, evidence)

        best_diagnostic = None
        best_voi_value = 0.0

        if voi_results:
            best_diagnostic = list(voi_results.keys())[0]
            best_voi_value = voi_results[best_diagnostic]

        return best_action, expected_utilities, best_diagnostic, best_voi_value

    # ─────────────────────────────────────────────────────────────────
    #  NEW: Decision Boundary Analysis
    # ─────────────────────────────────────────────────────────────────
    def get_decision_boundary(self, action_a: str, action_b: str) -> float:
        """
        Computes the failure probability threshold p* at which EU(action_a) = EU(action_b).
        
        EU_a(p) = U_a(Yes)*p + U_a(No)*(1-p)
        EU_b(p) = U_b(Yes)*p + U_b(No)*(1-p)
        
        Set EU_a = EU_b and solve for p:
          p = (U_b(No) - U_a(No)) / ((U_a(Yes) - U_b(Yes)) - (U_a(No) - U_b(No)))
        
        Returns:
            The threshold p* in [0, 1], or NaN if the lines are parallel (no crossing).
        """
        ua_yes, ua_no = self.utility[action_a]["Yes"], self.utility[action_a]["No"]
        ub_yes, ub_no = self.utility[action_b]["Yes"], self.utility[action_b]["No"]

        denominator = (ua_yes - ub_yes) - (ua_no - ub_no)
        if abs(denominator) < 1e-9:
            return float("nan")  # Parallel lines — no crossover
        
        p_star = (ub_no - ua_no) / denominator
        return round(float(np.clip(p_star, 0.0, 1.0)), 4)

    def decision_boundary_curve(self, save_path: str = "report/figures/decision_boundary.png"):
        """
        Plots Expected Utility for all 4 actions across the full range of P(Task_Failure=Yes).
        Marks all pairwise decision boundaries where the optimal action switches.
        
        This answers: "At what failure probability does Halt become better than Continue?"
        """
        prob_range = np.linspace(0, 1, 300)
        
        fig, ax = plt.subplots(figsize=(11, 6))
        
        colors = {
            "Continue":          "#27ae60",
            "Retry_Step":        "#f39c12",
            "Escalate_to_Human": "#8e44ad",
            "Halt":              "#e74c3c",
        }
        linestyles = {
            "Continue":          "-",
            "Retry_Step":        "--",
            "Escalate_to_Human": "-.",
            "Halt":              ":",
        }
        
        eu_curves = {}
        for action in self.actions:
            eu_vals = [
                self.utility[action]["Yes"] * p + self.utility[action]["No"] * (1 - p)
                for p in prob_range
            ]
            eu_curves[action] = eu_vals
            ax.plot(
                prob_range, eu_vals,
                label=ACTION_LABELS[action],
                color=colors[action],
                linewidth=2.5,
                linestyle=linestyles[action],
            )
        
        # Mark optimal action regions (shading)
        optimal_eu = np.array([max(eu_curves[a][i] for a in self.actions) for i in range(len(prob_range))])
        ax.plot(prob_range, optimal_eu, color="black", linewidth=1.2, alpha=0.5,
                label="Max EU envelope (optimal boundary)")
        
        # Mark crossover thresholds between key pairs
        key_pairs = [
            ("Continue", "Halt"),
            ("Continue", "Retry_Step"),
            ("Retry_Step", "Halt"),
            ("Escalate_to_Human", "Halt"),
        ]
        for a, b in key_pairs:
            p_star = self.get_decision_boundary(a, b)
            if 0 < p_star < 1:
                ax.axvline(p_star, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
                ax.text(p_star + 0.005, ax.get_ylim()[0] + 1.5,
                        f"p*={p_star:.2f}\n({a[:4]}↔{b[:4]})",
                        fontsize=7.5, color="gray", rotation=0)
        
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, alpha=0.5,
                   label="Default threshold (0.5)")
        ax.set_xlabel("P(Task_Failure = Yes)", fontsize=12)
        ax.set_ylabel("Expected Utility", fontsize=12)
        ax.set_title(
            "Decision Boundary Curve — Expected Utility per Action\n"
            "Vertical dashed lines show where the optimal action switches",
            fontsize=12, fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        
        fig.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Decision boundary curve saved to: {save_path}")
        return save_path


if __name__ == "__main__":
    model = learn_parameters()
    inference = FailNetInference(model)
    dn = MockDecisionNetwork(inference)

    # Test case 1: No evidence
    action, eus = dn.get_optimal_policy({})
    print("No evidence - Optimal Action:", action)
    print("Expected Utilities:", eus)

    # Test case 2: Execution error occurred
    action, eus = dn.get_optimal_policy({"Execution_Error_Occurred": "Yes"})
    print("\nExecution Error - Optimal Action:", action)
    print("Expected Utilities:", eus)

    # Test case 3: Decision boundaries
    print("\nDecision Boundaries (P* where EU(A) = EU(B)):")
    pairs = [("Continue", "Halt"), ("Continue", "Retry_Step"), ("Retry_Step", "Halt")]
    for a, b in pairs:
        p = dn.get_decision_boundary(a, b)
        print(f"  {a} ↔ {b}: p* = {p}")

    # Plot decision boundary curve
    dn.decision_boundary_curve()
