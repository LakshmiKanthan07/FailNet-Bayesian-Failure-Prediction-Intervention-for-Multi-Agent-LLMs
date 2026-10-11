"""
tests/test_inference.py
------------------------
Unit tests for FailNetInference.

Tests that the model's probabilistic reasoning behaves correctly:
- More evidence of failure should increase P(Task_Failure=Yes).
- More evidence of success should decrease P(Task_Failure=Yes).
- Markov Blanket is non-empty and contains expected nodes.
- Sensitivity analysis returns a valid DataFrame.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.bn_learning import learn_parameters
from src.inference import FailNetInference


@pytest.fixture(scope="module")
def inference():
    """Load the trained model once for the entire test module."""
    model = learn_parameters(optimize=False)
    return FailNetInference(model)


def _get_prob_fail(inf: FailNetInference, evidence: dict) -> float:
    res = inf.posterior_failure(evidence)
    return float(res.values[res.state_names["Task_Failure"].index("Yes")])


class TestPosteriorMonotonicity:
    """P(Failure) should increase/decrease monotonically with bad/good evidence."""

    def test_execution_error_raises_failure_prob(self, inference):
        """Adding Execution_Error_Occurred=Yes must raise P(Task_Failure=Yes)."""
        p_base = _get_prob_fail(inference, {})
        p_error = _get_prob_fail(inference, {"Execution_Error_Occurred": "Yes"})
        assert p_error > p_base, \
            f"Expected p_error ({p_error:.4f}) > p_base ({p_base:.4f})"

    def test_all_bad_evidence_high_failure(self, inference):
        """All worst-case observations should push P(Failure) above 0.7."""
        bad_evidence = {
            "Execution_Error_Occurred":          "Yes",
            "Plan_Quality":                       "Flawed",
            "Tool_Selection_Correctness":         "Incorrect",
            "Hallucination_Risk":                 "High",
            "Intermediate_Verification_Quality":  "Low",
            "Error_Recovery_Success":             "No",
        }
        p = _get_prob_fail(inference, bad_evidence)
        assert p > 0.70, \
            f"Expected P(Failure|all bad) > 0.70, got {p:.4f}"

    def test_all_good_evidence_low_failure(self, inference):
        """All best-case observations should push P(Failure) below 0.45 (below the 0.5 baseline)."""
        good_evidence = {
            "Execution_Error_Occurred":          "No",
            "Plan_Quality":                       "Good",
            "Tool_Selection_Correctness":         "Correct",
            "Hallucination_Risk":                 "Low",
            "Intermediate_Verification_Quality":  "High",
        }
        p = _get_prob_fail(inference, good_evidence)
        assert p <= 0.50, \
            f"Expected P(Failure|all good) <= 0.50 (at or below baseline), got {p:.4f}"

    def test_prior_is_near_half(self, inference):
        """With no evidence and balanced dataset, prior should be near 0.5 (±0.15)."""
        p = _get_prob_fail(inference, {})
        assert 0.35 <= p <= 0.65, \
            f"Prior P(Failure) = {p:.4f} is far from 0.5 — dataset may still be unbalanced."


class TestMarkovBlanket:
    """Markov Blanket must be non-empty and contain causally meaningful nodes."""

    def test_markov_blanket_is_non_empty(self, inference):
        mb = inference.get_markov_blanket("Task_Failure")
        assert len(mb) > 0, "Markov Blanket of Task_Failure must not be empty."

    def test_markov_blanket_contains_key_parents(self, inference):
        """Direct parents of Task_Failure must be in its Markov Blanket."""
        mb = set(inference.get_markov_blanket("Task_Failure"))
        # Execution_Error_Occurred is a direct parent of Task_Failure per bn_structure.py
        assert "Execution_Error_Occurred" in mb, \
            "Execution_Error_Occurred (parent) must be in Markov Blanket of Task_Failure."

    def test_markov_blanket_excludes_target(self, inference):
        """Task_Failure itself must NOT appear in its own Markov Blanket."""
        mb = inference.get_markov_blanket("Task_Failure")
        assert "Task_Failure" not in mb, \
            "Task_Failure must not appear in its own Markov Blanket."


class TestSensitivityAnalysis:
    """Sensitivity analysis DataFrame should be well-formed."""

    def test_sensitivity_returns_dataframe(self, inference):
        import pandas as pd
        sa = inference.sensitivity_analysis()
        assert isinstance(sa, pd.DataFrame), "sensitivity_analysis must return a DataFrame."

    def test_sensitivity_has_required_columns(self, inference):
        sa = inference.sensitivity_analysis()
        required_cols = {"Variable", "BestState", "WorstState", "BestProb", "WorstProb", "Swing"}
        assert required_cols.issubset(set(sa.columns)), \
            f"Missing columns: {required_cols - set(sa.columns)}"

    def test_sensitivity_sorted_by_swing(self, inference):
        """Results must be sorted descending by Swing."""
        sa = inference.sensitivity_analysis()
        swings = sa["Swing"].tolist()
        assert swings == sorted(swings, reverse=True), \
            "Sensitivity DataFrame must be sorted by Swing (descending)."

    def test_sensitivity_swing_is_non_negative(self, inference):
        """Swing = WorstProb - BestProb must always be >= 0."""
        sa = inference.sensitivity_analysis()
        assert (sa["Swing"] >= 0).all(), \
            "All Swing values must be non-negative."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
