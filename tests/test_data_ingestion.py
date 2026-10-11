"""
tests/test_data_ingestion.py
-----------------------------
Unit tests for the data ingestion and labeling pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
from src.labeling import (
    derive_task_failure_label,
    derive_failure_severity_label,
    get_provenance_tag,
    encode_dataframe,
    get_label_distribution,
)
from src.feature_engineering import (
    extract_task_complexity,
    extract_trajectory_length,
    extract_context_window_load,
    extract_execution_error,
    extract_error_recovery_attempted,
    extract_planning_complexity,
)


class TestLabelingFunctions:

    def test_derive_failure_label_correct(self):
        assert derive_task_failure_label(True)  == "No"
        assert derive_task_failure_label(False) == "Yes"

    def test_derive_severity_label(self):
        assert derive_failure_severity_label("Yes") == "Critical"
        assert derive_failure_severity_label("No")  == "None"

    def test_provenance_valid_tags(self):
        assert get_provenance_tag("who_and_when_heuristic_parser") == "who_and_when_heuristic_parser"
        assert get_provenance_tag("synthetic_success") == "synthetic_success"
        assert get_provenance_tag("augmented") == "augmented"

    def test_provenance_invalid_tag_raises(self):
        with pytest.raises(ValueError):
            get_provenance_tag("unknown_source")


class TestFeatureEngineeringFunctions:

    def test_task_complexity_mapping(self):
        assert extract_task_complexity({"level": "Hard"})   == "High"
        assert extract_task_complexity({"level": "Easy"})   == "Low"
        assert extract_task_complexity({"level": "Medium"}) == "Medium"
        assert extract_task_complexity({})                  == "Medium"  # default

    def test_trajectory_length_thresholds(self):
        assert extract_trajectory_length([{}] * 3)  == "Short"
        assert extract_trajectory_length([{}] * 10) == "Medium"
        assert extract_trajectory_length([{}] * 20) == "Long"

    def test_context_window_load(self):
        short_history = [{"content": "hello"}]  # < 2000 chars
        assert extract_context_window_load(short_history) == "Low"
        
        long_content = "x" * 10000
        long_history = [{"content": long_content}]
        assert extract_context_window_load(long_history) == "High"

    def test_execution_error_detected(self):
        history_with_error = [
            {"role": "user",      "content": "run the script"},
            {"role": "assistant", "content": "running..."},
            {"role": "user",      "content": "exitcode: 1\nTraceback (most recent call last)..."},
        ]
        assert extract_execution_error(history_with_error) == "Yes"

    def test_no_execution_error(self):
        clean_history = [
            {"role": "user",      "content": "run the script"},
            {"role": "assistant", "content": "The script completed successfully."},
            {"role": "user",      "content": "exitcode: 0"},
        ]
        assert extract_execution_error(clean_history) == "No"

    def test_error_recovery_requires_error(self):
        """Without an error, recovery should be 'No' regardless."""
        assert extract_error_recovery_attempted("No", []) == "No"

    def test_planning_complexity_derivation(self):
        assert extract_planning_complexity("Low",  "Few")     == "Simple"
        assert extract_planning_complexity("High",  "Many")   == "Complex"
        assert extract_planning_complexity("Medium", "Moderate") == "Moderate"


class TestProcessedDataset:
    """Integration tests that validate the processed CSV on disk."""

    PROCESSED_PATH = "data/processed/bn_evidence_table.csv"

    @pytest.fixture(scope="class")
    def df(self):
        if not os.path.exists(self.PROCESSED_PATH):
            pytest.skip(f"Processed dataset not found at '{self.PROCESSED_PATH}'. Run run_pipeline.py first.")
        return pd.read_csv(self.PROCESSED_PATH)

    def test_no_null_values(self, df):
        # Columns that legitimately store 'N/A' as a string category
        # (e.g., Error_Recovery_Success is 'N/A' when no error occurred).
        # Pandas reads 'N/A' back from CSV as NaN, so we exclude those columns.
        na_string_cols = {
            "Error_Recovery_Success",
            "Inter_Agent_Communication_Quality",
            "Role_Assignment_Correctness",
            "Failure_Severity",   # can be 'None' string but not NaN
        }
        check_cols = [c for c in df.columns if c not in na_string_cols]
        null_counts = df[check_cols].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        assert len(cols_with_nulls) == 0, \
            f"Found unexpected null values in columns: {cols_with_nulls.to_dict()}"

    def test_required_columns_present(self, df):
        required = [
            "Task_Complexity", "Task_Failure", "Execution_Error_Occurred",
            "Plan_Quality", "Hallucination_Risk", "Failure_Severity", "label_provenance"
        ]
        missing = [c for c in required if c not in df.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_task_failure_only_binary(self, df):
        valid_values = {"Yes", "No"}
        actual_values = set(df["Task_Failure"].unique())
        assert actual_values == valid_values, \
            f"Task_Failure has unexpected values: {actual_values}"

    def test_class_balance(self, df):
        """After augmentation, dataset should be near 50/50 (within 10% tolerance)."""
        dist = get_label_distribution(df, "Task_Failure")
        for label, pct in dist.items():
            assert 0.40 <= pct <= 0.60, \
                f"Class '{label}' has {pct*100:.1f}% — dataset is more than 10% imbalanced."

    def test_provenance_column_valid(self, df):
        valid_sources = {"who_and_when_heuristic_parser", "synthetic_success", "augmented"}
        actual = set(df["label_provenance"].unique())
        assert actual.issubset(valid_sources), \
            f"Unexpected provenance values: {actual - valid_sources}"

    def test_encode_dataframe_shape(self, df):
        X, y = encode_dataframe(df)
        assert len(X) == len(df), "Encoded X must have same row count as original df."
        assert len(y) == len(df), "Encoded y must have same row count as original df."
        assert set(y.unique()).issubset({0, 1}), "y must be binary (0/1)."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
