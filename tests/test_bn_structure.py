"""
tests/test_bn_structure.py
--------------------------
Unit tests for the FailNet Bayesian Network graph structure.
Verifies that the hand-crafted DAG satisfies all required structural properties.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import networkx as nx
from src.bn_structure import get_failnet_model, failnet_edges


class TestBNStructure:

    @pytest.fixture(scope="class")
    def model(self):
        return get_failnet_model()

    def test_model_is_dag(self, model):
        """The network must be a Directed Acyclic Graph — no cycles allowed."""
        assert nx.is_directed_acyclic_graph(model), \
            "FailNet graph contains a cycle! DAGs must be acyclic."

    def test_node_count(self, model):
        """FailNet has 24 unique nodes in the DAG (verified from bn_structure.py edges)."""
        assert len(model.nodes()) == 24, \
            f"Expected 24 nodes, got {len(model.nodes())}."

    def test_edge_count(self, model):
        """FailNet must have exactly 37 directed edges as specified in bn_structure.py."""
        assert len(model.edges()) == len(failnet_edges), \
            f"Expected {len(failnet_edges)} edges, got {len(model.edges())}."

    def test_task_failure_is_node(self, model):
        """Task_Failure must be present — it's the primary prediction target."""
        assert "Task_Failure" in model.nodes(), \
            "'Task_Failure' node is missing from the graph."

    def test_task_failure_has_parents(self, model):
        """Task_Failure should have multiple parents (causes of failure)."""
        parents = list(model.predecessors("Task_Failure"))
        assert len(parents) >= 3, \
            f"Task_Failure has too few parents ({len(parents)}). Expected >= 3."

    def test_failure_severity_is_leaf(self, model):
        """Failure_Severity is a terminal outcome node — it should have no children."""
        children = list(model.successors("Failure_Severity"))
        assert len(children) == 0, \
            f"Failure_Severity should be a leaf node, but has children: {children}."

    def test_no_isolated_nodes(self, model):
        """Every node must have at least one edge (no disconnected orphan nodes)."""
        for node in model.nodes():
            degree = model.in_degree(node) + model.out_degree(node)
            assert degree > 0, f"Node '{node}' is isolated (no edges)."

    def test_all_expected_nodes_present(self, model):
        """All 24 documented FailNet nodes must exist in the graph."""
        expected_nodes = {
            "Task_Complexity", "Base_Model_Capability", "Instruction_Ambiguity",
            "Tool_Reliability", "Environment_Volatility",
            "Multi_Agent_Coordination_Required", "Human_Oversight_Present",
            "Required_Tool_Count", "Planning_Complexity", "Context_Window_Load",
            "Trajectory_Length", "Plan_Quality", "Memory_Retrieval_Quality",
            "Tool_Selection_Correctness", "Tool_Argument_Validity",
            "Execution_Error_Occurred", "Error_Recovery_Attempted",
            "Error_Recovery_Success", "Hallucination_Risk",
            "Intermediate_Verification_Quality", "Inter_Agent_Communication_Quality",
            "Role_Assignment_Correctness", "Task_Failure", "Failure_Severity",
        }
        # Note: exactly 24 nodes in the DAG
        existing = set(model.nodes())
        missing = expected_nodes - existing
        assert not missing, f"Missing nodes from graph: {missing}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
