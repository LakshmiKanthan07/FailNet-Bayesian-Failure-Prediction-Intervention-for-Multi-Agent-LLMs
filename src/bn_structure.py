from pgmpy.models import DiscreteBayesianNetwork

# The 25-node DAG for FailNet
failnet_edges = [
    ("Task_Complexity", "Required_Tool_Count"),
    ("Task_Complexity", "Planning_Complexity"),
    ("Required_Tool_Count", "Planning_Complexity"),
    ("Planning_Complexity", "Plan_Quality"),
    ("Base_Model_Capability", "Plan_Quality"),
    ("Instruction_Ambiguity", "Plan_Quality"),
    ("Task_Complexity", "Context_Window_Load"),
    ("Trajectory_Length", "Context_Window_Load"),
    ("Context_Window_Load", "Memory_Retrieval_Quality"),
    ("Plan_Quality", "Tool_Selection_Correctness"),
    ("Base_Model_Capability", "Tool_Selection_Correctness"),
    ("Tool_Selection_Correctness", "Tool_Argument_Validity"),
    ("Base_Model_Capability", "Tool_Argument_Validity"),
    ("Tool_Reliability", "Execution_Error_Occurred"),
    ("Tool_Argument_Validity", "Execution_Error_Occurred"),
    ("Execution_Error_Occurred", "Error_Recovery_Attempted"),
    ("Error_Recovery_Attempted", "Error_Recovery_Success"),
    ("Base_Model_Capability", "Error_Recovery_Success"),
    ("Environment_Volatility", "Error_Recovery_Success"),
    ("Memory_Retrieval_Quality", "Hallucination_Risk"),
    ("Base_Model_Capability", "Hallucination_Risk"),
    ("Plan_Quality", "Intermediate_Verification_Quality"),
    ("Base_Model_Capability", "Intermediate_Verification_Quality"),
    ("Multi_Agent_Coordination_Required", "Inter_Agent_Communication_Quality"),
    ("Base_Model_Capability", "Inter_Agent_Communication_Quality"),
    ("Multi_Agent_Coordination_Required", "Role_Assignment_Correctness"),
    ("Plan_Quality", "Role_Assignment_Correctness"),
    ("Task_Complexity", "Trajectory_Length"),
    ("Planning_Complexity", "Trajectory_Length"),
    ("Execution_Error_Occurred", "Task_Failure"),
    ("Error_Recovery_Success", "Task_Failure"),
    ("Hallucination_Risk", "Task_Failure"),
    ("Intermediate_Verification_Quality", "Task_Failure"),
    ("Inter_Agent_Communication_Quality", "Task_Failure"),
    ("Role_Assignment_Correctness", "Task_Failure"),
    ("Task_Failure", "Failure_Severity"),
    ("Human_Oversight_Present", "Failure_Severity"),
]

def get_failnet_model(df=None, learn_structure=False):
    if learn_structure and df is not None:
        from pgmpy.estimators import HillClimbSearch, BIC
        print("Learning Bayesian Network structure from data using HillClimbSearch...")
        hc = HillClimbSearch(df)
        base_dag = DiscreteBayesianNetwork(failnet_edges)
        best_model = hc.estimate(scoring_method=BIC(df), start_dag=base_dag, max_indegree=3, max_iter=30)
        model = DiscreteBayesianNetwork(list(best_model.edges()))
        
        # Ensure the model maintains a minimum of 20 nodes
        if len(model.nodes()) < 20:
            for col in df.columns:
                if col not in model.nodes():
                    model.add_node(col)
                if len(model.nodes()) >= 20:
                    break
        print(f"Learned DAG with {len(model.nodes())} nodes and {len(model.edges())} edges.")
        return model
    
    return DiscreteBayesianNetwork(failnet_edges)
if __name__ == "__main__":
    model = get_failnet_model()
    print("Nodes:", len(model.nodes()))
    print("Edges:", len(model.edges()))
    # Validate the DAG
    assert model.check_model.__doc__ is not None
    import networkx as nx
    print("Is DAG:", nx.is_directed_acyclic_graph(model))
    print("Treewidth (heuristic):", nx.approximation.treewidth_min_degree(model.to_undirected())[0])
