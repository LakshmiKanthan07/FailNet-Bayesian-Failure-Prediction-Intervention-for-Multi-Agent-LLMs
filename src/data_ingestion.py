import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_samples=2000, output_path="data/processed/bn_evidence_table.csv"):
    """
    Generates synthetic trajectory data for the FailNet BN based on the 25-node structure.
    This unblocks Phase 3-8 until the real Who&When / AEGIS parsing pipeline is fully implemented.
    """
    np.random.seed(42)

    data = {
        "Task_Complexity": np.random.choice(["Low", "Medium", "High"], p=[0.3, 0.5, 0.2], size=num_samples),
        "Base_Model_Capability": np.random.choice(["Low", "High"], p=[0.2, 0.8], size=num_samples),
        "Instruction_Ambiguity": np.random.choice(["Low", "High"], p=[0.7, 0.3], size=num_samples),
        "Tool_Reliability": np.random.choice(["Low", "High"], p=[0.1, 0.9], size=num_samples),
        "Environment_Volatility": np.random.choice(["Low", "High"], p=[0.8, 0.2], size=num_samples),
        "Multi_Agent_Coordination_Required": np.random.choice(["Yes", "No"], p=[0.4, 0.6], size=num_samples),
        "Human_Oversight_Present": np.random.choice(["Yes", "No"], p=[0.2, 0.8], size=num_samples),
    }

    df = pd.DataFrame(data)

    df["Required_Tool_Count"] = df["Task_Complexity"].map({"Low": "Few", "Medium": "Moderate", "High": "Many"})
    
    # Planning_Complexity depends on Task_Complexity and Required_Tool_Count
    df["Planning_Complexity"] = df["Task_Complexity"].map({"Low": "Simple", "Medium": "Moderate", "High": "Complex"})

    # Context_Window_Load
    df["Context_Window_Load"] = df["Task_Complexity"].map({"Low": "Low", "Medium": "Medium", "High": "High"})

    # Trajectory_Length
    df["Trajectory_Length"] = df["Task_Complexity"].map({"Low": "Short", "Medium": "Medium", "High": "Long"})

    # Plan_Quality
    df["Plan_Quality"] = np.where(
        (df["Base_Model_Capability"] == "High") & (df["Planning_Complexity"] == "Simple") & (df["Instruction_Ambiguity"] == "Low"),
        "Good", "Flawed"
    )
    # Add some noise
    df["Plan_Quality"] = np.where(np.random.rand(num_samples) < 0.1, "Good", df["Plan_Quality"])
    df["Plan_Quality"] = np.where(np.random.rand(num_samples) < 0.1, "Flawed", df["Plan_Quality"])

    # Memory_Retrieval_Quality
    df["Memory_Retrieval_Quality"] = np.where(df["Context_Window_Load"] == "High", "Poor", "Good")

    # Tool_Selection_Correctness
    df["Tool_Selection_Correctness"] = np.where(
        (df["Plan_Quality"] == "Good") & (df["Base_Model_Capability"] == "High"),
        "Correct", "Incorrect"
    )
    # Add some noise
    df.loc[np.random.rand(num_samples) < 0.15, "Tool_Selection_Correctness"] = "Incorrect"

    # Tool_Argument_Validity
    df["Tool_Argument_Validity"] = np.where(
        (df["Tool_Selection_Correctness"] == "Correct") & (df["Base_Model_Capability"] == "High"),
        "Valid", "Invalid"
    )

    # Execution_Error_Occurred
    df["Execution_Error_Occurred"] = np.where(
        (df["Tool_Argument_Validity"] == "Invalid") | (df["Tool_Reliability"] == "Low"),
        "Yes", "No"
    )
    df.loc[np.random.rand(num_samples) < 0.05, "Execution_Error_Occurred"] = "Yes"

    # Error_Recovery_Attempted
    df["Error_Recovery_Attempted"] = np.where(df["Execution_Error_Occurred"] == "Yes", "Yes", "No")

    # Error_Recovery_Success
    df["Error_Recovery_Success"] = np.where(
        (df["Error_Recovery_Attempted"] == "Yes") & (df["Base_Model_Capability"] == "High") & (df["Environment_Volatility"] == "Low"),
        "Yes", "No"
    )
    df.loc[df["Error_Recovery_Attempted"] == "No", "Error_Recovery_Success"] = "N/A"

    # Hallucination_Risk
    df["Hallucination_Risk"] = np.where(
        (df["Memory_Retrieval_Quality"] == "Poor") | (df["Base_Model_Capability"] == "Low"),
        "High", "Low"
    )

    # Intermediate_Verification_Quality
    df["Intermediate_Verification_Quality"] = np.where(
        (df["Plan_Quality"] == "Good") & (df["Base_Model_Capability"] == "High"),
        "High", "Low"
    )

    # Inter_Agent_Communication_Quality
    df["Inter_Agent_Communication_Quality"] = np.where(
        (df["Multi_Agent_Coordination_Required"] == "Yes") & (df["Base_Model_Capability"] == "High"),
        "High", "Low"
    )
    df.loc[df["Multi_Agent_Coordination_Required"] == "No", "Inter_Agent_Communication_Quality"] = "N/A"

    # Role_Assignment_Correctness
    df["Role_Assignment_Correctness"] = np.where(
        (df["Multi_Agent_Coordination_Required"] == "Yes") & (df["Plan_Quality"] == "Good"),
        "Correct", "Incorrect"
    )
    df.loc[df["Multi_Agent_Coordination_Required"] == "No", "Role_Assignment_Correctness"] = "N/A"

    # Task_Failure
    fail_prob = np.zeros(num_samples)
    fail_prob += np.where(df["Execution_Error_Occurred"] == "Yes", 0.3, 0)
    fail_prob -= np.where(df["Error_Recovery_Success"] == "Yes", 0.3, 0)
    fail_prob += np.where(df["Hallucination_Risk"] == "High", 0.4, 0)
    fail_prob += np.where(df["Intermediate_Verification_Quality"] == "Low", 0.2, 0)
    fail_prob += np.where(df["Inter_Agent_Communication_Quality"] == "Low", 0.2, 0)
    fail_prob += np.where(df["Role_Assignment_Correctness"] == "Incorrect", 0.2, 0)
    
    # Clip prob to [0.05, 0.95]
    fail_prob = np.clip(fail_prob, 0.05, 0.95)
    
    # Sample Task_Failure
    rands = np.random.rand(num_samples)
    df["Task_Failure"] = np.where(rands < fail_prob, "Yes", "No")

    # Failure_Severity
    df["Failure_Severity"] = np.where(
        df["Task_Failure"] == "Yes",
        np.random.choice(["Minor", "Moderate", "Critical"], p=[0.4, 0.4, 0.2], size=num_samples),
        "None"
    )
    # Severity mitigated by Human Oversight
    df["Failure_Severity"] = np.where(
        (df["Failure_Severity"] == "Critical") & (df["Human_Oversight_Present"] == "Yes"),
        "Moderate", df["Failure_Severity"]
    )
    
    df["label_provenance"] = "heuristic_dummy_generator"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {num_samples} synthetic samples at {output_path}")

if __name__ == "__main__":
    generate_synthetic_data()
