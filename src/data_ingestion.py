"""
data_ingestion.py
-----------------
Orchestrates the full data pipeline:
  1. Parse raw JSON trajectories from the Who&When dataset.
  2. Extract BN features via src/feature_engineering.py.
  3. Derive labels via src/labeling.py.
  4. Balance the dataset by augmenting synthetic success rows with realistic variation.
  5. Save the processed evidence table to data/processed/.
"""

import os
import json
import glob
import random
import pandas as pd
import numpy as np

from src.feature_engineering import (
    extract_task_complexity,
    extract_trajectory_length,
    extract_context_window_load,
    extract_required_tool_count,
    extract_multi_agent,
    extract_planning_complexity,
    extract_plan_quality,
    extract_memory_retrieval_quality,
    extract_execution_error,
    extract_error_recovery_attempted,
    extract_error_recovery_success,
    extract_tool_selection,
    extract_tool_argument_validity,
    extract_hallucination_risk,
    extract_intermediate_verification,
    extract_inter_agent_comm,
    extract_role_assignment,
    extract_failure_severity,
)
from src.labeling import (
    derive_task_failure_label,
    get_provenance_tag,
    summarize_labels,
)

# ─────────────────────────────────────────────────────────────────
#  Discrete value domains for each node (used in augmentation)
# ─────────────────────────────────────────────────────────────────
NODE_DOMAINS = {
    "Task_Complexity":                  ["Low", "Medium", "High"],
    "Base_Model_Capability":            ["High", "Low"],
    "Instruction_Ambiguity":            ["Low", "High"],
    "Tool_Reliability":                 ["High", "Low"],
    "Environment_Volatility":           ["Low", "High"],
    "Multi_Agent_Coordination_Required":["No", "Yes"],
    "Human_Oversight_Present":          ["No", "Yes"],
    "Required_Tool_Count":              ["Few", "Moderate", "Many"],
    "Planning_Complexity":              ["Simple", "Moderate", "Complex"],
    "Context_Window_Load":              ["Low", "Medium", "High"],
    "Trajectory_Length":                ["Short", "Medium", "Long"],
}

# Realistic success-case weights per domain value
# (biased toward "easy" conditions that lead to success)
SUCCESS_WEIGHTS = {
    "Task_Complexity":                  [0.40, 0.45, 0.15],   # mostly Low/Medium
    "Base_Model_Capability":            [0.85, 0.15],          # mostly High
    "Instruction_Ambiguity":            [0.80, 0.20],          # mostly Low
    "Tool_Reliability":                 [0.88, 0.12],          # mostly High
    "Environment_Volatility":           [0.82, 0.18],          # mostly Low
    "Multi_Agent_Coordination_Required":["No", "Yes"],         # placeholder, see below
    "Human_Oversight_Present":          [0.88, 0.12],          # mostly No (self-directed)
    "Required_Tool_Count":              [0.55, 0.35, 0.10],    # fewer tools
    "Planning_Complexity":              [0.50, 0.38, 0.12],    # simpler plans
    "Context_Window_Load":              [0.55, 0.35, 0.10],    # lower load
    "Trajectory_Length":                [0.50, 0.40, 0.10],    # shorter trajectories
}


def _generate_synthetic_success_row(rng: np.random.Generator) -> dict:
    """
    Generates a single synthetic success row with realistic variation.
    All fields are sampled from domain-conditioned distributions so the row
    is plausible for a successful agent run, not a copy-paste clone.
    """
    task_comp = rng.choice(
        NODE_DOMAINS["Task_Complexity"],
        p=SUCCESS_WEIGHTS["Task_Complexity"]
    )
    base_model = rng.choice(
        NODE_DOMAINS["Base_Model_Capability"],
        p=SUCCESS_WEIGHTS["Base_Model_Capability"]
    )
    inst_ambig = rng.choice(
        NODE_DOMAINS["Instruction_Ambiguity"],
        p=SUCCESS_WEIGHTS["Instruction_Ambiguity"]
    )
    tool_rel = rng.choice(
        NODE_DOMAINS["Tool_Reliability"],
        p=SUCCESS_WEIGHTS["Tool_Reliability"]
    )
    env_vol = rng.choice(
        NODE_DOMAINS["Environment_Volatility"],
        p=SUCCESS_WEIGHTS["Environment_Volatility"]
    )
    multi_agent = rng.choice(["No", "Yes"], p=[0.75, 0.25])
    human_over = rng.choice(["No", "Yes"], p=SUCCESS_WEIGHTS["Human_Oversight_Present"])
    req_tools = rng.choice(
        NODE_DOMAINS["Required_Tool_Count"],
        p=SUCCESS_WEIGHTS["Required_Tool_Count"]
    )
    cw_load = rng.choice(
        NODE_DOMAINS["Context_Window_Load"],
        p=SUCCESS_WEIGHTS["Context_Window_Load"]
    )
    traj_len = rng.choice(
        NODE_DOMAINS["Trajectory_Length"],
        p=SUCCESS_WEIGHTS["Trajectory_Length"]
    )

    # Derived nodes (conditioned on sampled root values)
    plan_comp = extract_planning_complexity(task_comp, req_tools)
    plan_qual = "Good"                          # it's a success row
    mem_qual  = "Poor" if cw_load == "High" else "Good"
    tool_sel  = "Correct"                       # success rows succeed at tool use
    tool_arg  = "Valid"
    exec_err  = rng.choice(["No", "Yes"], p=[0.90, 0.10])  # rare errors that were recovered
    rec_att   = "Yes" if exec_err == "Yes" else "No"
    rec_succ  = "Yes" if rec_att == "Yes" else "N/A"
    halluc    = "High" if (cw_load == "High" and base_model == "Low") else "Low"
    intermed  = "High"                          # good intermediate verification
    inter_comm = "High" if multi_agent == "Yes" else "N/A"
    role_assign = "Correct" if multi_agent == "Yes" else "N/A"

    return {
        "Task_Complexity":                   task_comp,
        "Base_Model_Capability":             base_model,
        "Instruction_Ambiguity":             inst_ambig,
        "Tool_Reliability":                  tool_rel,
        "Environment_Volatility":            env_vol,
        "Multi_Agent_Coordination_Required": multi_agent,
        "Human_Oversight_Present":           human_over,
        "Required_Tool_Count":               req_tools,
        "Planning_Complexity":               plan_comp,
        "Context_Window_Load":               cw_load,
        "Trajectory_Length":                 traj_len,
        "Plan_Quality":                      plan_qual,
        "Memory_Retrieval_Quality":          mem_qual,
        "Tool_Selection_Correctness":        tool_sel,
        "Tool_Argument_Validity":            tool_arg,
        "Execution_Error_Occurred":          exec_err,
        "Error_Recovery_Attempted":          rec_att,
        "Error_Recovery_Success":            rec_succ,
        "Hallucination_Risk":                halluc,
        "Intermediate_Verification_Quality": intermed,
        "Inter_Agent_Communication_Quality": inter_comm,
        "Role_Assignment_Correctness":       role_assign,
        "Task_Failure":                      "No",
        "Failure_Severity":                  "None",
        "label_provenance":                  get_provenance_tag("synthetic_success"),
    }

def ingest_original_data(
    data_dir: str = "data/original data/Who&When",
    output_path: str = "data/processed/bn_evidence_table.csv",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Main ingestion function.

    Steps:
        1. Glob all JSON files in data_dir recursively.
        2. Extract 24 BN features per trajectory using feature_engineering.py and labeling.py.
        3. Derive binary Task_Failure labels using labeling.py.
        4. Generate realistic synthetic success rows to balance class distribution.
        5. Save and return the final processed DataFrame.

    Args:
        data_dir:    Path to the Who&When dataset root.
        output_path: Destination CSV path.
        seed:        Random seed for synthetic row generation.

    Returns:
        Processed DataFrame with 24 BN feature columns + label_provenance.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    all_json_files = glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True)
    rows = []

    print(f"Found {len(all_json_files)} JSON trajectory files in '{data_dir}'.")

    for filepath in all_json_files:
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        history = data.get("history", [])
        is_correct = data.get("is_correct", False)
        mistake_reason = data.get("mistake_reason", "").lower()

        # ── Tier 1: Observable features ──────────────────────────────
        task_comp  = extract_task_complexity(data)
        traj_len   = extract_trajectory_length(history)
        cw_load    = extract_context_window_load(history)
        req_tools  = extract_required_tool_count(history)
        multi_agent = extract_multi_agent(data)
        task_fail  = derive_task_failure_label(is_correct)

        # ── Tier 2: Derived planning features ────────────────────────
        plan_comp  = extract_planning_complexity(task_comp, req_tools)
        plan_qual  = extract_plan_quality(history, mistake_reason)
        mem_qual   = extract_memory_retrieval_quality(cw_load)

        # ── Tier 3: Execution & error features ───────────────────────
        exec_err   = extract_execution_error(history)
        rec_att    = extract_error_recovery_attempted(exec_err, history)
        rec_succ   = extract_error_recovery_success(rec_att, is_correct)
        tool_sel   = extract_tool_selection(history, mistake_reason)
        tool_arg   = extract_tool_argument_validity(history, mistake_reason)

        # ── Tier 4: Proxy features (sampled from realistic priors) ───
        base_model = rng.choice(["High", "Low"], p=[0.80, 0.20])
        inst_ambig = rng.choice(["Low", "High"], p=[0.80, 0.20])
        tool_rel   = rng.choice(["High", "Low"], p=[0.90, 0.10])
        env_vol    = rng.choice(["Low", "High"], p=[0.85, 0.15])
        human_over = rng.choice(["No", "Yes"],   p=[0.90, 0.10])

        # ── Tier 5: Hallucination & verification ─────────────────────
        halluc     = extract_hallucination_risk(cw_load, base_model)
        intermed   = extract_intermediate_verification(history)

        # ── Tier 6: Multi-agent coordination ─────────────────────────
        inter_comm  = extract_inter_agent_comm(multi_agent, history)
        role_assign = extract_role_assignment(multi_agent, data)

        # ── Outcome ──────────────────────────────────────────────────
        fail_sev  = extract_failure_severity(task_fail)

        rows.append({
            "Task_Complexity":                   task_comp,
            "Base_Model_Capability":             base_model,
            "Instruction_Ambiguity":             inst_ambig,
            "Tool_Reliability":                  tool_rel,
            "Environment_Volatility":            env_vol,
            "Multi_Agent_Coordination_Required": multi_agent,
            "Human_Oversight_Present":           human_over,
            "Required_Tool_Count":               req_tools,
            "Planning_Complexity":               plan_comp,
            "Context_Window_Load":               cw_load,
            "Trajectory_Length":                 traj_len,
            "Plan_Quality":                      plan_qual,
            "Memory_Retrieval_Quality":          mem_qual,
            "Tool_Selection_Correctness":        tool_sel,
            "Tool_Argument_Validity":            tool_arg,
            "Execution_Error_Occurred":          exec_err,
            "Error_Recovery_Attempted":          rec_att,
            "Error_Recovery_Success":            rec_succ,
            "Hallucination_Risk":                halluc,
            "Intermediate_Verification_Quality": intermed,
            "Inter_Agent_Communication_Quality": inter_comm,
            "Role_Assignment_Correctness":       role_assign,
            "Task_Failure":                      task_fail,
            "Failure_Severity":                  fail_sev,
            "label_provenance":                  get_provenance_tag("who_and_when_heuristic_parser"),
        })

    # ── Balance classes with realistically varied synthetic successes ──
    n_failures = sum(1 for r in rows if r["Task_Failure"] == "Yes")
    n_success  = sum(1 for r in rows if r["Task_Failure"] == "No")
    n_to_add   = max(0, n_failures - n_success)

    if n_to_add > 0:
        print(f"Class imbalance detected: {n_failures} failures vs {n_success} successes.")
        print(f"Generating {n_to_add} synthetic success rows with realistic variation...")
        for _ in range(n_to_add):
            rows.append(_generate_synthetic_success_row(rng))
    else:
        print("Dataset is already balanced -- no augmentation needed.")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nIngested {len(rows)} total rows ({n_failures + n_to_add} after balancing) -- '{output_path}'")
    summarize_labels(df)

    return df


if __name__ == "__main__":
    ingest_original_data()
