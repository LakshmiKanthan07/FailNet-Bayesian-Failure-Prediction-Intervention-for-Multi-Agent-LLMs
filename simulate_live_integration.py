"""
simulate_live_integration.py
-----------------------------
FailNet Live Integration Demo.

This script simulates how FailNet would behave as a real-time probabilistic
supervisor attached to a live multi-agent AI system.

What it does
------------
1. Loads a single raw trajectory JSON file from the Who&When dataset (this is
   exactly what a live agent framework such as AutoGen / Magentic-One would
   emit turn-by-turn).
2. Streams it turn-by-turn, revealing turns one at a time -- exactly as they
   would arrive from a live agent.
3. After each new turn, it:
      a. Builds the current evidence set by calling the same feature_engineering
         functions that the training pipeline uses (so the model sees the same
         feature space it was trained on).
      b. Queries the FailNet Bayesian Network via Variable Elimination to get
         the current posterior P(Task_Failure = Yes | evidence).
      c. Queries the Decision Network to get the optimal action and its
         Expected Utility.
      d. Detects the Bayesian-optimal intervention threshold and prints a
         colour-coded live monitor line.
4. Prints a final post-mortem: the Markov Blanket of Task_Failure, the MAP
   causal diagnosis, and a comparison of what the Bayesian Network predicted
   vs. what actually happened.

Run from the project root:
    python simulate_live_integration.py
    python simulate_live_integration.py --file "data/original data/Who&When/Hand-Crafted/1.json"
    python simulate_live_integration.py --index 5
    python simulate_live_integration.py --list

Key design constraints
----------------------
- All probabilistic logic stays within the syllabus:
    Bayesian Networks, Exact Inference (Variable Elimination), Decision Networks,
    Expected Utility, Markov Blanket.
- No external LLM API calls are made -- we replay a real recorded trajectory.
"""

import os
import sys
import json
import glob
import time
import argparse
import textwrap

# ── path setup so imports work when run from project root ─────────────────────
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.bn_learning import learn_parameters
from src.inference import FailNetInference
from src.decision_network import MockDecisionNetwork
from src.feature_engineering import (
    extract_task_complexity,
    extract_trajectory_length,
    extract_context_window_load,
    extract_required_tool_count,
    extract_multi_agent,
    extract_planning_complexity,
    extract_memory_retrieval_quality,
    extract_execution_error,
    extract_error_recovery_attempted,
    extract_tool_selection,
    extract_tool_argument_validity,
    extract_hallucination_risk,
    extract_inter_agent_comm,
    extract_role_assignment,
)


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = os.path.join("data", "original data", "Who&When")
MODEL_DATA = os.path.join("data", "processed", "bn_evidence_table.csv")

# Decision Network thresholds -- derived analytically from the Utility Table.
# These are the P* values where EU(action_a) = EU(action_b).
THRESHOLD_RETRY    = 0.30   # below this  -> Continue
THRESHOLD_HALT     = 0.60   # above this  -> Halt
THRESHOLD_CRITICAL = 0.85   # circuit-breaker zone

BAR_WIDTH = 20


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: ASCII Progress Bar
# ─────────────────────────────────────────────────────────────────────────────

def make_bar(prob: float, width: int = BAR_WIDTH) -> str:
    filled = int(round(prob * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: Intervention label from probability
# ─────────────────────────────────────────────────────────────────────────────

def get_intervention(prob: float) -> tuple:
    """
    Maps P(Task_Failure) to the optimal action using Decision Network thresholds.
    Returns (action_str, risk_label).
    """
    if prob < THRESHOLD_RETRY:
        return "CONTINUE", "LOW"
    elif prob < THRESHOLD_HALT:
        return "RETRY STEP", "MEDIUM"
    elif prob < THRESHOLD_CRITICAL:
        return "ESCALATE TO HUMAN", "HIGH"
    else:
        return "HALT (Circuit Breaker)", "CRITICAL"


# ─────────────────────────────────────────────────────────────────────────────
#  CORE: Extract Evidence from a Partial Trajectory
# ─────────────────────────────────────────────────────────────────────────────

def extract_evidence_from_partial(data: dict, history_slice: list) -> dict:
    """
    Given the full trajectory metadata and a prefix of the conversation history,
    compute the current BN evidence dict.

    Uses the same feature extraction functions as data_ingestion.py so the
    model sees exactly the feature space it was trained on.

    Dynamic design: every feature that CAN change with new turns is
    recomputed on the growing history slice, so the BN belief shifts
    as evidence accumulates.
    """
    n_turns = len(history_slice)

    # ── Tier-1: observable from the live slice (all dynamic) ────────────────
    task_comp   = extract_task_complexity(data)
    traj_len    = extract_trajectory_length(history_slice)
    cw_load     = extract_context_window_load(history_slice)
    req_tools   = extract_required_tool_count(history_slice)
    multi_agent = extract_multi_agent(data)

    # ── Tier-2: derived planning features ───────────────────────────────────
    plan_comp = extract_planning_complexity(task_comp, req_tools)

    # ── Tier-3: execution / error features (dynamic per-turn) ───────────────
    exec_err = extract_execution_error(history_slice)
    rec_att  = extract_error_recovery_attempted(exec_err, history_slice)

    # ── Tier-3 extended: per-turn heuristic signals ──────────────────────────
    # Count distinct error occurrences in visible history
    error_patterns = ["exitcode: 1", "Traceback", "Error:", "Exception:", "stderr:", "FAILED"]
    n_errors = sum(
        1 for turn in history_slice
        for pat in error_patterns
        if pat in str(turn.get("content", ""))
    )

    # Detect looping / repetition: same role appearing 3+ times consecutively
    roles = [t.get("role", "") for t in history_slice]
    is_looping = any(
        roles[i] == roles[i+1] == roles[i+2]
        for i in range(len(roles) - 2)
    )

    # Stalled trajectory: many turns but still Short trajectory length signal
    stalled = (n_turns >= 8 and traj_len == "Short")

    # ── Tier-4: proxy variables -- adapt to runtime signals ─────────────────
    # Base_Model_Capability: downgrade if we see many errors
    base_model = "Low" if n_errors >= 2 else "High"

    # Instruction_Ambiguity: elevate if the agent is looping or stalling
    inst_ambig = "High" if (is_looping or stalled) else "Low"

    # Tool_Reliability: downgrade if execution errors are present
    tool_rel = "Low" if exec_err == "Yes" else "High"

    # Environment_Volatility: elevate if many tool calls failed
    env_vol = "High" if n_errors >= 3 else "Low"

    # Human_Oversight_Present: always No in automated benchmarks
    human_over = "No"

    # ── Tier-5: derived from updated observables ────────────────────────────
    halluc   = extract_hallucination_risk(cw_load, base_model)
    mem_qual = extract_memory_retrieval_quality(cw_load)

    # ── Tier-6: multi-agent coordination ────────────────────────────────────
    # Inter-agent comm: downgrade if we see role confusion or stalling
    if multi_agent == "No":
        inter_comm  = "N/A"
        role_assign = "N/A"
    else:
        inter_comm  = "Low" if (is_looping or n_errors >= 2) else "High"
        role_assign = extract_role_assignment(multi_agent, data)

    # ── Tool quality: degrade if errors observed ─────────────────────────────
    tool_sel = "Incorrect" if exec_err == "Yes" else "Correct"
    tool_arg = "Invalid"   if n_errors >= 2   else "Valid"

    evidence = {
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
        "Memory_Retrieval_Quality":          mem_qual,
        "Tool_Selection_Correctness":        tool_sel,
        "Tool_Argument_Validity":            tool_arg,
        "Execution_Error_Occurred":          exec_err,
        "Error_Recovery_Attempted":          rec_att,
        "Hallucination_Risk":                halluc,
    }

    # Multi-agent-only features: omit if N/A (single-agent trajectory)
    if multi_agent != "No":
        if inter_comm not in ("N/A",):
            evidence["Inter_Agent_Communication_Quality"] = inter_comm
        if role_assign not in ("N/A",):
            evidence["Role_Assignment_Correctness"] = role_assign

    return evidence


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: Sanitize string for Windows cp1252 terminal
# ─────────────────────────────────────────────────────────────────────────────

def safe_str(s: str, max_len: int = 80) -> str:
    """
    Strips characters that cannot be encoded in cp1252 (Windows terminal default).
    Replaces them with '?' and truncates to max_len.
    """
    cleaned = s.encode("cp1252", errors="replace").decode("cp1252")
    return cleaned[:max_len]


# ─────────────────────────────────────────────────────────────────────────────
#  CORE: Streaming Monitor
# ─────────────────────────────────────────────────────────────────────────────

def stream_trajectory(
    data: dict,
    inference: FailNetInference,
    dn: MockDecisionNetwork,
    delay: float = 0.0,
    verbose_evidence: bool = False,
) -> list:
    """
    Streams the trajectory turn-by-turn.

    At each turn, runs exact BN inference (Variable Elimination) on the
    current evidence and prints:
      - Turn number and speaker role
      - P(Task_Failure=Yes) with ASCII bar
      - Decision Network optimal action
      - Alert when the recommended action changes

    Returns the list of P(Fail) values across all turns.
    """
    history = data.get("history", [])
    is_correct = data.get("is_correct", True)
    ground_truth_failed = not is_correct
    task_label = safe_str(data.get("question", "Unknown task"), max_len=75)
    agents = [safe_str(str(a)) for a in data.get("system_prompt", {}).keys()]

    print("=" * 72)
    print("  FailNet  --  Live Bayesian Failure Monitor")
    print("  Streaming real multi-agent trajectory through the Bayesian Network")
    print("=" * 72)
    print(f"  Task        : {task_label}...")
    print(f"  Agents      : {agents if agents else ['Single Agent (no system_prompt dict)']}")
    print(f"  Total turns : {len(history)}")
    print(f"  Actual outcome: {'FAILURE (is_correct=False)' if ground_truth_failed else 'SUCCESS (is_correct=True)'}")
    print()
    print(f"  {'Turn':>4}  {'Speaker':<26}  {'P(Fail)':>7}  {'Belief Bar':<{BAR_WIDTH+2}}  Risk    Action")
    print("-" * 72)

    prob_history = []
    prev_action  = None

    for i, turn in enumerate(history):
        history_slice = history[: i + 1]
        evidence = extract_evidence_from_partial(data, history_slice)

        # ── Bayesian Network: Exact Inference via Variable Elimination ──────
        try:
            result    = inference.posterior_failure(evidence)
            prob_fail = float(
                result.values[result.state_names["Task_Failure"].index("Yes")]
            )
        except Exception:
            prob_fail = 0.50   # fallback to prior if VE fails on this slice

        prob_history.append(prob_fail)

        # ── Decision Network: Expected Utility ───────────────────────────────
        eu_dict     = dn._compute_eu(prob_fail)
        best_action = max(eu_dict, key=eu_dict.get)
        action_str, risk_label = get_intervention(prob_fail)

        speaker = safe_str(str(turn.get("role", "unknown")), max_len=24)
        bar     = make_bar(prob_fail)

        # Mark turns where the recommended action changes
        action_changed = (best_action != prev_action) and (prev_action is not None)
        alert = "  *** ACTION CHANGE ***" if action_changed else ""

        print(
            f"  {i+1:>4}  {speaker:<26}  {prob_fail:>6.1%}  {bar}  "
            f"{risk_label:<8} {action_str}{alert}"
        )

        if verbose_evidence:
            for k, v in sorted(evidence.items()):
                print(f"           {k}: {v}")
            print()

        prev_action = best_action

        if delay > 0:
            time.sleep(delay)

    print("-" * 72)
    return prob_history


# ─────────────────────────────────────────────────────────────────────────────
#  POST-MORTEM
# ─────────────────────────────────────────────────────────────────────────────

def print_postmortem(
    data: dict,
    inference: FailNetInference,
    dn: MockDecisionNetwork,
    prob_history: list,
):
    """
    Prints post-mortem analysis after the full trajectory replay:
      1. Prediction vs. Ground Truth
      2. Danger profile (peak turn)
      3. Belief trajectory (mini trend)
      4. Markov Blanket of Task_Failure
      5. MAP Causal Diagnosis
      6. Who&When ground-truth annotation
      7. Decision Network utility breakdown
    """
    history               = data.get("history", [])
    ground_truth_failed   = not data.get("is_correct", True)
    mistake_agent         = safe_str(data.get("mistake_agent", "Unknown"))
    mistake_reason        = safe_str(data.get("mistake_reason", "Not provided"), max_len=200)
    mistake_step          = safe_str(str(data.get("mistake_step", "Unknown")))

    # Final evidence = all turns visible
    final_evidence = extract_evidence_from_partial(data, history)

    final_prob      = prob_history[-1] if prob_history else 0.50
    action_str, _   = get_intervention(final_prob)
    predicted_fail  = final_prob >= 0.50

    print()
    print("=" * 72)
    print("  POST-MORTEM ANALYSIS")
    print("=" * 72)

    # [1] Prediction vs ground truth
    print()
    print("  [1] PREDICTION  vs  GROUND TRUTH")
    print(f"      Final P(Task_Failure=Yes) : {final_prob:.1%}")
    print(f"      FailNet verdict           : {'FAILURE' if predicted_fail else 'SUCCESS'}")
    print(f"      Actual outcome            : {'FAILURE' if ground_truth_failed else 'SUCCESS'}")
    correct = predicted_fail == ground_truth_failed
    print(f"      Prediction result         : {'CORRECT' if correct else 'INCORRECT'}")

    # [2] Peak danger
    if prob_history:
        peak_prob = max(prob_history)
        peak_turn = prob_history.index(peak_prob) + 1
        print()
        print("  [2] TRAJECTORY DANGER PROFILE")
        print(f"      Peak P(Fail)  = {peak_prob:.1%}  at Turn {peak_turn} of {len(prob_history)}")
        print(f"      Final action  : {action_str}")

    # [3] Mini belief trend
    print()
    print("  [3] BELIEF TRAJECTORY  (sampled every few turns)")
    if prob_history:
        step = max(1, len(prob_history) // 8)
        for idx in range(0, len(prob_history), step):
            bar = make_bar(prob_history[idx], width=15)
            print(f"       Turn {idx+1:>3}:  {prob_history[idx]:>5.1%}  {bar}")
        # Always show final turn
        last = len(prob_history) - 1
        if last % step != 0:
            bar = make_bar(prob_history[last], width=15)
            print(f"       Turn {last+1:>3}:  {prob_history[last]:>5.1%}  {bar}  <- final")

    # [4] Markov Blanket
    print()
    print("  [4] MARKOV BLANKET  of  Task_Failure")
    print("      The minimal set of nodes sufficient to predict Task_Failure,")
    print("      rendering it independent of all other network variables.")
    try:
        blanket = inference.get_markov_blanket("Task_Failure")
        for node in blanket:
            val = final_evidence.get(node, "(unobserved)")
            print(f"        {node:<40} = {val}")
    except Exception as e:
        print(f"      (Markov Blanket computation failed: {e})")

    # [5] MAP Causal Diagnosis
    print()
    print("  [5] MAP CAUSAL DIAGNOSIS")
    print("      Most probable causal configuration given Task_Failure=Yes.")
    print("      (Computed via MAP query over key diagnostic nodes.)")
    diag_ev = {**final_evidence, "Task_Failure": "Yes"}
    try:
        map_result = inference.map_diagnosis(diag_ev)
        for node, state in map_result.items():
            print(f"        {node:<40} -> {state}")
    except Exception as e:
        print(f"      (MAP diagnosis failed: {e})")

    # [6] Ground truth annotation from Who&When
    print()
    print("  [6] WHO&WHEN GROUND-TRUTH ANNOTATION")
    print(f"      Responsible agent : {mistake_agent}")
    print(f"      Failure step      : {mistake_step}")
    print(f"      Failure reason    : {mistake_reason[:120]}")

    # [7] Decision Network EU breakdown
    print()
    print("  [7] DECISION NETWORK  --  Expected Utilities at Final P(Fail)")
    eu_dict = dn._compute_eu(final_prob)
    best_eu = max(eu_dict.values())
    for action, eu in sorted(eu_dict.items(), key=lambda x: -x[1]):
        tag = "  <-- OPTIMAL" if eu == best_eu else ""
        print(f"      EU( {action:<22} ) = {eu:+6.2f}{tag}")

    print()
    print("=" * 72)
    print("  Demo complete.")
    print("  FailNet acted as a real-time Bayesian supervisor on a live trajectory.")
    print("=" * 72)
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY: List available trajectory files
# ─────────────────────────────────────────────────────────────────────────────

def list_trajectories():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.json"), recursive=True))
    print(f"\nAvailable trajectory files ({len(files)} total):\n")
    for i, f in enumerate(files):
        size_kb = os.path.getsize(f) // 1024
        print(f"  [{i:>3}]  {os.path.relpath(f)}  ({size_kb} KB)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FailNet Live Integration Demo -- stream a real agent trajectory through the BN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python simulate_live_integration.py
          python simulate_live_integration.py --file "data/original data/Who&When/Hand-Crafted/1.json"
          python simulate_live_integration.py --index 5
          python simulate_live_integration.py --list
          python simulate_live_integration.py --delay 0.3 --verbose
        """),
    )
    parser.add_argument("--file",    type=str,   default=None,
                        help="Path to a specific JSON trajectory file.")
    parser.add_argument("--index",   type=int,   default=None,
                        help="Index into the sorted file list (0-based). Use --list to see indices.")
    parser.add_argument("--list",    action="store_true",
                        help="List all available trajectory files and exit.")
    parser.add_argument("--delay",   type=float, default=0.0,
                        help="Seconds to pause between streaming turns (default: 0 = instant).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print the full evidence dict at each turn.")
    args = parser.parse_args()

    if args.list:
        list_trajectories()
        return

    # ── Load FailNet Model ───────────────────────────────────────────────────
    if not os.path.exists(MODEL_DATA):
        print(f"\n[ERROR] Processed data not found at '{MODEL_DATA}'.")
        print("        Run:  python run_pipeline.py")
        sys.exit(1)

    print("\nLoading FailNet Bayesian Network (BDeu parameter estimator)...")
    model     = learn_parameters(data_path=MODEL_DATA, optimize=False)
    inference = FailNetInference(model)
    dn        = MockDecisionNetwork(inference)
    print("Model loaded and ready.\n")

    # ── Select trajectory file ───────────────────────────────────────────────
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.json"), recursive=True))
    if not all_files:
        print(f"[ERROR] No JSON files found in '{DATA_DIR}'.")
        sys.exit(1)

    if args.file:
        traj_path = args.file
    elif args.index is not None:
        if args.index >= len(all_files):
            print(f"[ERROR] Index {args.index} out of range (max: {len(all_files)-1}). Use --list.")
            sys.exit(1)
        traj_path = all_files[args.index]
    else:
        # Default: smallest file = shortest trajectory = cleanest demo
        traj_path = min(all_files, key=os.path.getsize)

    print(f"Trajectory : {os.path.relpath(traj_path)}  ({os.path.getsize(traj_path)//1024} KB)\n")

    with open(traj_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data.get("history"):
        print("[ERROR] Selected file has no 'history'. Choose another file with --index or --file.")
        sys.exit(1)

    # ── Stream ──────────────────────────────────────────────────────────────
    prob_history = stream_trajectory(
        data=data,
        inference=inference,
        dn=dn,
        delay=args.delay,
        verbose_evidence=args.verbose,
    )

    # ── Post-mortem ──────────────────────────────────────────────────────────
    print_postmortem(
        data=data,
        inference=inference,
        dn=dn,
        prob_history=prob_history,
    )


if __name__ == "__main__":
    main()
