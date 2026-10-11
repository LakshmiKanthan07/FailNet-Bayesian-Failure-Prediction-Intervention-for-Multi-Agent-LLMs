"""
feature_engineering.py
-----------------------
Transforms raw JSON trajectory data from the Who&When dataset into
structured, discrete features for the FailNet Bayesian Network.

Each function takes the raw data dict (from one JSON file) and returns
a single string category for that feature node.

All features are derived from OBSERVABLE signals only — no feature
uses the final Task_Failure label (no label leakage).
"""

import re


# ─────────────────────────────────────────────
#  Tier 1: Observable Task-Level Features
# ─────────────────────────────────────────────

def extract_task_complexity(data: dict) -> str:
    """
    Task_Complexity: Low | Medium | High
    Maps from the 'level' field in the Who&When schema.
    """
    level = data.get("level", "Medium")
    mapping = {"Hard": "High", "Easy": "Low", "Medium": "Medium"}
    return mapping.get(level, "Medium")


def extract_trajectory_length(history: list) -> str:
    """
    Trajectory_Length: Short | Medium | Long
    Based on the number of turns in the conversation history.
    """
    n = len(history)
    if n < 5:
        return "Short"
    elif n <= 15:
        return "Medium"
    return "Long"


def extract_context_window_load(history: list) -> str:
    """
    Context_Window_Load: Low | Medium | High
    Approximated by total character count of all history content.
    """
    total_chars = sum(len(str(turn.get("content", ""))) for turn in history)
    if total_chars < 2000:
        return "Low"
    elif total_chars <= 8000:
        return "Medium"
    return "High"


def extract_required_tool_count(history: list) -> str:
    """
    Required_Tool_Count: Few | Moderate | Many
    Counts turns where the user returned a tool exit code.
    """
    tool_calls = sum(
        1 for turn in history
        if turn.get("role") == "user" and "exitcode" in str(turn.get("content", ""))
    )
    if tool_calls < 3:
        return "Few"
    elif tool_calls <= 8:
        return "Moderate"
    return "Many"


def extract_multi_agent(data: dict) -> str:
    """
    Multi_Agent_Coordination_Required: Yes | No
    Inferred from the number of keys in the system_prompt dict.
    """
    system_prompt = data.get("system_prompt", {})
    return "Yes" if len(system_prompt.keys()) > 1 else "No"


# ─────────────────────────────────────────────
#  Tier 2: Derived Complexity & Planning
# ─────────────────────────────────────────────

def extract_planning_complexity(task_comp: str, req_tools: str) -> str:
    """
    Planning_Complexity: Simple | Moderate | Complex
    Derived from task complexity and tool count.
    """
    if task_comp == "Low" and req_tools == "Few":
        return "Simple"
    elif task_comp == "High" or req_tools == "Many":
        return "Complex"
    return "Moderate"


def extract_plan_quality(history: list, mistake_reason: str) -> str:
    """
    Plan_Quality: Good | Flawed
    Derived from OBSERVABLE signals only — NOT from the Task_Failure label.
    Flawed if the mistake_reason mentions planning keywords, or if the agent
    produces no structured planning turns (no assistant turns before tool calls).
    """
    # Check mistake reason for planning keywords
    if any(kw in mistake_reason for kw in ["plan", "strateg", "goal", "approach", "misunderstood"]):
        return "Flawed"
    # Heuristic: if the agent uses tools without any prior planning assistant turn, plan is flawed
    saw_assistant = False
    for turn in history:
        role = turn.get("role", "")
        content = str(turn.get("content", ""))
        if role == "assistant":
            saw_assistant = True
        if role == "user" and "exitcode" in content and not saw_assistant:
            return "Flawed"
    return "Good"


def extract_memory_retrieval_quality(cw_load: str) -> str:
    """
    Memory_Retrieval_Quality: Good | Poor
    High context window load correlates with poor retrieval.
    """
    return "Poor" if cw_load == "High" else "Good"


# ─────────────────────────────────────────────
#  Tier 3: Execution & Error Features
# ─────────────────────────────────────────────

def extract_execution_error(history: list) -> str:
    """
    Execution_Error_Occurred: Yes | No
    Scans for error signatures in history content.
    """
    error_patterns = ["exitcode: 1", "Traceback", "Error:", "Exception:", "stderr:"]
    for turn in history:
        content = str(turn.get("content", ""))
        if any(p in content for p in error_patterns):
            return "Yes"
    return "No"


def extract_error_recovery_attempted(exec_error: str, history: list) -> str:
    """
    Error_Recovery_Attempted: Yes | No
    Yes if the agent produced another assistant turn after an error turn.
    """
    if exec_error == "No":
        return "No"
    error_found = False
    for turn in history:
        content = str(turn.get("content", ""))
        if any(p in content for p in ["exitcode: 1", "Traceback", "Error:"]):
            error_found = True
        elif error_found and turn.get("role") == "assistant":
            return "Yes"
    return "No"


def extract_error_recovery_success(recovery_attempted: str, is_correct: bool) -> str:
    """
    Error_Recovery_Success: Yes | No | N/A
    Only meaningful when recovery was attempted.
    Uses is_correct from dataset ground truth (this is outcome-based but
    only applies to the recovery sub-outcome, not the main label).
    """
    if recovery_attempted == "No":
        return "N/A"
    return "Yes" if is_correct else "No"


def extract_tool_selection(history: list, mistake_reason: str) -> str:
    """
    Tool_Selection_Correctness: Correct | Incorrect
    Derived from OBSERVABLE signals — NOT from the Task_Failure label.
    Incorrect if the mistake reason mentions tool selection issues,
    or if the agent tries to call a tool that responds with a 'not found' error.
    """
    if any(kw in mistake_reason for kw in ["tool", "function", "api", "call"]):
        return "Incorrect"
    # Heuristic: look for "not found" or "no such command" responses from tools
    not_found_patterns = ["not found", "no such", "command not found", "ModuleNotFoundError", "ImportError"]
    for turn in history:
        if turn.get("role") == "user":
            content = str(turn.get("content", "")).lower()
            if any(p.lower() in content for p in not_found_patterns):
                return "Incorrect"
    return "Correct"


def extract_tool_argument_validity(history: list, mistake_reason: str) -> str:
    """
    Tool_Argument_Validity: Valid | Invalid
    Derived from OBSERVABLE signals — NOT from the Task_Failure label.
    Invalid if the mistake reason mentions argument/parameter issues,
    or if any tool response has a type/value error.
    """
    if any(kw in mistake_reason for kw in ["argument", "parameter", "input", "format"]):
        return "Invalid"
    # Heuristic: look for TypeError / ValueError in tool responses
    type_error_patterns = ["TypeError", "ValueError", "AttributeError", "KeyError", "IndexError"]
    for turn in history:
        content = str(turn.get("content", ""))
        if any(p in content for p in type_error_patterns):
            return "Invalid"
    return "Valid"


# ─────────────────────────────────────────────
#  Tier 4: Hallucination & Verification
# ─────────────────────────────────────────────

def extract_hallucination_risk(cw_load: str, base_model_cap: str) -> str:
    """
    Hallucination_Risk: Low | High
    High when context is overloaded and model capability is low.
    """
    return "High" if (cw_load == "High" and base_model_cap == "Low") else "Low"


def extract_intermediate_verification(history: list) -> str:
    """
    Intermediate_Verification_Quality: High | Low
    Derived from OBSERVABLE signals — NOT from the Task_Failure label.
    Low if the agent never produces an assistant turn that inspects
    or summarizes the result of a previous tool call before the next one.
    High if there are 'reflection' turns between tool uses.
    """
    prev_was_tool_result = False
    for turn in history:
        role = turn.get("role", "")
        content = str(turn.get("content", ""))
        # Tool result = user turn with exitcode
        if role == "user" and "exitcode" in content:
            prev_was_tool_result = True
        elif role == "assistant" and prev_was_tool_result:
            # If the assistant produced a turn between tool results → it's verifying
            return "High"
            prev_was_tool_result = False
    return "Low"


# ─────────────────────────────────────────────
#  Tier 5: Multi-Agent Coordination
# ─────────────────────────────────────────────

def extract_inter_agent_comm(multi_agent: str, history: list) -> str:
    """
    Inter_Agent_Communication_Quality: High | Low | N/A
    Only meaningful in multi-agent settings.
    Derived from OBSERVABLE signals — NOT from the Task_Failure label.
    Low if there are long gaps between agent roles (no handoff acknowledgement turns).
    """
    if multi_agent == "No":
        return "N/A"
    # Heuristic: in multi-agent runs, count distinct roles in history
    roles = set(turn.get("role", "") for turn in history)
    # If only one role aside from 'user', coordination quality is likely Low
    assistant_roles = [r for r in roles if r not in ("user", "system", "")]
    return "High" if len(assistant_roles) > 1 else "Low"


def extract_role_assignment(multi_agent: str, data: dict) -> str:
    """
    Role_Assignment_Correctness: Correct | Incorrect | N/A
    Only meaningful in multi-agent settings.
    """
    if multi_agent == "No":
        return "N/A"
    return "Incorrect" if data.get("mistake_agent") else "Correct"


# ─────────────────────────────────────────────
#  Tier 6: Outcome Features
# ─────────────────────────────────────────────

def extract_failure_severity(task_fail: str) -> str:
    """
    Failure_Severity: Critical | None
    Straightforward from task outcome.
    """
    return "Critical" if task_fail == "Yes" else "None"
