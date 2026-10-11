import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.graph_objects as go
import plotly.express as px

# ── path setup ──────────────────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.bn_learning import learn_parameters
from src.inference import FailNetInference
from src.decision_network import MockDecisionNetwork

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FailNet — AI Agent Failure Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-card {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    color: white;
    margin-bottom: 1.5rem;
}
.hero-card h1 { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.3rem; }
.hero-card p  { font-size: 1.0rem; color: #c0c0e0; line-height: 1.7; }

.tier-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 0.7rem;
    font-weight: 600;
    margin-left: 6px;
    vertical-align: middle;
}
.tier1 { background: #1a3a4a; color: #4fc3f7; }
.tier2 { background: #1a3a2a; color: #81c784; }
.tier3 { background: #3a1a3a; color: #ce93d8; }

.risk-low    { background: linear-gradient(90deg,#1b4332,#2d6a4f); color:#b7e4c7; }
.risk-medium { background: linear-gradient(90deg,#7b4500,#a05b00); color:#ffe0a3; }
.risk-high   { background: linear-gradient(90deg,#6b0000,#9b0000); color:#ffc0c0; }

.risk-box {
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    text-align: center;
    margin-bottom: 1rem;
}
.risk-box .risk-pct  { font-size: 3.5rem; font-weight: 700; }
.risk-box .risk-label{ font-size: 1rem; opacity: 0.85; }

.action-box {
    border-radius: 10px;
    padding: 1rem 1.5rem;
    font-size: 1.1rem;
    font-weight: 600;
    text-align: center;
    margin: 0.5rem 0;
    border: 2px solid;
}
.action-continue   { background:#0d2e1a; color:#4ade80; border-color:#4ade80; }
.action-retry      { background:#2e2200; color:#facc15; border-color:#facc15; }
.action-escalate   { background:#1e1a00; color:#fb923c; border-color:#fb923c; }
.action-halt       { background:#2e0000; color:#f87171; border-color:#f87171; }

.map-row {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.88rem;
    color: #e2e8f0;
    padding: 4px 0;
}
.map-arrow { color: #818cf8; font-size: 1.1rem; }

.info-pill {
    background: #1e293b;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    color: #94a3b8;
    border-left: 3px solid #818cf8;
    margin: 4px 0;
}

.node-desc-box {
    background: #111827;
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
    font-size: 0.78rem;
    color: #9ca3af;
    margin-top: 2px;
    border-left: 2px solid #374151;
}

.mb-chip {
    display: inline-block;
    background: #1e3a5f;
    color: #93c5fd;
    border: 1px solid #3b82f6;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 3px 3px;
    white-space: nowrap;
}
.mb-chip.target {
    background: #4c1d95;
    color: #c4b5fd;
    border-color: #7c3aed;
    font-size: 0.85rem;
    padding: 4px 14px;
}
.mb-section {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── model loader ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Bayesian Network… (first run may take ~30 s)")
def load_models():
    data_path = os.path.join(ROOT_DIR, "data", "processed", "bn_evidence_table.csv")
    model = learn_parameters(
        data_path=data_path, optimize=False
    )
    inference = FailNetInference(model)
    dn = MockDecisionNetwork(inference)
    return inference, dn


# ════════════════════════════════════════════════════════════════════════════
#   SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/64/brain.png", width=56)
    st.markdown("## FailNet Monitor")
    st.caption("Bayesian Failure Prediction · v1.0")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Overview", "Live Monitor", "Agent Simulator", "Dataset Stats", "Results", "How It Works"],
        label_visibility="collapsed",
    )
    st.divider()

    st.divider()
    st.caption("Built with pgmpy · Streamlit · Plotly")


# ════════════════════════════════════════════════════════════════════════════
#   PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("""
    <div class="hero-card">
      <h1>FailNet</h1>
      <p><strong>Bayesian Failure Prediction &amp; Intervention for Multi-Agent LLMs</strong><br>
      An interpretable, causal alternative to black-box agent observability tools.<br>
      Instead of simply logging <em>that</em> an agent failed, FailNet uses a 
      <strong>24-node Bayesian Network</strong> to predict failures dynamically,
      diagnose root causes, and recommend automated safety interventions.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Network Nodes", "24", help="Bayesian Network nodes across 6 tiers")
    col2.metric("Network Edges", "37", help="Causal dependency edges in the DAG")
    col3.metric("Training Samples", "204+", help="Real Who&When traces used for parameter learning")

    st.markdown("---")
    st.subheader("The Problem")
    st.markdown("""
    As AI shifts to **multi-agent systems** (AutoGen, LangGraph, CrewAI), multiple agents
    collaborate: a *Planner* → *Coder* → *Executor*. This creates two critical risks:

    | Problem | Description |
    |---|---|
    | **Cascading Errors** | One agent's small mistake becomes the next agent's truth. Errors compound exponentially. |
    | **Black-Box Observability** | Tools like LangSmith only tell you *that* a crash happened — never *why*, and always *after* it occurs. |
    """)

    st.subheader("The FailNet Solution")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **1. Live Belief Update**  
        As agents run, FailNet streams their telemetry into the BN and continuously updates  
        `P(Task_Failure | Evidence)` using Variable Elimination.

        **2. Root Cause Diagnosis (MAP)**  
        When failure probability spikes, a MAP query finds the *most probable causal chain*:  
        e.g., `Instruction Ambiguity → Plan Quality → Hallucination Risk`.
        """)
    with c2:
        st.markdown("""
        **3. Decision-Theoretic Intervention**  
        Expected Utility is computed for 4 actions and the optimal one is surfaced:  
        `Continue` / `Retry_Step` / `Escalate_to_Human` / `Halt`.

        **4. Handles Incomplete Evidence**  
        If only 3 of 24 nodes are observed, FailNet *marginalizes out* the rest —  
        unlike XGBoost, which breaks without all features.
        """)

    st.subheader("Architecture")
    st.code("""
  [Raw Agent Logs / Telemetry]
          ↓
  [24-Node Feature Extraction]
   Tier 1: Observable  (e.g., Execution Error, Trajectory Length)
   Tier 2: Heuristic   (e.g., Tool Argument Validity)
   Tier 3: LLM-Proxy   (e.g., Plan Quality, Hallucination Risk)
          ↓
  [Bayesian Network — Dirichlet/BDeu Parameter Learning]
          ↓
  [Variable Elimination → P(Task_Failure | Evidence)]
          ↓
  [MAP Inference → Root Cause Diagnosis]
          ↓
  [Expected Utility → Intervention Policy]
    """, language="text")


# ════════════════════════════════════════════════════════════════════════════
#   PAGE 2 — LIVE MONITOR
# ════════════════════════════════════════════════════════════════════════════
elif page == "Live Monitor":

    # ── Node definitions for the UI ──────────────────────────────────────
    NODE_DEFS = {
        # (label, tier, options, description, most-concerning value)
        "Task_Complexity": (
            "Task Complexity", "Tier 1 · Observable",
            ["(Not Set)", "Low", "Medium", "High"],
            "How difficult / multi-step is the task being assigned to the agent pipeline?",
            "High",
        ),
        "Instruction_Ambiguity": (
            "Instruction Ambiguity", "Tier 3 · LLM-Proxy",
            ["(Not Set)", "Low", "High"],
            "Are the instructions given to the agents vague or contradictory? Proxy-labeled by LLM judge.",
            "High",
        ),
        "Execution_Error_Occurred": (
            "Execution Error Occurred", "Tier 1 · Observable",
            ["(Not Set)", "No", "Yes"],
            "Did any agent in the pipeline throw a runtime/execution error (e.g., Python Traceback)?",
            "Yes",
        ),
        "Hallucination_Risk": (
            "Hallucination Risk", "Tier 3 · LLM-Proxy",
            ["(Not Set)", "Low", "High"],
            "Did any agent likely fabricate facts, steps, or tool outputs? Proxy-labeled by LLM judge.",
            "High",
        ),
        "Tool_Reliability": (
            "Tool Reliability", "Tier 2 · Heuristic",
            ["(Not Set)", "High", "Low"],
            "Are the external tools (APIs, code executors) the agents use known to be stable?",
            "Low",
        ),
        "Plan_Quality": (
            "Plan Quality", "Tier 3 · LLM-Proxy",
            ["(Not Set)", "Good", "Flawed"],
            "Was the plan generated by the Planner agent logically correct and complete? Proxy-labeled.",
            "Flawed",
        ),
        "Error_Recovery_Attempted": (
            "Error Recovery Attempted", "Tier 2 · Heuristic",
            ["(Not Set)", "No", "Yes"],
            "After an error occurred, did any agent attempt to re-plan, retry, or self-correct?",
            "No",
        ),
        "Multi_Agent_Coordination_Required": (
            "Multi-Agent Coordination", "Tier 1 · Observable",
            ["(Not Set)", "No", "Yes"],
            "Does the task require multiple agents to hand-off work between each other?",
            "Yes",
        ),
        "Context_Window_Load": (
            "Context Window Load", "Tier 2 · Heuristic",
            ["(Not Set)", "Low", "Medium", "High"],
            "How much of the LLM's context window is consumed by the conversation history so far?",
            "High",
        ),
        "Base_Model_Capability": (
            "Base Model Capability", "Tier 3 · LLM-Proxy",
            ["(Not Set)", "High", "Low"],
            "Is the underlying LLM being used known to be powerful (e.g., GPT-4) or limited (e.g., GPT-3.5)?",
            "Low",
        ),
    }

    TIER_CLASS = {
        "Tier 1 · Observable": "tier1",
        "Tier 2 · Heuristic": "tier2",
        "Tier 3 · LLM-Proxy": "tier3",
    }

    ACTION_CLASS = {
        "Continue": "action-continue",
        "Retry_Step": "action-retry",
        "Escalate_to_Human": "action-escalate",
        "Halt": "action-halt",
    }



    ACTION_DESC = {
        "Continue": "Risk is low. Allow the agent pipeline to proceed normally.",
        "Retry_Step": "Moderate risk detected. Trigger an automatic retry of the last step.",
        "Escalate_to_Human": "High ambiguity or risk. Route to a human operator for review.",
        "Halt": "Critical failure predicted. Stop the pipeline immediately to prevent damage.",
    }

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-card">
      <h1>Live Agent Monitor</h1>
      <p>Inject real-time telemetry signals from your running agent pipeline below.
      The Bayesian Network will update failure probability and recommend an intervention
      action <strong>instantly</strong> as you change the signals.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load model ────────────────────────────────────────────────────────
    try:
        inference, dn = load_models()
    except Exception as e:
        st.error(f"Could not load the Bayesian Network model.\n\nEnsure you have run `python run_pipeline.py` first.\n\n**Error:** `{e}`")
        st.stop()

    # ── Evidence injection panel ──────────────────────────────────────────
    st.subheader("Step 1 · Inject Agent Telemetry Signals")
    st.caption("Select what you are observing from the running agent pipeline. Leave a signal as '(Not Set)' if it is unknown — the BN will marginalize it out.")

    evidence = {}
    cols = st.columns(2)
    node_keys = list(NODE_DEFS.keys())

    for i, node_key in enumerate(node_keys):
        label, tier, options, description, concerning = NODE_DEFS[node_key]
        tier_cls = TIER_CLASS[tier]
        col = cols[i % 2]
        with col:
            st.markdown(f"**{label}** <span class='tier-badge {tier_cls}'>{tier}</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='node-desc-box'>{description}</div>", unsafe_allow_html=True)
            val = st.selectbox(
                label, options=options,
                key=f"node_{node_key}",
                label_visibility="collapsed",
            )
            if val != "(Not Set)":
                evidence[node_key] = val

    st.divider()

    # ── Run inference ─────────────────────────────────────────────────────
    st.subheader("Step 2 · Live Inference Results")

    with st.spinner("Computing Bayesian inference…"):
        try:
            # --- Exact inference (Variable Elimination) ---
            res_exact     = inference.posterior_failure(evidence)
            prob_fail = float(res_exact.values[res_exact.state_names["Task_Failure"].index("Yes")])

            map_diag      = inference.map_diagnosis(evidence)
            opt_action, eus, best_diag, best_voi = dn.recommend_active_intervention(evidence)
        except Exception as e:
            st.error(f"Inference error: {e}")
            st.stop()

    # ── Risk gauge ────────────────────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("#### Failure Probability")
        if prob_fail < 0.35:
            risk_cls, risk_label = "risk-low", "LOW RISK"
        elif prob_fail < 0.65:
            risk_cls, risk_label = "risk-medium", "MEDIUM RISK"
        else:
            risk_cls, risk_label = "risk-high", "HIGH RISK"

        st.markdown(f"""
        <div class="risk-box {risk_cls}">
          <div class="risk-pct">{prob_fail:.1%}</div>
          <div class="risk-label">P(Task Failure | Evidence) &nbsp;·&nbsp; {risk_label}</div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_fail * 100,
            number={"suffix": "%", "font": {"size": 32}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#555"},
                "bar": {"color": "#ef4444" if prob_fail > 0.65 else "#f59e0b" if prob_fail > 0.35 else "#22c55e"},
                "bgcolor": "#1e293b",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35],  "color": "#14532d"},
                    {"range": [35, 65], "color": "#713f12"},
                    {"range": [65, 100],"color": "#7f1d1d"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.8,
                    "value": prob_fail * 100,
                },
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0"},
            height=220,
            margin=dict(t=10, b=10, l=20, r=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Evidence count
        if evidence:
            st.success(f"**{len(evidence)} signal(s)** injected · {24 - len(evidence)} nodes marginalized by BN")
        else:
            st.info("No evidence set — showing **prior marginal** (base rate before any observation).")

    with right:
        st.markdown("#### Recommended Intervention")
        ac = ACTION_CLASS.get(opt_action, "action-continue")
        desc = ACTION_DESC.get(opt_action, "")
        st.markdown(f"""
        <div class="action-box {ac}">
          {opt_action.replace('_', ' ')}
        </div>
        """, unsafe_allow_html=True)
        st.caption(desc)

        st.markdown("#### Expected Utility per Action")
        eus_df = pd.DataFrame.from_dict(eus, orient="index", columns=["Expected Utility"])
        eus_df.index = [e.replace("_", " ") for e in eus_df.index]
        eus_df["Color"] = ["#22c55e" if v == max(eus.values()) else "#64748b" for v in eus.values()]
        fig_eu = px.bar(
            eus_df, x=eus_df.index, y="Expected Utility",
            color="Color", color_discrete_map="identity",
            text="Expected Utility",
        )
        fig_eu.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_eu.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0"},
            height=240,
            showlegend=False,
            xaxis_title="",
            yaxis_title="EU",
            margin=dict(t=10, b=10, l=10, r=10),
        )
        fig_eu.update_xaxes(tickfont={"size": 11})
        st.plotly_chart(fig_eu, use_container_width=True)

    st.divider()

    # ── MAP Root Cause Diagnosis ──────────────────────────────────────────
    st.markdown("#### MAP Root Cause Diagnosis")
    st.caption("The most probable hidden causal state configuration given the current evidence. This is computed via MAP Inference (argmax over hidden nodes).")

    MAP_LABELS = {
        "Plan_Quality":                    ("Plan Quality",                   "Tier 3 · LLM-Proxy"),
        "Tool_Selection_Correctness":      ("Tool Selection Correctness",     "Tier 2 · Heuristic"),
        "Hallucination_Risk":              ("Hallucination Risk",             "Tier 3 · LLM-Proxy"),
        "Intermediate_Verification_Quality": ("Intermediate Verification",   "Tier 3 · LLM-Proxy"),
    }
    RISK_VALS = {"Flawed", "Incorrect", "High", "Low", "Yes"}

    if map_diag:
        map_cols = st.columns(len(map_diag))
        for idx, (node, val) in enumerate(map_diag.items()):
            label, tier = MAP_LABELS.get(node, (node, ""))
            tc = TIER_CLASS.get(tier, "tier1")
            is_risky = val in RISK_VALS
            with map_cols[idx]:
                st.markdown(f"<span class='tier-badge {tc}'>{tier}</span>", unsafe_allow_html=True)
                st.markdown(f"**{label}**")
                if is_risky:
                    st.error(f"**{val}**")
                else:
                    st.success(f"**{val}**")
    else:
        st.info("All MAP nodes are already set via evidence.")

    # Causal chain narrative
    causal_chain = []
    for node, val in map_diag.items():
        label, _ = MAP_LABELS.get(node, (node, ""))
        causal_chain.append(f"<strong>{label}</strong> = {val}")

    if causal_chain:
        chain_html = ' &nbsp;→&nbsp; '.join(causal_chain) + ' &nbsp;→&nbsp; <strong>Task Failure</strong>'
        st.markdown(f"<div class='info-pill'>Most probable causal chain: {chain_html}</div>", unsafe_allow_html=True)

    st.divider()

    # ── Value of Information ──────────────────────────────────────────────
    st.markdown("#### Active Diagnostic Recommendation (Value of Information)")
    if best_diag:
        st.markdown(f"""
        <div class="info-pill">
        <strong>Ask the agent about: <code>{best_diag.replace('_', ' ')}</code></strong><br>
        Querying this node will improve decision certainty by the most.
        Expected Utility Gain: <strong>{best_voi:.2f}</strong>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-pill">
        Current evidence is sufficient. No further diagnostic queries will significantly improve decision certainty.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Markov Blanket ────────────────────────────────────────────────────
    st.markdown("#### Markov Blanket of Task_Failure")
    st.caption("""
    The **Markov Blanket** is the minimal set of nodes that, when observed, makes `Task_Failure`
    conditionally independent of ALL other variables in the network. These are the **only nodes
    you ever need to observe** to perfectly predict failure — everything else is redundant.
    """)
    try:
        mb_nodes = inference.get_markov_blanket("Task_Failure")
        mb_in_evidence = set(mb_nodes) & set(evidence.keys())
        mb_unobserved  = set(mb_nodes) - set(evidence.keys())

        st.markdown(f"""
        <div class="mb-section">
          <div style="margin-bottom:0.5rem;">
            <span class="mb-chip target">Task_Failure</span>
            &nbsp; ← shielded by its Markov Blanket ({len(mb_nodes)} nodes)
          </div>
          <div>
            {''.join(f"<span class='mb-chip' style='background:#0f2e1a;color:#4ade80;border-color:#22c55e;'>{n.replace('_',' ')}</span>" for n in sorted(mb_in_evidence))}
            {''.join(f"<span class='mb-chip'>{n.replace('_',' ')}</span>" for n in sorted(mb_unobserved))}
          </div>
          <div style="margin-top:0.6rem;font-size:0.78rem;color:#64748b;">
            Observed in evidence (green) &nbsp;|&nbsp; Not yet observed ({len(mb_unobserved)} remaining)
          </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not compute Markov Blanket: {e}")

    st.divider()

    # ── Download Report ────────────────────────────────────────────────────
    st.markdown("#### Download Inference Report")
    import json as _json
    report_data = {
        "evidence":             evidence,
        "prob_failure":         round(prob_fail, 4),
        "risk_level":           risk_label,
        "recommended_action":   opt_action,
        "expected_utilities":   {k: round(v, 4) for k, v in eus.items()},
        "map_diagnosis":        {str(k): str(v) for k, v in map_diag.items()},
        "voi_recommendation":   best_diag,
        "voi_gain":             round(best_voi, 4) if best_voi else None,
        "markov_blanket":       mb_nodes if 'mb_nodes' in dir() else [],
    }
    report_json = _json.dumps(report_data, indent=2)
    st.download_button(
        label="Download JSON Report",
        data=report_json,
        file_name="failnet_inference_report.json",
        mime="application/json",
        help="Download the current inference result (evidence, probabilities, recommendation) as a JSON file.",
    )


# ════════════════════════════════════════════════════════════════════════════
#   PAGE 3 — DATASET STATS
# ════════════════════════════════════════════════════════════════════════════
elif page == "Dataset Stats":
    st.markdown("""
    <div class="hero-card">
      <h1>Dataset Statistics</h1>
      <p>Explore the <strong>Who&amp;When</strong> dataset used to train the Bayesian Network parameters.</p>
    </div>
    """, unsafe_allow_html=True)

    data_path = os.path.join(ROOT_DIR, "data", "processed", "bn_evidence_table.csv")
    if not os.path.exists(data_path):
        st.error(f"Dataset not found at `{data_path}`. Please run `python run_pipeline.py` first.")
        st.stop()

    df = pd.read_csv(data_path)
    if "label_provenance" in df.columns:
        df = df.drop(columns=["label_provenance"])

    st.subheader("Raw Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Key Distribution Plots")

    col1, col2 = st.columns(2)
    with col1:
        vc = df["Task_Failure"].value_counts().reset_index()
        fig = px.pie(vc, names="Task_Failure", values="count",
                     title="Task Failure Distribution",
                     color_discrete_sequence=["#22c55e", "#ef4444"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e2e8f0"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        vc2 = df["Task_Complexity"].value_counts().reset_index()
        fig2 = px.bar(vc2, x="Task_Complexity", y="count",
                      title="Task Complexity Breakdown",
                      color="Task_Complexity",
                      color_discrete_map={"Low":"#22c55e","Medium":"#f59e0b","High":"#ef4444"})
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color":"#e2e8f0"}, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        vc3 = df.groupby(["Plan_Quality","Task_Failure"]).size().reset_index(name="count")
        fig3 = px.bar(vc3, x="Plan_Quality", y="count", color="Task_Failure",
                      title="Plan Quality vs Task Failure",
                      color_discrete_map={"No":"#22c55e","Yes":"#ef4444"},
                      barmode="group")
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color":"#e2e8f0"})
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        vc4 = df.groupby(["Execution_Error_Occurred","Task_Failure"]).size().reset_index(name="count")
        fig4 = px.bar(vc4, x="Execution_Error_Occurred", y="count", color="Task_Failure",
                      title="Execution Error vs Task Failure",
                      color_discrete_map={"No":"#22c55e","Yes":"#ef4444"},
                      barmode="group")
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color":"#e2e8f0"})
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(include="all").astype(str), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#   PAGE 4 — EXPERIMENT RESULTS
# ════════════════════════════════════════════════════════════════════════════
elif page == "Results":
    st.markdown("""
    <div class="hero-card">
      <h1>Experiment Results</h1>
      <p>Key results from the 8 FailNet evaluation experiments. Run <code>python run_pipeline.py</code> to generate all figures and CSVs.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Sensitivity (Tornado)",
        "K-Fold Cross-Validation",
        "Decision Boundaries",
        "Streaming Belief",
        "Interactive DAG",
        "Partial Evidence Study",
    ])

    with tab1:
        st.markdown("#### Sensitivity Analysis — Tornado Chart")
        st.caption("""
        For each variable X, the Swing = P(Fail | X=worst) − P(Fail | X=best).
        A wider bar means that variable has more influence on failure probability.
        The Markov Blanket nodes are the top contributors.
        """)
        tornado_path = os.path.join(ROOT_DIR, "report", "figures", "exp5_tornado_chart.png")
        if os.path.exists(tornado_path):
            st.image(tornado_path, use_container_width=True)
        else:
            st.warning("Tornado chart not found. Run `python run_pipeline.py` to generate it.")
            try:
                inference_r, _ = load_models()
                with st.spinner("Computing sensitivity analysis (this takes ~30s)…"):
                    sa_df = inference_r.sensitivity_analysis()
                top_n = 12
                df_plot = sa_df.head(top_n)
                fig_tornado = go.Figure()
                base_p = float(sa_df["BaseProb"].iloc[0])
                for _, row in df_plot.iterrows():
                    fig_tornado.add_trace(go.Bar(
                        x=[row["BestProb"] - base_p],
                        y=[row["Variable"].replace("_", " ")],
                        orientation="h",
                        base=[base_p],
                        marker_color="#22c55e",
                        name="Best case",
                        showlegend=_ == 0,
                    ))
                    fig_tornado.add_trace(go.Bar(
                        x=[row["WorstProb"] - base_p],
                        y=[row["Variable"].replace("_", " ")],
                        orientation="h",
                        base=[base_p],
                        marker_color="#ef4444",
                        name="Worst case",
                        showlegend=_ == 0,
                    ))
                fig_tornado.add_vline(x=base_p, line_dash="dash", line_color="white",
                                      annotation_text=f"Base P={base_p:.2f}")
                fig_tornado.update_layout(
                    barmode="overlay",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#e2e8f0"},
                    xaxis_title="P(Task_Failure = Yes)",
                    height=500,
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(fig_tornado, use_container_width=True)
                st.dataframe(sa_df[["Variable","BestState","BestProb","WorstState","WorstProb","Swing"]].head(12),
                             use_container_width=True)
            except Exception as e:
                st.error(f"Could not compute: {e}")

    with tab2:
        st.markdown("#### Bayesian Network — 5-Fold Cross-Validation")
        cv_path = os.path.join(ROOT_DIR, "report", "exp8_kfold_cv_results.csv")
        if os.path.exists(cv_path):
            cv_df = pd.read_csv(cv_path)
            st.dataframe(cv_df, use_container_width=True)
            fig_cv = px.bar(
                cv_df, x="Metric", y="Mean", error_y="Std Dev",
                color="Metric",
                color_discrete_sequence=["#22c55e", "#3b82f6", "#f59e0b", "#ef4444"],
                title="5-Fold CV Mean Score ± Std Dev",
                text="Mean",
            )
            fig_cv.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_cv.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e2e8f0"},
                showlegend=False,
                yaxis_range=[0, 1.1],
                height=380,
            )
            st.plotly_chart(fig_cv, use_container_width=True)
            kfold_img = os.path.join(ROOT_DIR, "report", "figures", "exp8_kfold_cv.png")
            if os.path.exists(kfold_img):
                st.image(kfold_img, use_container_width=True)
        else:
            st.info("K-Fold CV results not found. Run `python run_pipeline.py` (Exp 8) to generate them.")

    with tab3:
        st.markdown("#### Decision Boundary Curve")
        st.caption("""
        Shows Expected Utility for each of the 4 actions across the full range of P(Task_Failure=Yes).
        Vertical dashed lines mark where the optimal action switches.
        """)
        db_img = os.path.join(ROOT_DIR, "report", "figures", "exp8_decision_boundary.png")
        if os.path.exists(db_img):
            st.image(db_img, use_container_width=True)
        else:
            st.info("Decision boundary figure not found. Run Exp 8 to generate it.")

        db_csv = os.path.join(ROOT_DIR, "report", "exp8_decision_boundaries.csv")
        if os.path.exists(db_csv):
            st.markdown("##### Analytical Crossover Thresholds")
            db_df = pd.read_csv(db_csv)
            st.dataframe(db_df, use_container_width=True)

    with tab4:
        st.markdown("#### Streaming Belief Update")
        st.caption("P(Task_Failure=Yes) evolves as evidence arrives step-by-step during an agent run.")
        sb_img = os.path.join(ROOT_DIR, "report", "figures", "exp5_streaming_belief.png")
        if os.path.exists(sb_img):
            st.image(sb_img, use_container_width=True)
        else:
            st.info("Streaming belief figure not found. Run `python run_pipeline.py` to generate it.")

    # ═══════════════════════════════════════════════════════════════
    #  TAB 5 — INTERACTIVE DAG
    # ═══════════════════════════════════════════════════════════════
    with tab5:
        st.markdown("#### Interactive Bayesian Network DAG")
        st.caption("""
        The 24-node Directed Acyclic Graph (DAG) of FailNet.  
        **Node colors:** Red = Root cause nodes (no parents) · Yellow = Intermediate nodes · Blue = Outcome nodes.  
        Drag nodes to rearrange. Hover for node names.
        """)

        try:
            from pyvis.network import Network
            import tempfile, base64

            # Build color mapping
            from src.bn_structure import failnet_edges as _edges
            all_nodes = set()
            has_parent = set()
            has_child  = set()
            for (u, v) in _edges:
                all_nodes.add(u); all_nodes.add(v)
                has_parent.add(v)
                has_child.add(u)

            def _node_color(n):
                is_root    = n not in has_parent
                is_outcome = n not in has_child
                if is_root:    return "#ef4444"   # red   — root cause
                if is_outcome: return "#3b82f6"   # blue  — outcome
                return "#f59e0b"                  # amber — intermediate

            net = Network(
                height="520px", width="100%",
                bgcolor="#0f172a", font_color="#e2e8f0",
                directed=True,
            )
            net.set_options("""
            {
              "nodes": {"font": {"size": 11, "color": "#e2e8f0"},
                         "borderWidth": 2, "shadow": true},
              "edges": {"arrows": {"to": {"enabled": true, "scaleFactor": 0.7}},
                         "color": {"color": "#475569", "highlight": "#818cf8"},
                         "smooth": {"type": "curvedCW", "roundness": 0.2}},
              "physics": {"barnesHut": {"gravitationalConstant": -6000,
                                         "centralGravity": 0.3,
                                         "springLength": 120},
                           "stabilization": {"iterations": 120}},
              "interaction": {"hover": true, "tooltipDelay": 100}
            }
            """)

            for node in sorted(all_nodes):
                color = _node_color(node)
                label = node.replace("_", "\n")
                net.add_node(node, label=label, color=color,
                             title=node.replace("_", " "),
                             shape="dot", size=14)

            for (u, v) in _edges:
                net.add_edge(u, v)

            with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w",
                                             encoding="utf-8") as f:
                net.save_graph(f.name)
                html_path = f.name

            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            # Inject the HTML via components.html
            import streamlit.components.v1 as components
            components.html(html_content, height=540, scrolling=False)

            # Legend
            st.markdown("""
            <div style="display:flex;gap:1.5rem;margin-top:0.5rem;font-size:0.82rem;color:#94a3b8;">
              <span><span style="color:#ef4444;font-size:1rem;">●</span> Root cause (no parents)</span>
              <span><span style="color:#f59e0b;font-size:1rem;">●</span> Intermediate node</span>
              <span><span style="color:#3b82f6;font-size:1rem;">●</span> Outcome / leaf node</span>
            </div>
            """, unsafe_allow_html=True)

        except ImportError:
            st.warning("""
            **pyvis** is not installed. Install it to enable the interactive DAG:
            ```bash
            pip install pyvis
            ```
            Falling back to static BN visualization:
            """)
            static_bn = "bn_visualization.png"
            if os.path.exists(static_bn):
                st.image(static_bn, use_container_width=True)
            else:
                st.info("Static BN image not found either. Run `python run_pipeline.py`.")
        except Exception as exc:
            st.error(f"DAG render error: {exc}")

    with tab6:
        st.markdown("#### Partial Evidence Robustness Study (Exp 9)")
        st.caption("""
        **Key claim of FailNet**: Unlike XGBoost or Logistic Regression (which crash without all features),
        the Bayesian Network gracefully marginalizes unobserved variables.

        This experiment proves it: as features are progressively removed, accuracy degrades smoothly
        — and stays well above chance even with only 3 variables observed.
        """)

        pe_csv  = os.path.join(ROOT_DIR, "report", "exp9_partial_evidence.csv")
        pe_img  = os.path.join(ROOT_DIR, "report", "figures", "exp9_partial_evidence.png")

        if os.path.exists(pe_csv):
            pe_df = pd.read_csv(pe_csv)

            # Interactive Plotly chart
            fig_pe = go.Figure()
            color_map = {"Accuracy": "#22c55e", "F1": "#818cf8", "AUC-ROC": "#f59e0b"}
            for metric, color in color_map.items():
                if metric in pe_df.columns:
                    fig_pe.add_trace(go.Scatter(
                        x=pe_df["Features_Observed"], y=pe_df[metric],
                        mode="lines+markers+text",
                        name=metric,
                        line=dict(color=color, width=2.5),
                        marker=dict(size=9, color=color),
                        text=[f"{v:.2f}" for v in pe_df[metric]],
                        textposition="top center",
                        textfont=dict(size=10, color=color),
                    ))

            # Chance line
            fig_pe.add_hline(y=0.5, line_dash="dash", line_color="#475569",
                             annotation_text="Chance baseline (0.50)",
                             annotation_position="bottom right",
                             annotation_font_color="#94a3b8")

            fig_pe.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e2e8f0"},
                xaxis_title="Number of Features Observed (out of 22)",
                yaxis_title="Score",
                yaxis_range=[0.3, 1.08],
                height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=30, b=20, l=10, r=10),
            )
            fig_pe.update_xaxes(dtick=2, gridcolor="#334155")
            fig_pe.update_yaxes(gridcolor="#334155")

            st.plotly_chart(fig_pe, use_container_width=True)
            st.dataframe(pe_df, use_container_width=True)

        elif os.path.exists(pe_img):
            st.image(pe_img, use_container_width=True)
        else:
            st.info("Partial evidence study not yet run. Execute: `python run_pipeline.py`")
            try:
                from src.evaluation import partial_evidence_degradation
                with st.spinner("Running partial evidence study (~30 s)…"):
                    pe_df = partial_evidence_degradation(
                        os.path.join(ROOT_DIR, "data", "processed", "bn_evidence_table.csv"),
                        n_samples=50,
                    )

                fig_pe = go.Figure()
                color_map = {"Accuracy": "#22c55e", "F1": "#818cf8", "AUC-ROC": "#f59e0b"}
                for metric, color in color_map.items():
                    if metric in pe_df.columns:
                        fig_pe.add_trace(go.Scatter(
                            x=pe_df["Features_Observed"], y=pe_df[metric],
                            mode="lines+markers+text",
                            name=metric,
                            line=dict(color=color, width=2.5),
                            marker=dict(size=9, color=color),
                            text=[f"{v:.2f}" for v in pe_df[metric]],
                            textposition="top center",
                            textfont=dict(size=10, color=color),
                        ))
                fig_pe.add_hline(y=0.5, line_dash="dash", line_color="#475569",
                                 annotation_text="Chance baseline",
                                 annotation_font_color="#94a3b8")
                fig_pe.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#e2e8f0"}, height=400,
                    xaxis_title="Features Observed", yaxis_title="Score",
                    yaxis_range=[0.3, 1.08],
                )
                st.plotly_chart(fig_pe, use_container_width=True)
                st.dataframe(pe_df, use_container_width=True)
            except Exception as e:
                st.error(f"Could not run study: {e}")


# ════════════════════════════════════════════════════════════════════════════
#   PAGE 5 — HOW IT WORKS

# ════════════════════════════════════════════════════════════════════════════
elif page == "How It Works":
    st.markdown("""
    <div class="hero-card">
      <h1>How FailNet Works</h1>
      <p>A step-by-step guide to the mathematics and design decisions behind the system.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("1. What is a Bayesian Network?", expanded=True):
        st.markdown("""
        A **Bayesian Network (BN)** is a Directed Acyclic Graph (DAG) where:
        - Each **node** represents a random variable (e.g., `Plan_Quality`, `Task_Failure`).
        - Each **directed edge** represents a causal or probabilistic dependency.
        - Each node stores a **Conditional Probability Table (CPT)** encoding `P(node | parents)`.

        The **Joint Probability Distribution** over all 24 nodes is factored as:
        ```
        P(X₁, X₂, ..., X₂₄) = ∏ P(Xᵢ | Parents(Xᵢ))
        ```
        This allows tractable inference even in large networks.
        """)

    with st.expander("2. The 24-Node Taxonomy (3 Tiers)"):
        st.markdown("""
        | Tier | Category | How it's obtained | Examples |
        |------|----------|-------------------|---------|
        | **Tier 1** | Directly Observable | Parsed deterministically from JSON logs | `Trajectory_Length`, `Execution_Error_Occurred` |
        | **Tier 2** | Weakly Derivable | Rule-based heuristics on structured fields | `Tool_Argument_Validity`, `Error_Recovery_Attempted` |
        | **Tier 3** | Proxy-Labeled (Latent) | LLM judge ratings from transcripts | `Plan_Quality`, `Instruction_Ambiguity`, `Hallucination_Risk` |
        """)

    with st.expander("3. Parameter Learning (Dirichlet / BDeu)"):
        st.markdown(r"""
        CPT parameters are learned from the **Who&When** dataset using **Bayesian Estimation with BDeu smoothing**:
        ```
        P(X = x | Parents(X) = y) = (N_{x,y} + α) / (N_y + α · K)
        ```
        - `α` = Equivalent Sample Size (ESS), chosen by K-Fold Cross-Validation.
        - **Why not MLE?** With ~200 training traces and 24 nodes, many CPT cells have zero counts.
          BDeu smoothing prevents zero-probability crashes during inference.
        """)

    with st.expander("4. Live Belief Update (Variable Elimination)"):
        st.markdown(r"""
        As evidence arrives, FailNet computes the **posterior marginal** via Variable Elimination:
        ```
        P(Task_Failure | e) = Σ_h P(Task_Failure, e, h) / Σ_{f,h} P(f, e, h)
        ```
        Unobserved hidden nodes **h** are marginalized out mathematically.
        This means the BN gracefully handles **partial evidence** — unlike XGBoost, which breaks.
        """)

    with st.expander("5. MAP Root Cause Diagnosis"):
        st.markdown(r"""
        When failure probability is high, a **MAP (Maximum A Posteriori) Query** finds the
        most probable configuration of hidden causal variables:
        ```
        h* = argmax_h P(h | e, Task_Failure = Yes)
        ```
        This directly answers: *"What is the most likely reason this agent is going to fail?"*  
        For example: `Instruction_Ambiguity=High → Plan_Quality=Flawed → Hallucination_Risk=High`
        """)

    with st.expander("6. Decision Network (Expected Utility)"):
        st.markdown(r"""
        FailNet converts the failure probability into an **action recommendation** using a
        Utility Matrix and Expected Utility maximization:
        ```
        EU(Action) = P(Fail|e) × U(Action, Fail) + P(No Fail|e) × U(Action, No Fail)
        ```
        | Action | U(·, No Fail) | U(·, Fail) | Rationale |
        |--------|--------------|-----------|-----------|
        | Continue | +50 | -20 | High reward if correct; big penalty if wrong |
        | Retry_Step | +15 | -5 | Moderate restart cost |
        | Escalate_to_Human | -2 | -2 | Fixed operator cost |
        | Halt | 0 | +5 | Saves damage if truly failing |

        The action with the **highest expected utility** is recommended.
        """)

    with st.expander("7. Value of Information (VoI)"):
        st.markdown("""
        FailNet can also recommend **which signal to query next** to maximally reduce decision uncertainty.

        The **Value of Information (VoI)** of a node X is:
        ```
        VoI(X) = EU*(after observing X) − EU*(before observing X)
        ```
        A high VoI means querying the agent about that property will change the recommended action the most.
        """)

    with st.expander("8. Why is FailNet better than XGBoost/Logistic Regression?"):
        st.markdown("""
        | Capability | Logistic Regression | XGBoost | FailNet (BN) |
        |---|---|---|---|
        | Handles Incomplete Evidence | No (requires imputation) | No | Yes (marginalizes) |
        | Root Cause Explanation | No | No | Yes (MAP inference) |
        | Intervention Recommendation | No | No | Yes (Expected Utility) |
        | Proactive (before crash) | No | No | Yes (live streaming) |
        | Interpretable Causal Structure | No | No | Yes (DAG) |
        """)

# ════════════════════════════════════════════════════════════════════════════
#   PAGE 6 — AGENT SIMULATOR
# ════════════════════════════════════════════════════════════════════════════
elif page == "Agent Simulator":
    import glob
    import json
    import time
    from simulate_live_integration import extract_evidence_from_partial, get_intervention

    st.markdown("""
    <div class="hero-card">
      <h1>Real-Time Agent Simulator</h1>
      <p>Simulate a live multi-agent session. FailNet monitors the growing conversation turn-by-turn, updates its Bayesian belief of failure, and recommends interventions in real-time.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        inference, dn = load_models()
    except Exception as e:
        st.error("Failed to load BN models. Run `python run_pipeline.py`.")
        st.stop()

    data_dir = os.path.join(ROOT_DIR, "data", "original data", "Who&When")
    json_files = glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True)
    if not json_files:
        st.warning("No JSON trajectory files found.")
        st.stop()

    selected_file = st.selectbox("Select an Agent Trajectory to Simulate", json_files, format_func=lambda x: os.path.relpath(x, data_dir))
    
    with open(selected_file, "r", encoding="utf-8") as f:
        traj_data = json.load(f)
        
    st.markdown(f"**Task:** {traj_data.get('question', 'Unknown')}")
    st.markdown(f"**Ground Truth:** {'FAILED' if not traj_data.get('is_correct', True) else 'SUCCESS'}")
    
    if st.button("Start Live Simulation"):
        history = traj_data.get("history", [])
        
        st.markdown("### Live Telemetry Stream")
        
        # UI Placeholders
        gauge_col, act_col = st.columns(2)
        with gauge_col:
            gauge_placeholder = st.empty()
        with act_col:
            act_placeholder = st.empty()
            
        chart_placeholder = st.empty()
        log_placeholder = st.empty()
        
        prob_history = []
        turn_labels = []
        logs = []
        
        for i, turn in enumerate(history):
            history_slice = history[:i+1]
            evidence = extract_evidence_from_partial(traj_data, history_slice)
            
            try:
                res = inference.posterior_failure(evidence)
                prob_fail = float(res.values[res.state_names["Task_Failure"].index("Yes")])
            except Exception:
                prob_fail = 0.50
                
            prob_history.append(prob_fail)
            turn_labels.append(f"Turn {i+1}")
            
            action_str, risk_label = get_intervention(prob_fail)
            
            speaker = turn.get("role", "Unknown")
            # Safe str representation
            raw_content = str(turn.get("content", ""))
            snippet = raw_content[:200].replace('\\n', ' ')
            logs.append(f"<b>Turn {i+1} ({speaker}):</b> {snippet}...")
            
            # Update Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_fail * 100,
                number={"suffix": "%", "font": {"size": 32}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#555"},
                    "bar": {"color": "#ef4444" if prob_fail > 0.65 else "#f59e0b" if prob_fail > 0.35 else "#22c55e"},
                    "bgcolor": "#1e293b", "borderwidth": 0,
                    "steps": [
                        {"range": [0, 35],  "color": "#14532d"},
                        {"range": [35, 65], "color": "#713f12"},
                        {"range": [65, 100],"color": "#7f1d1d"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": prob_fail * 100},
                },
                title={"text": "P(Task Failure | Evidence)", "font": {"size": 14}}
            ))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e2e8f0"}, height=240, margin=dict(t=30, b=10, l=20, r=20))
            gauge_placeholder.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{i}")
            
            # Update Action
            if "CONTINUE" in action_str:
                ac = "action-continue"
            elif "RETRY" in action_str:
                ac = "action-retry"
            elif "ESCALATE" in action_str:
                ac = "action-escalate"
            else:
                ac = "action-halt"
                
            act_placeholder.markdown(f"<div class='action-box {ac}' style='margin-top: 2rem;'>RECOMMENDED ACTION:<br>{action_str}</div>", unsafe_allow_html=True)
            
            # Update Line Chart
            fig_line = px.line(x=turn_labels, y=prob_history, markers=True, title="Live Belief Trajectory")
            fig_line.update_traces(line_color="#818cf8", marker=dict(size=8, color="#c4b5fd"))
            fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#e2e8f0"}, yaxis_range=[0, 1], yaxis_title="Probability of Failure", xaxis_title="Turn")
            chart_placeholder.plotly_chart(fig_line, use_container_width=True, key=f"line_{i}")
            
            # Update Logs
            log_html = "<div style='height: 300px; overflow-y: auto; background: #0f172a; padding: 1rem; border-radius: 8px; border: 1px solid #1e3a5f;'>"
            for log in reversed(logs):
                log_html += f"<div style='margin-bottom: 0.8rem; color: #cbd5e1; font-size: 0.9rem; line-height: 1.4;'>{log}</div>"
            log_html += "</div>"
            log_placeholder.markdown(log_html, unsafe_allow_html=True)
            
            time.sleep(1.0)
            
        st.success("Simulation Complete!")
        
        st.markdown("### Post-Mortem Analysis")
        final_evidence = extract_evidence_from_partial(traj_data, history)
        map_diag = inference.map_diagnosis(final_evidence)
        
        st.markdown("**Most Probable Causal Factors (MAP Inference):**")
        if map_diag:
            cols = st.columns(len(map_diag))
            for idx, (node, val) in enumerate(map_diag.items()):
                with cols[idx]:
                    st.markdown(f"**{node.replace('_', ' ')}**")
                    if val in ["Flawed", "Incorrect", "High", "Yes"]:
                        st.error(f"{val}")
                    else:
                        st.success(f"{val}")
        else:
            st.info("All causal factors are already observed in the evidence.")
                    
        st.markdown("### Markov Blanket of Task_Failure")
        try:
            mb_nodes = inference.get_markov_blanket("Task_Failure")
            st.info("The Markov Blanket shields the outcome from all other nodes. Only these nodes matter for predicting failure:")
            st.write(", ".join([n.replace('_', ' ') for n in mb_nodes]))
        except Exception as e:
            st.error(f"Could not compute Markov blanket: {e}")
