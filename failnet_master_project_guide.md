# 📚 FailNet: Master Project Guide

## 1. Executive Summary & One-Line Pitch
> **"FailNet is an interpretable, causal alternative to black-box agent observability tools. It uses a 25-node Bayesian Network to dynamically predict multi-agent LLM failures, diagnose their root causes using MAP inference, and recommend automated safety interventions based on Expected Utility."**

* **Project Title:** FailNet — Bayesian Failure Prediction & Intervention for Multi-Agent LLMs
* **Domain:** Probabilistic Graphical Models, AI Observability, Multi-Agent Systems (LLMs)
* **Key Innovation:** Moving AI agent monitoring from **reactive black-box logging** to **proactive, causal probabilistic reasoning**.

---

## 2. Problem Statement & Motivation
### The Problem: Multi-Agent Systems are Unpredictable
As AI shifts from single LLM prompts to autonomous multi-agent frameworks (e.g., AutoGen, LangGraph, CrewAI), multiple agents collaborate (e.g., Planner → Coder → Executor). 
* **Cascading Errors:** If a Planner agent makes a slight logic error, the Coder agent assumes it is truth, writes bad code, and the Executor crashes. Errors compound exponentially across turns.
* **Cost & Safety Risk:** Unmonitored agents enter infinite loops, hallucinate factual claims, or execute dangerous API calls—silently burning thousands of dollars in tokens or corrupting databases.

### The Current Gap: Observability is a "Black Box"
Existing monitoring tools (like LangSmith, Arize, or XGBoost classifiers) have two massive flaws:
1. **They are Reactive:** They log an error *after* the crash has already occurred.
2. **They are Uninterpretable:** A model like XGBoost outputs a single number ($P(\text{fail}) = 0.85$) without explaining *why* it failed, and breaks if any telemetry data is missing.

---

## 3. The FailNet Solution Architecture
FailNet models the multi-agent execution pipeline as a **25-node Directed Acyclic Graph (DAG) Bayesian Network**.

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Telemetry Ingestion & Feature Extraction                 │
│    - Parses raw conversation JSON into 25 discrete nodes    │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 25-Node Causal Bayesian Network ([bn_structure.py])      │
│    - Tier 1: Directly Observable (e.g., Trajectory Length)  │
│    - Tier 2: Weakly Derivable (e.g., Tool Argument Schema)  │
│    - Tier 3: Proxy-Labeled / Hidden (e.g., Plan Quality)    │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Dirichlet Parameter Learning ([bn_learning.py])          │
│    - Fits CPTs with K2 Prior Smoothing ( ESS = 5 )          │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Probabilistic Inference Engine ([inference.py])          │
│    - Live Risk: P(Task_Failure | Evidence) via VarElim      │
│    - Root Cause: MAP Query argmax P(Plan, Tool, Halluc | e) │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Decision Network & Intervention Policy ([decision_network])│
│    - Computes Expected Utility EU(Action)                   │
│    - Recommends: Continue | Retry_Step | Escalate | Halt    │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Live Dashboard & Benchmarks ([dashboard/app.py])         │
└──────────────────────────────┴──────────────────────────────┘
```

---

## 4. The 3-Tier Node Taxonomy (25 Nodes Total)
The 25 nodes in `src/bn_structure.py` are divided into three distinct categories based on data origin:

| Tier | Category | Description | Examples from Graph |
| :--- | :--- | :--- | :--- |
| **Tier 1** | **Directly Observable** | Objective facts measured directly from system JSON logs with 100% certainty. | `Trajectory_Length`, `Execution_Error_Occurred`, `Required_Tool_Count`, `Human_Oversight_Present` |
| **Tier 2** | **Weakly Derivable** | Heuristic rules/scripts evaluating mechanics (e.g., JSON schema validation). | `Tool_Argument_Validity`, `Error_Recovery_Attempted`, `Multi_Agent_Coordination_Required` |
| **Tier 3** | **Proxy-Labeled (Latent)** | Cognitive/root-cause states evaluated by an LLM Judge or dataset annotations. | `Plan_Quality`, `Instruction_Ambiguity`, `Hallucination_Risk`, `Inter_Agent_Communication_Quality` |

---

## 5. Mathematical Foundations & Core Formulas

### A. Joint Probability Distribution (Chain Rule)
$$ P(X_1, X_2, \dots, X_{25}) = \prod_{i=1}^{25} P(X_i \mid \text{Parents}(X_i)) $$

### B. Dirichlet Parameter Estimation (K2 Smoothing)
To avoid zero-probability crashes on unobserved combinations in sparse trace data:
$$ P(X = x \mid \text{Parents}(X) = y) = \frac{N_{x,y} + \alpha}{N_y + \alpha \cdot K} $$
*(Where $\alpha = 5$ is the Equivalent Sample Size).*

### C. Live Posterior Probability & Marginalization (Variable Elimination)
$$ P(\text{Task\_Failure} \mid \mathbf{e}) = \frac{\sum_{\mathbf{h} \in \mathbf{H}} P(\text{Task\_Failure}, \mathbf{e}, \mathbf{h})}{\sum_{f \in \{\text{Yes, No}\}} \sum_{\mathbf{h} \in \mathbf{H}} P(f, \mathbf{e}, \mathbf{h})} $$
*(Marginalizes out unobserved hidden nodes $\mathbf{H}$ dynamically).*

### D. Maximum A Posteriori (MAP) Root Cause Diagnosis
Identifies the most probable hidden cause of a crash:
$$ \mathbf{h}^* = \arg\max_{\mathbf{h}} P(\mathbf{h} \mid \mathbf{e}, \text{Task\_Failure} = \text{Yes}) $$

### E. Expected Utility Policy Decision
$$ EU(\text{Action}) = \sum_{s \in \{\text{Fail}, \text{Success}\}} P(s \mid \mathbf{e}) \cdot U(\text{Action}, s) $$
* Utility Matrix: `Continue` on Failure = $-20$, `Halt` on Failure = $+5$, `Continue` on Success = $+10$.

---

## 6. Codebase Architecture & File Responsibilities

```
project/
├── data/
│   ├── raw/who_and_when/     # Cloned real-world Who&When JSON execution dataset
│   └── processed/            # Final 25-node evidence table (bn_evidence_table.csv)
├── src/
│   ├── bn_structure.py       # Defines 25-node DAG network topology using pgmpy
│   ├── data_ingestion.py     # Generates/parses multi-agent traces into categorical table
│   ├── bn_learning.py        # Bayesian Estimator parameter fitting (CPTs + K2 prior)
│   ├── inference.py          # Variable Elimination, MAP diagnosis, Gibbs sampling
│   ├── decision_network.py   # Expected Utility engine for action policies
│   ├── baselines.py          # Logistic Regression & XGBoost benchmark models
│   └── evaluate.py           # Log-Loss, Brier Score, ECE, F1 metric implementations
├── experiments/
│   └── exp1_bn_vs_baseline.py# Comparative experiment runner
├── dashboard/
│   └── app.py                # Streamlit live monitoring web interface
└── run_pipeline.py           # Orchestrator script for end-to-end execution
```

---

## 7. Datasets Used

1. **Synthetic Multi-Agent Trajectories (`data/processed/bn_evidence_table.csv`):**
   * 2,000 synthetic samples generated via `src/data_ingestion.py` to validate parameter learning, Variable Elimination, and the dashboard end-to-end.
2. **Who & When Dataset (`data/raw/who_and_when`):**
   * Real-world dataset of 126+ multi-agent execution traces (CaptainAgent & Magnetic-One runs) containing JSON step transcripts, mistake agents, mistake steps, and failure reasons.

---

## 8. Model Evaluation & Baseline Comparison

We compare FailNet against standard discriminative machine learning baselines (**Logistic Regression** & **XGBoost**):

| Model | Log-Loss | Brier Score | Interpretability (Root Cause) | Handles Missing Telemetry? |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | ~0.40 | — | ❌ No | ❌ No (Requires imputation) |
| **XGBoost** | ~0.44 | — | ❌ No | ❌ No (Requires imputation) |
| **FailNet (Bayesian Network)** | ~0.63 | **0.22** | **YES (via MAP Query)** | **YES (via Marginalization)** |

### Why FailNet is Superior in Real-World Operations:
1. **Handles Incomplete Evidence:** If an agent is at Step 2 and only 3 nodes are observed, XGBoost breaks or requires fake feature imputation. FailNet marginalizes out unobserved nodes mathematically.
2. **Root-Cause Attribution:** XGBoost gives a single scalar ("80% fail"). FailNet isolates the exact failing component (`Plan_Quality = Flawed` → `Hallucination_Risk = High`).
3. **Decision Utility:** FailNet maps probabilities directly to actionable safety decisions (`Halt`, `Retry`, `Escalate`).

---

## 9. How to Execute the Entire Pipeline

```bash
# 1. Activate environment
.\failnet-env\Scripts\activate

# 2. Run full pipeline (Data Generation -> Training -> Evaluation)
python run_pipeline.py

# 3. Launch live dashboard
streamlit run dashboard/app.py
```
