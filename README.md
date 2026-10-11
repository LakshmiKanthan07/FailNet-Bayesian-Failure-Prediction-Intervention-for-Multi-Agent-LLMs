# FailNet — Bayesian Failure Prediction & Intervention for Multi-Agent LLMs

> **An interpretable, causal alternative to black-box agent observability tools.**  
> Instead of logging *that* an agent failed, FailNet uses a **24-node Bayesian Network**
> to predict failures live, diagnose root causes (MAP inference), and recommend
> automated safety interventions via a **Decision Network**.

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![pgmpy](https://img.shields.io/badge/pgmpy-0.1.x-orange)](https://pgmpy.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📐 Architecture

```
[Raw Agent JSON Trajectories — Who&When dataset]
        ↓
[src/data_ingestion.py] ← [src/feature_engineering.py] ← [src/labeling.py]
  Parse JSON → Extract 24 BN features → Derive Task_Failure label
  + Generate realistic synthetic success rows (weighted sampling)
        ↓
[src/bn_structure.py]            [src/bn_learning.py]
  24-node DAG definition    →    BDeu Parameter Learning (K-Fold CV selects ESS)
        ↓
[src/inference.py] — FailNetInference
  ├── posterior_failure(evidence)   →  Variable Elimination (Exact Inference)
  ├── map_diagnosis(evidence)       →  MAP Query (argmax over hidden nodes)
  ├── get_markov_blanket(node)      →  Markov Blanket (BN theory)
  └── sensitivity_analysis()       →  Tornado Chart data
        ↓
[src/decision_network.py] — MockDecisionNetwork
  Expected Utility maximisation → Optimal action: Continue / Retry / Escalate / Halt
  + Decision boundary curves + Utility sensitivity
        ↓
[dashboard/app.py]  ←  streamlit run dashboard/app.py
  Live Monitor · Calibration Curve · Interactive DAG · Exact vs Approx toggle
```

---

## 🧠 Probabilistic Methods Used (Syllabus Aligned)

| Method | Module | Syllabus Topic |
|--------|--------|----------------|
| Bayesian Network (24-node DAG, CPDs) | `bn_structure.py`, `bn_learning.py` | Introduction to BNs, Representation Learning |
| BDeu / Dirichlet parameter learning (MLE comparison) | `bn_learning.py` | Learning in BNs |
| K-Fold CV for estimator selection | `bn_learning.py`, `exp8` | Learning in BNs |
| Variable Elimination — Exact marginals | `inference.py` | Exact Inference in BNs |
| MAP Inference — Root cause diagnosis | `inference.py` | Inference in BNs |
| Markov Blanket — Minimal observation set | `inference.py` | BN theory |
| Sensitivity Analysis / Tornado Chart | `inference.py`, `exp5` | Exact Inference queries |
| Decision Network — Expected Utility | `decision_network.py`, `exp8` | Decision Networks |
| Value of Information | `voi.py` | Decision Networks |

---

## 🚀 Quickstart

### Prerequisites
- Python 3.9+
- ~1 GB disk (dataset + model artifacts)

### Installation

```bash
git clone https://github.com/LakshmiKanthan07/FailNet-Bayesian-Failure-Prediction-Intervention-for-Multi-Agent-LLMs.git
cd FailNet-Bayesian-Failure-Prediction-Intervention-for-Multi-Agent-LLMs

python -m venv failnet-env

# Windows
.\failnet-env\Scripts\activate
# Unix/macOS
source failnet-env/bin/activate

pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python run_pipeline.py
```

This runs all 8 experiments end-to-end and saves figures to `report/figures/`.

### Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser.

---

## 📁 Module Reference

| File | Purpose |
|------|---------|
| `src/feature_engineering.py` | 17 feature extractor functions — maps raw JSON trajectory data to BN node categories |
| `src/labeling.py` | Label derivation (`is_correct → Task_Failure`), provenance tagging, sklearn encoding |
| `src/data_ingestion.py` | Full pipeline orchestrator — parse → feature extract → label → augment → save CSV |
| `src/bn_structure.py` | 24-node DAG definition (37 expert edges). Supports Hill-Climb / PC structure learning |
| `src/bn_learning.py` | Parameter learning (MLE vs BDeu) with K-Fold CV estimator selection |
| `src/inference.py` | `FailNetInference` — VE, LW, MAP, Markov Blanket, Sensitivity Analysis |
| `src/decision_network.py` | `MockDecisionNetwork` — EU computation, decision boundaries, utility sensitivity |
| `src/voi.py` | Value of Information computation for active diagnostic queries |
| `src/evaluation.py` | Cross-validation and partial evidence degradation evaluations |
| `experiments/exp5_evidence_uncertainty.py` | Streaming belief update + Tornado chart (sensitivity) |
| `experiments/exp8_decision_boundary.py` | Decision boundary curve + K-Fold CV for BN |
| `experiments/exp9_partial_evidence.py` | Partial evidence robustness study across unobserved fractions |
| `simulate_live_integration.py` | Turn-by-turn simulation engine for live agent trajectories |
| `dashboard/app.py` | Streamlit monitor & analytics dashboard (6 pages) |
| `tests/test_bn_structure.py` | Assert valid DAG, node/edge count |
| `tests/test_inference.py` | Assert posterior monotonicity, Markov Blanket, sensitivity shape |
| `tests/test_data_ingestion.py` | Assert no nulls, correct columns, class balance, label integrity |

---

## 📊 Key Results

Results are generated by `python run_pipeline.py` and saved to `report/`.

### BN vs Baseline (Single 80/20 Split)

| Model | Accuracy | F1 | Brier Score |
|-------|----------|----|----|
| FailNet (BN + VE) | — | — | — |
| Logistic Regression | — | — | — |
| XGBoost | — | — | — |

> *(Run `python run_pipeline.py` to populate these numbers for your dataset.)*

### BN 5-Fold Cross-Validation

| Metric | Mean | Std Dev |
|--------|------|---------|
| Accuracy | — | — |
| F1 Score | — | — |
| Brier Score | — | — |
| Log-Loss | — | — |

### Decision Boundaries

| Action pair | Threshold p* |
|-------------|-------------|
| Continue ↔ Retry_Step | ~0.39 |
| Continue ↔ Halt | ~0.71 |
| Retry_Step ↔ Halt | ~0.83 |

---

## 🗺️ Bayesian Network Structure

The 24-node DAG is structured across 6 tiers:

```
Tier 1 — Root Causes (no parents):
  Task_Complexity, Base_Model_Capability, Instruction_Ambiguity,
  Tool_Reliability, Environment_Volatility, Multi_Agent_Coordination_Required,
  Human_Oversight_Present, Trajectory_Length

Tier 2 — Planning:
  Required_Tool_Count → Planning_Complexity → Plan_Quality
  Context_Window_Load → Memory_Retrieval_Quality

Tier 3 — Execution:
  Tool_Selection_Correctness → Tool_Argument_Validity → Execution_Error_Occurred
  → Error_Recovery_Attempted → Error_Recovery_Success

Tier 4 — Hallucination & Verification:
  Memory_Retrieval_Quality → Hallucination_Risk
  Plan_Quality → Intermediate_Verification_Quality

Tier 5 — Multi-Agent:
  Multi_Agent_Coordination_Required → Inter_Agent_Communication_Quality
  Multi_Agent_Coordination_Required → Role_Assignment_Correctness

Tier 6 — Outcome:
  {Execution_Error_Occurred, Error_Recovery_Success, Hallucination_Risk,
   Intermediate_Verification_Quality, Inter_Agent_Communication_Quality,
   Role_Assignment_Correctness} → Task_Failure → Failure_Severity
```

**Markov Blanket of `Task_Failure`** (only nodes you ever need to observe):  
Run `inference.get_markov_blanket("Task_Failure")` to see the current set.

---

## 📈 Dashboard Features

| Page | What It Shows |
|------|--------------|
| 🏠 Overview | Problem statement, FailNet solution, architecture diagram |
| 🔬 Live Monitor | Evidence injection panel → posterior gauge → MAP diagnosis → VoI → Markov Blanket → Download JSON. **Exact vs Approx inference side-by-side.** |
| 📊 Dataset Stats | Class distribution, feature breakdowns, descriptive statistics |
| 📈 Results | Tornado chart, K-Fold CV, Decision boundaries, Streaming belief, **Calibration curve**, **Interactive DAG** |
| 📖 How It Works | Step-by-step math for BNs, parameter learning, VE, MAP, EU, VoI |

---

## 🔬 Running Tests

```bash
pytest tests/ -v
```

---

## Related Work & Differentiation

This project builds on the **Who&When** dataset (`ag2ai/Agents_Failure_Attribution`).

While tools like ProbGuard and VerifyMAS offer monitoring or post-hoc attribution, FailNet differentiates itself with an **interpretable causal factor structure** (Bayesian Network) combined with a **decision-theoretic intervention policy** (Decision Network), making it proactive, probabilistic, and explainable.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
