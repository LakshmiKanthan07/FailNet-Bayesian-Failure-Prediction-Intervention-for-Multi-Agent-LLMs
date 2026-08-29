# FailNet — Bayesian Failure Prediction & Intervention for Multi-Agent LLMs

FailNet is an interpretable, causal alternative to black-box agent observability tools. Instead of simply logging *that* an agent failed, FailNet uses a 25-node Bayesian Network to predict failures dynamically and diagnose the root cause (e.g., Instruction Ambiguity, Flawed Plan Quality, Incorrect Tool Selection).

It provides a live monitoring and decision-theoretic intervention policy layer for Multi-Agent LLM systems.

## Architecture

FailNet models the agent execution process through a Directed Acyclic Graph (DAG) consisting of 25 categorical nodes divided into three tiers:
1. **Directly Observable:** e.g., Execution Error Occurred, Trajectory Length.
2. **Weakly Derivable (Heuristic):** e.g., Tool Argument Validity, Error Recovery Attempted.
3. **Proxy-labeled (LLM-assisted):** e.g., Instruction Ambiguity, Plan Quality, Hallucination Risk.

## Key Features

- **Prior Marginal (Base Rate):** Computes the inherent risk of a task failing before the agent takes any action.
- **Live Belief Update:** Re-evaluates `P(Task_Failure | Evidence)` dynamically as evidence from the trace is streamed in.
- **MAP Root Cause Diagnosis:** When a failure occurs, performs Maximum A Posteriori (MAP) inference to infer the most probable sequence of hidden states that caused the crash.
- **Decision Network:** Computes expected utilities to recommend actions like `Continue`, `Retry_Step`, `Escalate_to_Human`, or `Halt`.

## Quickstart

### Prerequisites
- Python 3.9+
- A virtual environment (recommended)

### Installation
```bash
git clone https://github.com/LakshmiKanthan07/FailNet-Bayesian-Failure-Prediction-Intervention-for-Multi-Agent-LLMs.git
cd FailNet-Bayesian-Failure-Prediction-Intervention-for-Multi-Agent-LLMs
python -m venv failnet-env

# Windows
.\failnet-env\Scripts\activate
# Unix/MacOS
source failnet-env/bin/activate

pip install -r requirements.txt
```

### Running the Pipeline

You can run the entire pipeline end-to-end (Data generation $\rightarrow$ Parameter Learning $\rightarrow$ Evaluation) using:

```bash
python run_pipeline.py
```
*(Alternatively, use `make pipeline` if you have `make` installed).*

### Live Dashboard

To launch the Streamlit monitoring dashboard:

```bash
streamlit run dashboard/app.py
```

## Related Work & Differentiation

This project builds on datasets like **Who&When** (`ag2ai/Agents_Failure_Attribution`), **AEGIS**, and **MAST-Data**. 

While tools like ProbGuard and VerifyMAS offer monitoring or post-hoc attribution, FailNet differentiates itself by providing an **interpretable causal factor structure** coupled with a **decision-theoretic intervention policy**, rather than treating failure observability as a standard state-machine or black-box classification problem.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
