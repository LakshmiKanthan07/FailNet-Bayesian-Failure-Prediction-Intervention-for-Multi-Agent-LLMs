# FailNet — Implementation Workflow
## A Bayesian Network for Diagnostic Failure Prediction & Intervention Policy in Multi-Agent LLM Systems

This is the build plan: environment → data → model → inference → decision layer → experiments → packaging. Every phase below flags **why it matters for novelty** and **why it matters for industry impact**, so those two threads stay visible while you're heads-down in code, not just in the pitch deck.

**One-line positioning to keep in mind throughout:** you are building an *interpretable, causal alternative to black-box agent observability tools* (the LangSmith/Arize/Galileo category) — a risk-scoring and diagnosis layer that can say *why* an agent is likely to fail, not just *that* it might.

---

## Phase 0 — Environment & Tooling Setup

**Goal:** a reproducible environment before any modeling starts.

```bash
python -m venv failnet-env
source failnet-env/bin/activate
pip install pgmpy pandas numpy networkx matplotlib seaborn scikit-learn xgboost \
            jupyterlab pyyaml tqdm --break-system-packages
pip install pycid --break-system-packages   # decision networks / influence diagrams, built on pgmpy + networkx
pip install streamlit plotly --break-system-packages  # for the monitoring-dashboard demo later
```

- **pgmpy** — core BN structure, CPD estimation, exact inference (Variable Elimination, Belief Propagation), approximate inference (Gibbs/MCMC sampling).
- **pycid** — built directly on pgmpy + NetworkX, purpose-built for causal influence diagrams (decision networks): defines chance/decision/utility nodes and computes optimal policies. This is what turns your BN into the decision-network deliverable from Step 5/8 of the project plan.
- **xgboost / scikit-learn** — the black-box baselines your central experiment (Experiment 1) needs to beat or match.
- **streamlit** — optional, but this is the single highest-leverage addition for the "industry impact" pitch: a live dashboard showing streaming risk updates looks like a real product demo, not a course assignment.

Repository skeleton to create now, before writing any logic:

```
failnet/
├── data/
│   ├── raw/                 # untouched downloaded datasets
│   ├── interim/             # parsed trajectory JSON → flat tables
│   └── processed/           # final categorical evidence tables (BN-ready)
├── notebooks/                # exploration only — nothing load-bearing lives here
├── src/
│   ├── data_ingestion.py
│   ├── feature_engineering.py
│   ├── labeling.py           # proxy-label logic + audit trail
│   ├── bn_structure.py       # DAG definition
│   ├── bn_learning.py        # CPD/parameter learning
│   ├── inference.py          # VE, MAP, MCMC, belief propagation wrappers
│   ├── decision_network.py   # pycid influence diagram
│   ├── baselines.py          # logistic regression / XGBoost
│   └── evaluate.py           # metrics from Step 10
├── experiments/
│   ├── exp1_bn_vs_baseline.py
│   ├── exp2_exact_vs_approx.py
│   ├── exp3_map_diagnosis.py
│   ├── exp4_network_size.py
│   └── exp5_evidence_uncertainty.py
├── dashboard/
│   └── app.py                # streamlit live-monitoring demo
├── report/
├── tests/
├── README.md
└── requirements.txt
```

Building this skeleton on day one — before the fun modeling work — is what keeps the "convert messy logs into 25 clean node values" bottleneck (identified as the #1 risk earlier) from turning into an unstructured mess halfway through the semester.

---

## Phase 1 — Data Acquisition

**Goal:** get the labeled failure data before writing a single line of BN code.

| Dataset | Where | What you get |
|---|---|---|
| **Who&When** | GitHub: `ag2ai/Agents_Failure_Attribution` · Hugging Face: `Kevin355/Who_and_When` (MIT license) | 184 annotated failure tasks across 127 multi-agent systems (CaptainAgent-generated + hand-crafted Magnetic-One runs), each with the responsible agent, the decisive error step, and a natural-language failure explanation, split into `Algorithm-Generated` and `Hand-Crafted` subfolders (one JSON per task). |
| **AEGIS** | Referenced in recent failure-attribution papers (arXiv 2605.17467 and related) | ~9,533 trajectories, 24,843 annotated errors across MATH/SciBench/GSM8K/HumanEval/MMLU/GAIA and six MAS frameworks — your main source of volume once Who&When is exhausted. |
| **MAST-Data** | Companion to the MAST failure taxonomy paper | 1,642 annotated traces across seven multi-agent frameworks, with a 14-mode failure taxonomy (system design / inter-agent misalignment / task verification) — extremely useful as a *label vocabulary* for several of your 25 nodes. |
| **TRAIL** | GAIA-based traces | 118 traces, ~8.3 spans average, useful as an additional evaluation-only holdout set (don't train on it, use it to test generalization). |
| **Successful trajectories (for base rate)** | AgentBench / raw GAIA runs / Magnetic-One success logs | Needed because Who&When contains *only* failures — see the class-imbalance fix below. |

**Action items:**
1. Clone/download Who&When first — it's the smallest and best-annotated, so it's where you validate your parsing pipeline before scaling to AEGIS.
2. Write `data_ingestion.py` to normalize the differing JSON schemas (Who&When's `Algorithm-Generated` vs. `Hand-Crafted` subsets already have minor field differences — e.g., `is_correct`/`level` appear only in one) into a single internal schema.
3. Pull in a matched set of *successful* trajectories from the same underlying frameworks (AgentBench, GAIA runs) so "Task Failure" has real negative examples — training only on failure logs with no successes is the single most likely way this project silently breaks.

---

## Phase 2 — Feature Engineering & Labeling Pipeline (the real bottleneck — budget the most time here)

**Goal:** turn free-text trajectories into the 25 categorical node values from the design table, with an honest record of which are directly observed vs. proxy-labeled.

**2.1 — Split nodes into three tiers by how they're obtained.** Don't skip this — it's exactly what you'll be asked about in the viva, so build the audit trail into the code from day one, not as an afterthought.

| Tier | Examples | Method |
|---|---|---|
| **Directly observable** | Trajectory Length, Required Tool Count, Execution Error Occurred, Task Domain | Deterministic parsing/counting from the raw trajectory JSON. |
| **Weakly derivable (heuristic)** | Tool Argument Validity, Error Recovery Attempted, Multi-Agent Coordination Required | Rule-based heuristics over structured fields (tool-call schemas, retry patterns, number of distinct agent roles). |
| **Proxy-labeled (LLM-assisted)** | Instruction Ambiguity, Plan Quality, Intermediate Verification Quality, Hallucination Risk | Use an LLM labeler **that did not generate the trajectory** (critical — see risk mitigation) to rate these from the transcript, with a fixed rubric and a small hand-audited sample. |

**2.2 — Labeling protocol to avoid circularity (this is the make-or-break step):**
- If a trajectory was generated by GPT-4o/Claude/etc., label the proxy nodes with a *different* model family, and document this explicitly in the report.
- Hand-audit a random 10–15% sample of proxy labels yourself against the natural-language `mistake_reason` field Who&When already provides — this doubles as free validation data.
- Log every labeling decision with the source (`heuristic:tool_schema_check`, `llm_proxy:claude-labeler-v1`, `human_audit`) in a `label_provenance` column. This single column is what turns "trust me" into "here's the receipt" when a professor pushes on data quality.

**2.3 — Output:** `data/processed/bn_evidence_table.csv` — one row per trajectory, 25 categorical columns + `label_provenance` metadata + the original task/dataset identifiers for traceability.

**Novelty note:** this tiered, provenance-tracked labeling pipeline is itself a small methodological contribution worth writing up separately in the report — most published failure-attribution work treats labels as ground truth without discussing how uncertain those labels really are. Being explicit about it is a strength, not a weakness, if framed correctly.

---

## Phase 3 — Bayesian Network Structure Definition

**Goal:** encode the 25-node DAG from the design phase in pgmpy.

```python
# src/bn_structure.py
from pgmpy.models import BayesianNetwork

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

failnet_model = BayesianNetwork(failnet_edges)
assert failnet_model.check_model.__doc__ is not None  # placeholder — real check runs after CPDs are attached
```

**Validation checklist before moving on:**
- Run `networkx.is_directed_acyclic_graph()` on the edge list — pgmpy will also reject cycles, but check early.
- Compute the network's **treewidth** (via `pgmpy`'s junction-tree construction, or `networkx` triangulation heuristics) — this number directly justifies your Experiment 2/4 claims about when exact inference (VE) becomes expensive relative to MCMC. Report it in the paper; "our network has treewidth *k*" is exactly the kind of number a probabilistic-reasoning viva rewards.
- Sanity-check every edge against a real trajectory example — for each parent→child link, be able to point to a concrete log snippet that illustrates the dependency. This is your defense against the "meaningless nodes just to hit 20" criticism.

---

## Phase 4 — Parameter Learning (CPDs)

**Goal:** learn CPTs from `bn_evidence_table.csv` rather than hand-specifying them.

```python
# src/bn_learning.py
from pgmpy.estimators import BayesianEstimator
import pandas as pd

data = pd.read_csv("data/processed/bn_evidence_table.csv")
failnet_model.fit(
    data,
    estimator=BayesianEstimator,
    prior_type="K2",          # Dirichlet/K2 smoothing — critical given sparse cells
    equivalent_sample_size=5, # tune per node if some have very few examples
)
```

- Use **Bayesian estimation with K2 (or BDeu) smoothing**, not raw MLE — with only ~184–2,000 labeled trajectories spread across 25 nodes, many CPT cells will have zero or near-zero counts, and MLE will produce brittle zero-probability entries that break downstream inference. This is not optional.
- After fitting, run `failnet_model.check_model()` to confirm every CPD is properly normalized and consistent with the graph.
- For nodes with very sparse evidence (e.g., rare combinations of Multi-Agent Coordination + Role Assignment states), consider merging categories (e.g., collapsing a 4-level severity scale to 3) rather than leaving cells to be smoothed into near-uniform noise — report this decision explicitly.

---

## Phase 5 — Inference Engine

**Goal:** implement the four inference modes the project's novelty rests on.

```python
# src/inference.py
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.sampling import GibbsSampling

ve = VariableElimination(failnet_model)
bp = BeliefPropagation(failnet_model)

def posterior_failure(evidence: dict):
    """Exact marginal P(Task_Failure | evidence) via Variable Elimination."""
    return ve.query(variables=["Task_Failure"], evidence=evidence, show_progress=False)

def map_diagnosis(evidence: dict):
    """Most probable causal configuration given an observed failure."""
    return ve.map_query(
        variables=["Plan_Quality", "Tool_Selection_Correctness", "Hallucination_Risk",
                   "Intermediate_Verification_Quality"],
        evidence=evidence, show_progress=False,
    )

def approximate_posterior(evidence: dict, n_samples: int = 5000):
    """Gibbs-sampling MCMC posterior — for latent/proxy-labeled nodes and larger evidence sets."""
    gibbs = GibbsSampling(failnet_model)
    samples = gibbs.sample(size=n_samples)
    for var, val in evidence.items():
        samples = samples[samples[var] == val]
    return samples["Task_Failure"].value_counts(normalize=True)

def streaming_belief_update(evidence_sequence: list[dict]):
    """Belief propagation re-run as each new evidence item arrives — the monitoring use case."""
    beliefs = []
    accumulated = {}
    for step_evidence in evidence_sequence:
        accumulated.update(step_evidence)
        beliefs.append(bp.query(variables=["Task_Failure"], evidence=accumulated, show_progress=False))
    return beliefs
```

**Convergence diagnostics for MCMC (do not skip — this is graded/viva-checked material):** track trace plots per sampled variable, compute Gelman–Rubin R-hat across multiple chains, and report effective sample size. A single unchecked Gibbs run is the fastest way to lose credibility on the "why do you need MCMC" question.

---

## Phase 6 — Decision Network / Intervention Policy Layer

**Goal:** the piece that differentiates this from a plain classifier — turn a probability into an action.

```python
# src/decision_network.py
from pycid import CID

decision_nodes = ["Intervention_Action"]  # {Continue, Retry_Step, Escalate_to_Human, Halt}
utility_nodes = ["System_Utility"]

cid_edges = failnet_edges + [
    ("Task_Failure", "System_Utility"),
    ("Failure_Severity", "System_Utility"),
    ("Intervention_Action", "System_Utility"),
    ("Task_Failure", "Intervention_Action"),        # decision informed by current belief
    ("Failure_Severity", "Intervention_Action"),
]

failnet_cid = CID(cid_edges, decisions=decision_nodes, utilities=utility_nodes)
# Attach CPDs for chance nodes (reuse failnet_model's), define a utility table
# over (Task_Failure, Failure_Severity, Intervention_Action) reflecting the
# real cost trade-off: false halts waste work, missed halts risk damage.
optimal_policy = failnet_cid.solve()  # pycid computes the expected-utility-maximizing policy
```

- Design the **utility table** carefully and defend the numbers in the report: e.g., `Continue`+`No Failure` = +10, `Halt`+`Critical Failure avoided` = +5 (saved harm minus lost progress), `Continue`+`Critical Failure` = −20, `Escalate`+`Minor Failure` = −2 (unnecessary human cost). These numbers should be justified qualitatively (e.g., "a false halt costs ~1 wasted agent step; an unflagged critical failure costs a full task redo plus trust damage"), not asserted arbitrarily.
- This is the artifact that answers "so what should the system actually do?" — screenshot/diagram this decision network for the report; it's the most demo-able part of the whole project.

---

## Phase 7 — Baselines

**Goal:** the comparison Experiment 1 needs.

```python
# src/baselines.py
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.get_dummies(data.drop(columns=["Failure_Severity"]))  # one-hot the same 25 features
X, y = df.drop(columns=["Task_Failure"]), df["Task_Failure"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
xgb = XGBClassifier(eval_metric="logloss").fit(X_train, y_train)
```

Same feature set, same train/test split as the BN, so the comparison in Experiment 1 is apples-to-apples.

---

## Phase 8 — Running the Five Experiments

Map each experiment script directly to the design from the evaluation report:

| Script | What it produces |
|---|---|
| `exp1_bn_vs_baseline.py` | F1/log loss/Brier/ECE table comparing BN, logistic regression, XGBoost |
| `exp2_exact_vs_approx.py` | Runtime + KL-divergence curves, VE vs. Gibbs MCMC, across 0/30/60/100% evidence |
| `exp3_map_diagnosis.py` | MAP-query agreement rate vs. Who&When's human-annotated `mistake_agent`/`mistake_step` |
| `exp4_network_size.py` | Accuracy/runtime/parameter-count comparison across 10-node, 15-node (pruned), 25-node variants |
| `exp5_evidence_uncertainty.py` | Early-warning lead-time curve under 20/50/80% evidence masking, via `streaming_belief_update` |

Run all five, dump results to `report/results/*.csv`, and generate the plots directly from those CSVs (not from re-running experiments interactively) so results are reproducible from a single `make experiments` step.

---

## Phase 9 — Evaluation & Metrics Implementation

Implement each metric from the design report as a small, tested function in `evaluate.py`: F1/precision/recall, log loss, Brier score, Expected Calibration Error (bin predicted probabilities into deciles, compare to observed frequency), MAP agreement rate, KL divergence between exact and approximate posteriors, and MCMC diagnostics (R-hat via multiple chains). Write unit tests for at least ECE and KL divergence — these are the two most commonly miscomputed in student projects, and a wrong metric undermines every downstream claim.

---

## Phase 10 — Visualization & the "Industry Impact" Demo

This phase is where the project stops looking like a course assignment and starts looking like a prototype tool — worth real time investment.

1. **Static network diagram:** render the 25-node DAG with `networkx` + `matplotlib` (or `graphviz` via pgmpy's export), color-coded by tier (observed / heuristic / proxy-labeled) from Phase 2. Include this in both the report and the README.
2. **Live monitoring dashboard (`dashboard/app.py`, Streamlit):** feed a trajectory step-by-step, show the streaming `Task_Failure` belief update live (Phase 5's `streaming_belief_update`), and surface the current MAP diagnosis and recommended `Intervention_Action` side-by-side. This single demo is what makes the GitHub repo and the resume line land — it's the visual proof that this isn't just an offline classifier, it's a monitor.
3. **Explanation view:** for any flagged trajectory, show the MAP-inferred causal chain as a highlighted subgraph ("Instruction Ambiguity = High → Plan Quality = Flawed → Tool Selection = Incorrect → Task Failure"). This is the concrete artifact that operationalizes the "interpretability" claim from your novelty statement — don't just assert interpretability, show it.

---

## Phase 11 — Packaging as a Resume/GitHub Project

- **README.md** structure: one-paragraph pitch (the observability-tool framing), architecture diagram, quickstart (`pip install -r requirements.txt && python -m src.pipeline`), sample dashboard screenshot/GIF, results summary table (Experiment 1's headline numbers), and a "Related Work & How This Differs" section explicitly citing ProbGuard/Pro²Guard, Who&When, and VerifyMAS — this is exactly the differentiation argument from the novelty claim, made public.
- Pin dependency versions in `requirements.txt`; include a `Makefile` or `justfile` with `make data`, `make train`, `make experiments`, `make dashboard` targets so a stranger can reproduce results in under 10 minutes.
- MIT or Apache-2.0 license (Who&When itself is MIT — keep attribution).
- Add a `CITATION.cff` or a references section crediting Who&When/AEGIS/MAST-Data explicitly — both for academic honesty and because reviewers/recruiters checking data provenance is a real signal of rigor.

---

## Phase 12 — Report & Viva Preparation

Structure the written report to mirror this workflow (Data → Structure → Learning → Inference → Decision Network → Experiments → Results → Limitations), and prepare short, rehearsed answers to the questions the evaluation report already flagged:

| Likely question | Your answer, in one line |
|---|---|
| "Why a BN, not a neural net?" | Interpretability + causal MAP diagnosis a classifier can't give you. |
| "Why 20+ nodes?" | Point to Experiment 4's accuracy/runtime curve — empirical, not a syllabus requirement. |
| "Why MCMC?" | Point to the treewidth number and Experiment 2's exact-vs-approximate crossover. |
| "Where's the novelty, given ProbGuard/Who&When already exist?" | Different paradigm — interpretable causal factor structure + decision-theoretic intervention, not state-machine monitoring or post-hoc attribution. |
| "How reliable are your labels?" | Show the tiered provenance system from Phase 2 and the hand-audit numbers. |
| "What happens if your independence assumptions are wrong?" | Discuss as a limitation: some edges (e.g., Base_Model_Capability fanning into six children) assume conditional independence given the parent that real trajectories may violate; mention this was checked qualitatively against sampled logs, and propose structure-learning (e.g., Hill-Climb search with a BIC score) as future validation. |

---

## Suggested Semester Timeline (14 weeks)

| Weeks | Focus |
|---|---|
| 1–2 | Phase 0–1: environment, repo skeleton, download and parse Who&When |
| 3–5 | Phase 2: feature engineering + tiered labeling pipeline (biggest time sink — protect this window) |
| 6 | Phase 3: finalize and validate the 25-node DAG structure |
| 7 | Phase 4: parameter learning, CPD sanity checks |
| 8 | Phase 5: inference engine (VE, MAP, MCMC, belief propagation) |
| 9 | Phase 6: decision network / pycid integration |
| 10 | Phase 7–8: baselines + run all 5 experiments |
| 11 | Phase 9: metrics, result tables, plots |
| 12 | Phase 10: dashboard + visualizations |
| 13 | Phase 11: repo packaging, README, demo polish |
| 14 | Phase 12: report writing, viva rehearsal |

Build in a buffer: if Phase 2 overruns (it usually does), the dashboard (Phase 10) is the first thing to trim, not the experiments or metrics — those are what get graded and defended.

---

## Risk Register (carried forward and expanded)

| Risk | Mitigation |
|---|---|
| Labeling circularity (same model generates and labels data) | Use a different model family as labeler; document explicitly (Phase 2). |
| Class imbalance (Who&When has only failures) | Mix in successful trajectories from the same frameworks (Phase 1). |
| Sparse CPT cells from limited data | Bayesian estimation with K2/BDeu smoothing, not raw MLE (Phase 4). |
| Unchecked MCMC convergence | Multi-chain R-hat + trace plots as a required deliverable, not optional (Phase 5). |
| Novelty challenged in viva | Explicit "Related Work & How This Differs" section citing ProbGuard/Who&When/VerifyMAS by name (Phase 11). |
| Project scope creep into building agent infrastructure | Hard rule: no new agent framework is built; only existing trajectory datasets are used. |

---
## Stretch Goals (if time remains — highest additional novelty/impact per unit effort)
1. **Structure learning validation:** run a Hill-Climb search (BIC/K2 score) over the evidence table and compare the learned structure to the hand-designed DAG — a strong addition to the "why these edges" defense.
2. **Live agent-framework hook:** a thin adapter that ingests real-time trace events from a framework like LangGraph or AutoGen/AG2 and feeds them into `streaming_belief_update` — turns the dashboard from a replay demo into something that could genuinely sit in front of a running agent.
3. **Cross-dataset generalization test:** train on Who&When + AEGIS, evaluate zero-shot on TRAIL, and report the accuracy drop — directly strengthens the Research Potential and Impact claims from the evaluation report.