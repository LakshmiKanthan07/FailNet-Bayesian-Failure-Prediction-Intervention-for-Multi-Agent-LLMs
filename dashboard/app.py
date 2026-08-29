import streamlit as st
import pandas as pd
import numpy as np
import time
import os

# To import src modules from parent dir
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bn_learning import learn_parameters
from src.inference import FailNetInference
from src.decision_network import MockDecisionNetwork

st.set_page_config(page_title="FailNet Monitor", layout="wide")

@st.cache_resource
def load_models():
    model = learn_parameters(data_path="data/processed/bn_evidence_table.csv")
    inference = FailNetInference(model)
    dn = MockDecisionNetwork(inference)
    return inference, dn

st.title("FailNet — Diagnostic Failure Prediction Monitor")
st.markdown("Live monitoring dashboard for Multi-Agent Systems. Interpretable, causal alternative to black-box observability.")

try:
    inference, dn = load_models()
except Exception as e:
    st.error(f"Could not load model. Ensure data is generated. Error: {e}")
    st.stop()

# Sidebar for evidence injection
st.sidebar.header("Agent Telemetry Stream")
st.sidebar.markdown("Inject evidence into the Bayesian Network to see live belief updates.")

evidence = {}
if st.sidebar.checkbox("Task Complexity = High"):
    evidence["Task_Complexity"] = "High"
if st.sidebar.checkbox("Instruction Ambiguity = High"):
    evidence["Instruction_Ambiguity"] = "High"
if st.sidebar.checkbox("Execution Error Occurred = Yes"):
    evidence["Execution_Error_Occurred"] = "Yes"
if st.sidebar.checkbox("Hallucination Risk = High"):
    evidence["Hallucination_Risk"] = "High"
if st.sidebar.checkbox("Tool Reliability = Low"):
    evidence["Tool_Reliability"] = "Low"

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Posterior Belief Update")
    with st.spinner("Computing..."):
        res = inference.posterior_failure(evidence)
        prob_fail = res.values[res.state_names['Task_Failure'].index('Yes')]
        
    st.metric("P(Task_Failure = Yes | Evidence)", f"{prob_fail:.2%}")
    st.progress(float(prob_fail))
    
    st.markdown("### Current Evidence:")
    st.json(evidence if evidence else {"(None)": "Priors active"})

with col2:
    st.subheader("2. MAP Diagnosis & Intervention")
    with st.spinner("Diagnosing..."):
        map_diag = inference.map_diagnosis(evidence)
        opt_action, eus = dn.get_optimal_policy(evidence)
    
    st.markdown("**Most Probable Explanation (MAP):**")
    st.json(map_diag)
    
    st.markdown("**Recommended Policy Intervention:**")
    st.success(f"Action: {opt_action}")
    
    st.markdown("Expected Utilities:")
    st.bar_chart(pd.DataFrame.from_dict(eus, orient='index', columns=['Utility']))
