import sys
import os
import networkx as nx
import matplotlib.pyplot as plt

# Add the project root to path so we can import src.bn_structure if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.bn_structure import get_failnet_model

def visualize():
    model = get_failnet_model()
    
    plt.figure(figsize=(18, 14))
    
    # Tier mapping based on Phase 2 in workflow doc
    observable = [
        "Trajectory_Length", "Required_Tool_Count", 
        "Execution_Error_Occurred", "Task_Failure", "Failure_Severity",
        "Environment_Volatility", "Human_Oversight_Present"
    ]
    heuristic = [
        "Tool_Argument_Validity", "Error_Recovery_Attempted", 
        "Multi_Agent_Coordination_Required", "Error_Recovery_Success"
    ]
    proxy_labeled = [
        "Instruction_Ambiguity", "Plan_Quality", 
        "Intermediate_Verification_Quality", "Hallucination_Risk",
        "Inter_Agent_Communication_Quality", "Role_Assignment_Correctness"
    ]
    
    colors = []
    for node in model.nodes():
        if node in observable:
            colors.append("lightblue") # Directly observable
        elif node in heuristic:
            colors.append("lightgreen") # Weakly derivable
        elif node in proxy_labeled:
            colors.append("lightcoral") # Proxy-labeled
        else:
            colors.append("lightgrey") # Base capabilities / Context
            
    # Try different layouts
    # kamada_kawai_layout usually works well for such networks
    pos = nx.kamada_kawai_layout(model)
    
    nx.draw(
        model, 
        pos,
        with_labels=True,
        node_color=colors,
        node_size=10000,
        font_size=9,
        font_weight="bold",
        arrows=True,
        arrowsize=20,
        edge_color="gray",
        linewidths=1.5,
        margins=0.1
    )
    
    # Create legend
    import matplotlib.lines as mlines
    obs_patch = mlines.Line2D([], [], color='lightblue', marker='o', linestyle='None', markersize=12, label='Observable')
    heur_patch = mlines.Line2D([], [], color='lightgreen', marker='o', linestyle='None', markersize=12, label='Heuristic (Weakly derivable)')
    proxy_patch = mlines.Line2D([], [], color='lightcoral', marker='o', linestyle='None', markersize=12, label='Proxy-labeled (LLM-assisted)')
    other_patch = mlines.Line2D([], [], color='lightgrey', marker='o', linestyle='None', markersize=12, label='Other / Context')
    plt.legend(handles=[obs_patch, heur_patch, proxy_patch, other_patch], loc='upper right', fontsize=12)
    
    plt.title("FailNet Bayesian Network Structure", fontsize=20, pad=20)
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bn_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    visualize()
