from src.bn_learning import learn_parameters
from src.inference import FailNetInference
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

def run_experiment_5():
    print("=== Experiment 5: Sensitivity Analysis & Tornado Chart ===")
    
    model = learn_parameters()
    inference = FailNetInference(model)
    os.makedirs("report/figures", exist_ok=True)

    # ─── Part A: Streaming Belief Update ─────────────────────────────────
    print("\n[Part A] Streaming Evidence Update:")
    trace = [
        {"Task_Complexity": "High", "Base_Model_Capability": "High"},
        {"Plan_Quality": "Flawed"},
        {"Tool_Selection_Correctness": "Incorrect"},
        {"Execution_Error_Occurred": "Yes"},
        {"Error_Recovery_Success": "No"},
    ]
    beliefs = inference.streaming_belief_update(trace)
    prob_fail = []
    cumulative_evidence = {}
    for i, (step, belief) in enumerate(zip(trace, beliefs)):
        cumulative_evidence.update(step)
        p = belief.values[belief.state_names["Task_Failure"].index("Yes")]
        prob_fail.append(p)
        print(f"  Step {i+1}: {step} -> P(Task_Failure=Yes) = {p:.4f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    steps = list(range(1, len(trace) + 1))
    ax.plot(steps, prob_fail, marker="s", color="#e74c3c", linewidth=2.5, markersize=8)
    ax.fill_between(steps, prob_fail, alpha=0.12, color="#e74c3c")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Decision threshold (0.5)")
    ax.set_title("Streaming Belief Update — P(Task_Failure = Yes)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Execution Step")
    ax.set_ylabel("P(Task_Failure = Yes)")
    ax.set_xticks(steps)
    ax.set_xticklabels([list(s.keys())[0].replace("_", "\n") for s in trace], fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("report/figures/exp5_streaming_belief.png", dpi=150)
    print("  -> Saved: report/figures/exp5_streaming_belief.png")
    plt.close()

    # ─── Part B: Sensitivity Analysis / Tornado Chart ────────────────────
    print("\n[Part B] Computing Sensitivity Analysis (Tornado Chart)...")
    sa_df = inference.sensitivity_analysis(target="Task_Failure", target_state="Yes")
    print(sa_df.to_string(index=False))

    # Plot top-N variables as tornado chart
    top_n = 12
    df_plot = sa_df.head(top_n).iloc[::-1]  # reverse so largest swing is at top

    fig, ax = plt.subplots(figsize=(11, 7))
    base_prob = sa_df["BaseProb"].iloc[0] if len(sa_df) > 0 else 0.5

    colors_low  = "#2ecc71"   # green — better condition
    colors_high = "#e74c3c"   # red   — worse condition

    y_pos = np.arange(len(df_plot))
    bar_height = 0.55

    for i, (_, row) in enumerate(df_plot.iterrows()):
        # Left bar: best state → reduces failure prob
        ax.barh(
            y_pos[i], row["BaseProb"] - row["BestProb"],
            left=row["BestProb"], height=bar_height,
            color=colors_low, alpha=0.85,
        )
        # Right bar: worst state → increases failure prob
        ax.barh(
            y_pos[i], row["WorstProb"] - row["BaseProb"],
            left=base_prob, height=bar_height,
            color=colors_high, alpha=0.85,
        )
        # Labels
        ax.text(row["BestProb"]  - 0.01, y_pos[i], f"{row['BestState']}\n{row['BestProb']:.2f}",
                ha="right", va="center", fontsize=7.5, color="#1a5e35")
        ax.text(row["WorstProb"] + 0.01, y_pos[i], f"{row['WorstState']}\n{row['WorstProb']:.2f}",
                ha="left",  va="center", fontsize=7.5, color="#7b241c")

    ax.axvline(base_prob, color="black", linewidth=1.5, linestyle="--", label=f"Base P(Fail) = {base_prob:.2f}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["Variable"].str.replace("_", " "), fontsize=10)
    ax.set_xlabel("P(Task_Failure = Yes)", fontsize=11)
    ax.set_title(
        f"Sensitivity Analysis — Top {top_n} Influential Variables\n"
        "(Tornado Chart: wider bar = more influence on failure probability)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(0, 1)
    patch_low  = mpatches.Patch(color=colors_low,  alpha=0.85, label="Best-case state (↓ reduces failure)")
    patch_high = mpatches.Patch(color=colors_high, alpha=0.85, label="Worst-case state (↑ raises failure)")
    ax.legend(handles=[patch_low, patch_high, ax.lines[0]], loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig("report/figures/exp5_tornado_chart.png", dpi=150, bbox_inches="tight")
    print(f"  -> Saved: report/figures/exp5_tornado_chart.png")
    plt.close()

    print("\nExperiment 5 Completed.")


if __name__ == "__main__":
    run_experiment_5()
