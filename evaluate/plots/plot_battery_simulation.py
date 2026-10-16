import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11

OUT_DIR = f"{ROOT_DIR}/zresults/early_exit_results"

JSON_PATH = os.path.join(OUT_DIR, "early_exit_ratio30_finetuned_comprehensive_metrics.json")

# Load evaluation metrics
with open(JSON_PATH, "r") as f:
    metrics = json.load(f)

os.makedirs(OUT_DIR, exist_ok=True)

# Plot bar chart

def main():
    policies = metrics["adaptive_policy_metrics"]
    
    sorted_keys = sorted(policies.keys(), key=lambda k: policies[k].get("battery_level", 0))
    
    policy_labels = []
    e1_pcts, e2_pcts, e3_pcts = [], [], []

    for key in sorted_keys:
        data = policies[key]
        name = key.replace("policy_", "").capitalize()
        policy_labels.append(f"{name} Battery\n({data['battery_level']}% Bat)")
        
        pcts = data.get("exit_percentages", {})
        e1_pcts.append(pcts.get("exit_1", 0.0))
        e2_pcts.append(pcts.get("exit_2", 0.0))
        e3_pcts.append(pcts.get("exit_3", 0.0))

    x = np.arange(len(policy_labels))
    width = 0.52

    fig, ax = plt.subplots(figsize=(8, 5.5))

    b1 = ax.bar(x, e1_pcts, width, label="Exit 1 (Fast / Early)", color="#2ca02c", edgecolor="white")
    b2 = ax.bar(x, e2_pcts, width, bottom=e1_pcts, label="Exit 2 (Medium)", color="#ff7f0e", edgecolor="white")
    
    bottom_e3 = np.array(e1_pcts) + np.array(e2_pcts)
    b3 = ax.bar(x, e3_pcts, width, bottom=bottom_e3, label="Exit 3 (Deep / Full)", color="#1f77b4", edgecolor="white")

    for idx in range(len(policy_labels)):
        if e1_pcts[idx] > 6.0:
            ax.text(x[idx], e1_pcts[idx] / 2, f"{e1_pcts[idx]:.1f}%", ha="center", va="center", color="white", fontweight="bold")
        if e2_pcts[idx] > 6.0:
            ax.text(x[idx], e1_pcts[idx] + e2_pcts[idx] / 2, f"{e2_pcts[idx]:.1f}%", ha="center", va="center", color="white", fontweight="bold")
        if e3_pcts[idx] > 6.0:
            ax.text(x[idx], bottom_e3[idx] + e3_pcts[idx] / 2, f"{e3_pcts[idx]:.1f}%", ha="center", va="center", color="white", fontweight="bold")

    ax.set_ylabel("Percentage of Total Inferences (%)", fontweight="bold")
    ax.set_title("Exit Head Utilization Breakdown per Battery Policy", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(policy_labels, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "exit_distribution_breakdown.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()
    print("Plots generated successfully!")