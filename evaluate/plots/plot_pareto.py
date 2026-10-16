from pathlib import Path
import json
import os
import matplotlib.pyplot as plt
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

EARLY_EXIT_JSON = f"{ROOT_DIR}/zresults/early_exit_results/early_exit_ratio30_finetuned_comprehensive_metrics.json"
BASELINE_JSON = f"{ROOT_DIR}/zresults/baseline_results/baseline_metrics.json"
OUT_DIR = f"{ROOT_DIR}/zresults/early_exit_results"

# Load early exit metrics
with open(EARLY_EXIT_JSON, "r") as f:
    ee_data = json.load(f)

static = ee_data["static_exit_metrics"]
static_pts = [
    (static["exit_1"]["mflops"], static["exit_1"]["accuracy"] * 100, "Static Exit 1"),
    (static["exit_2"]["mflops"], static["exit_2"]["accuracy"] * 100, "Static Exit 2"),
    (static["exit_3"]["mflops"], static["exit_3"]["accuracy"] * 100, "Static Exit 3"),
]

adapt = ee_data["adaptive_policy_metrics"]
policy_pts = [
    (adapt["policy_low"]["average_mflops"], adapt["policy_low"]["accuracy"] * 100, "Low (15%)"),
    (adapt["policy_medium"]["average_mflops"], adapt["policy_medium"]["accuracy"] * 100, "Med (45%)"),
    (adapt["policy_high"]["average_mflops"], adapt["policy_high"]["accuracy"] * 100, "High (85%)"),
]

pareto_points = [
    static_pts[0],   # Static Exit 1
    policy_pts[0],   # Adaptive Low
    policy_pts[1],   # Adaptive Med
    policy_pts[2],   # Adaptive High
    static_pts[2],   # Static Exit 3
]

# Load Standard Baseline Metrics
with open(BASELINE_JSON, "r") as f:
    base_data = json.load(f)

baseline_mflops = base_data["flops"] / 1e6
baseline_acc = base_data["accuracy"]

# Plot
plt.figure(figsize=(10, 5.8))

# adaptive pareto frontier
front_x = [p[0] for p in pareto_points]
front_y = [p[1] for p in pareto_points]
plt.plot(front_x, front_y, "-", color="#2ca02c", linewidth=2.5, label="Empirical Pareto Frontier", zorder=3)

# adaptive operating points
p_x = [p[0] for p in policy_pts]
p_y = [p[1] for p in policy_pts]
plt.scatter(p_x, p_y, color="#d62728", marker=".", s=240, zorder=5, label="Adaptive Battery Policies (Low, Med, High)")

# Standard Baseline Model
plt.scatter(
    [baseline_mflops],
    [baseline_acc],
    color="#111111",
    marker="v",
    s=140,
    zorder=6,
    label=f"Standard ResNet Baseline ({baseline_acc:.1f}%, {baseline_mflops:.0f} MFLOPs)",
)

plt.title("Accuracy vs. FLOPs", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Computation per Window (MFLOPs)", fontsize=11)
plt.ylabel("Test Accuracy (%)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="lower right", frameon=True, fontsize=10)

plt.xlim(10, baseline_mflops * 1.08)
plt.ylim(64, 78)

plt.tight_layout()
save_path = os.path.join(OUT_DIR, "pareto_frontier_with_baseline.png")
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Plot saved successfully to: {save_path}")