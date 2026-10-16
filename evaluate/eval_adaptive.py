import json
import os
import numpy as np
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from utility import early_exit_model, model as utility_model
sys.modules["early_exit_model"] = early_exit_model
sys.modules["model"] = utility_model
from adaptivity_mechanism.controller import get_battery_policy
from adaptivity_mechanism.inference import AdaptiveInference
from utility.loaders import get_data_loaders
from utility.utils import benchmark_latency, count_parameters, get_device, model_size_mb

# configuration

device = get_device()
device_cpu = torch.device("cpu")
print(f"Primary Compute Device : {device}")

# -- DATABASE FILE PATH IF does not exist need to download and put in your path
USER = os.environ.get("USER", "")
DATA_DIR = f"/var/scratch/{USER}/capture24/processed"
SPLIT_BASE_DIR = f"/var/scratch/{USER}/capture24/baseline_results"
# -- 

BASE_DIR = f"{ROOT_DIR}/zresults/early_exit_results"
MODEL_PATH = os.path.join(BASE_DIR, "early_exit_ratio30_finetuned_best.pt")
NAME = "early_exit_ratio30_finetuned"

# load model

model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False).eval()
dummy_x = torch.randn(1, 3, 1000)
pool = getattr(model, "pool", nn.Identity())

exit_subnets = {
    1: nn.Sequential(model.stem, model.blocks[0], model.blocks[1], model.exit1),
    2: nn.Sequential(model.stem, *model.blocks[:4], model.exit2),
    3: nn.Sequential(model.stem, *model.blocks, pool, nn.Flatten(1), model.exit3),
}

# measure FLOPS
flops_per_exit = {}
for k, subnet in exit_subnets.items():
    analyzer = FlopCountAnalysis(subnet, (dummy_x,))
    analyzer.unsupported_ops_warnings(False)
    analyzer.uncalled_modules_warnings(False)
    flops_per_exit[k] = int(analyzer.total())

# static inference

_, _, test_loader_batched, classes, num_classes = get_data_loaders(
    data_dir=DATA_DIR,
    base_dir=SPLIT_BASE_DIR,
    batch_size=128,
    num_workers=2,
    device=device,
    include_test=True,
)

assert test_loader_batched is not None, "Test split not found!"
total_test_samples = len(test_loader_batched.dataset)
print(f"Loaded {total_test_samples:,} test samples across {num_classes} classes.")

model = model.to(device)
all_labels = []
all_preds = {1: [], 2: [], 3: []}

print("\nRunning fast batched pass for standalone exit metrics", flush=True)
with torch.no_grad():
    for inputs, labels in test_loader_batched:
        inputs = inputs.to(device, non_blocking=True)
        out1, out2, out3 = model(inputs)

        all_labels.extend(labels.tolist())
        all_preds[1].extend(out1.argmax(dim=1).cpu().tolist())
        all_preds[2].extend(out2.argmax(dim=1).cpu().tolist())
        all_preds[3].extend(out3.argmax(dim=1).cpu().tolist())

all_labels = np.array(all_labels)

static_exit_metrics = {}
for k in (1, 2, 3):
    preds = np.array(all_preds[k])
    subnet_cpu = exit_subnets[k].to(device_cpu)

    acc = float((preds == all_labels).mean())
    f1 = float(f1_score(all_labels, preds, average="macro"))
    bal_acc = float(balanced_accuracy_score(all_labels, preds))
    lat_ms = benchmark_latency(subnet_cpu, input_size=(1, 3, 1000), device=device_cpu)
    params = count_parameters(subnet_cpu)

    static_exit_metrics[f"exit_{k}"] = {
        "accuracy": acc,
        "macro_f1": f1,
        "balanced_accuracy": bal_acc,
        "latency_ms_cpu": lat_ms,
        "parameters": params,
        "flops": flops_per_exit[k],
        "mflops": flops_per_exit[k] / 1e6,
    }

# dynamic adaptive inference

_, _, test_loader_single, _, _ = get_data_loaders(
    data_dir=DATA_DIR,
    base_dir=SPLIT_BASE_DIR,
    batch_size=1,
    num_workers=1,
    device=device,
    include_test=True,
)

engine = AdaptiveInference(model=model, device=device)
battery_scenarios = [15, 45, 85]  
adaptive_policy_metrics = {}

print("\nRunning dynamic adaptive evaluation across battery policies", flush=True)
for bat in battery_scenarios:
    policy = get_battery_policy(bat)
    preds, targets = [], []
    exit_counts = {1: 0, 2: 0, 3: 0}

    for x, y in tqdm(test_loader_single, desc=f"Policy: {policy.name:<6} ({bat}% battery)"):
        res = engine.predict(x, policy=policy) 
        preds.append(res["prediction"]) 
        targets.append(y.item())
        exit_counts[res["exit"]] += 1  

    targets = np.array(targets)
    preds = np.array(preds)

    acc = float((preds == targets).mean())
    f1 = float(f1_score(targets, preds, average="macro"))
    bal_acc = float(balanced_accuracy_score(targets, preds))

    # calculate average FLOPS
    avg_flops = sum(exit_counts[k] * flops_per_exit[k] for k in (1, 2, 3)) / total_test_samples

    adaptive_policy_metrics[f"policy_{policy.name}"] = {
        "battery_level": bat,
        "accuracy": acc,
        "macro_f1": f1,
        "balanced_accuracy": bal_acc,
        "average_mflops": avg_flops / 1e6,
        "exit_counts": {
            "exit_1": exit_counts[1],
            "exit_2": exit_counts[2],
            "exit_3": exit_counts[3],
        },
        "exit_percentages": {
            "exit_1": round(100.0 * exit_counts[1] / total_test_samples, 2),
            "exit_2": round(100.0 * exit_counts[2] / total_test_samples, 2),
            "exit_3": round(100.0 * exit_counts[3] / total_test_samples, 2),
        },
    }

# export to json

full_results = {
    "model_name": NAME,
    "test_samples": total_test_samples,
    "num_classes": num_classes,
    "classes": list(classes),
    "model_total_parameters": count_parameters(model),
    "model_total_size_mb": model_size_mb(model),
    "static_exit_metrics": static_exit_metrics,
    "adaptive_policy_metrics": adaptive_policy_metrics,
}

out_path = os.path.join(BASE_DIR, f"{NAME}_comprehensive_metrics.json")
with open(out_path, "w") as f:
    json.dump(full_results, f, indent=2)

print(f"\nAll metrics successfully saved to: {out_path}\n")

# print summary of results
print(f"{'Exit':<8}{'Accuracy':>10}{'Macro F1':>10}{'Latency (ms)':>14}{'MFLOPs':>10}{'Params':>12}")
for k, data in static_exit_metrics.items():
    print(
        f"{k:<8}{data['accuracy']:>10.4f}{data['macro_f1']:>10.4f}"
        f"{data['latency_ms_cpu']:>14.2f}{data['mflops']:>10.2f}{data['parameters']:>12,}"
    )

print("\n--- Adaptive Policy Performance ---")
for p_name, data in adaptive_policy_metrics.items():
    dist = data["exit_percentages"]
    print(
        f"{p_name:<16} | Acc: {data['accuracy']:.4f} | F1: {data['macro_f1']:.4f} | "
        f"Avg MFLOPs: {data['average_mflops']:.2f} | "
        f"Exits: E1={dist['exit_1']}% E2={dist['exit_2']}% E3={dist['exit_3']}%"
    )