import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from utility.loaders import get_data_loaders
from utility.model import ResNet1D
from utility.utils import (
    benchmark_latency,
    count_parameters,
    evaluate,
    get_device,
    load_checkpoint,
    model_size_mb,
)

# configuration 
# -- DATABASE FILE PATH IF does not exist need to download and put in your path
USER = os.environ.get("USER", "")
DATA_DIR = f"/var/scratch/{USER}/capture24/processed"
# -- 

device = get_device()
device_cpu = torch.device("cpu")
print(f"Primary Compute Device : {device}")
BASE_DIR = f"{ROOT_DIR}/zresults/baseline_results"
CHECKPOINT_PATH = os.path.join(BASE_DIR, "resnet1d_baseline_best.pt")

# data loader

_, _, test_loader, classes, num_classes = get_data_loaders(
    data_dir=DATA_DIR,
    base_dir=BASE_DIR,
    batch_size=128,
    num_workers=1,
    device=device,
    include_test=True,
)

assert test_loader is not None, "Test split not found or include_test was False!"
num_test_samples = len(test_loader.dataset)
print(f"Loaded {num_test_samples:,} test samples across {num_classes} classes.")

# model setup
model = ResNet1D(in_channels=3, num_classes=num_classes)
loaded_model = load_checkpoint(model, CHECKPOINT_PATH)
loaded_model.to(device)
loaded_model.eval()

# evaluation and metrics
loss, accuracy = evaluate(loaded_model, test_loader, device)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

latency_ms = benchmark_latency(loaded_model, input_size=(1, 3, 1000), device=device_cpu)
parameters = count_parameters(loaded_model)
size_mb = model_size_mb(loaded_model)

print(f"Latency (CPU) : {latency_ms:.2f} ms")
print(f"Parameters    : {parameters:,}")
print(f"Model Size    : {size_mb:.2f} MB")

# FLOPS
model_cpu = ResNet1D(in_channels=3, num_classes=num_classes, width_mult=1.0).eval()
model_cpu.load_state_dict(loaded_model.state_dict())

dummy_input_cpu = (torch.randn(1, 3, 1000),)
flops = FlopCountAnalysis(model_cpu, dummy_input_cpu)
total_flops = flops.total()

print(f"Total FLOPs   : {total_flops:,}")
print(f"Total MFLOPs  : {total_flops / 1e6:.2f} MFLOPs")
print(flop_count_table(flops, max_depth=2))

loaded_model = loaded_model.to(device).eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        outputs = loaded_model(inputs)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())

macro_f1 = f1_score(all_labels, all_predictions, average="macro")
balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
cm = confusion_matrix(all_labels, all_predictions)

print(f"Macro F1          : {macro_f1:.4f}")
print(f"Balanced Accuracy : {balanced_acc:.4f}")
print("\nConfusion Matrix:")
print(cm)

# plot the confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks(range(num_classes), classes, rotation=45, ha="right")
plt.yticks(range(num_classes), classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Baseline Confusion Matrix")
plt.tight_layout()

cm_path = os.path.join(BASE_DIR, "baseline_confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"Saved confusion matrix plot to {cm_path}")

# save metrics to json

def to_builtin(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    raise TypeError(f"Not JSON serializable: {type(o)}")

results = {
    "model": "resnet1d_baseline",
    "test_samples": num_test_samples,
    "num_classes": num_classes,
    "classes": list(classes),
    "loss": loss,
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "balanced_accuracy": balanced_acc,
    "latency_ms_cpu": latency_ms,
    "parameters": parameters,
    "size_mb": size_mb,
    "flops": total_flops,
}

out_path = os.path.join(BASE_DIR, "baseline_metrics.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=to_builtin)

print(f"Saved metrics to {out_path}")