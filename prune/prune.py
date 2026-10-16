import torch
import os
import torch_pruning as tp
from fvcore.nn import FlopCountAnalysis
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from utility.utils import (
    load_checkpoint,
    count_parameters,
    model_size_mb,
    benchmark_latency,
    save_checkpoint,
    evaluate,
    train_one_epoch,
    fit
)

from utility.loaders import get_data_loaders
from utility.utils import get_device
from utility.model import ResNet1D


# Configuration
device = get_device()
device_cpu = torch.device("cpu")
print(f"Primary Compute Device : {device}")

num_classes = 10
BASE_DIR = f"{ROOT_DIR}/zresults/baseline_results/resnet1d_baseline_best.pt"

PRUNING_RATIO = 0.30
FINE_TUNE_EPOCHS = 10  # Increased to 10 epochs for better recovery

PRUNED_DIR = f"{ROOT_DIR}/zresults/pruning_results"
PRUNED_PATH = os.path.join(PRUNED_DIR, f"resnet1d_pruned_ratio{int(PRUNING_RATIO * 100)}.pt")


# Load baseline model
model = ResNet1D(in_channels=3, num_classes=num_classes)

loaded_model = load_checkpoint(model,BASE_DIR,map_location=device)
loaded_model.to(device)
loaded_model.eval()
print("Baseline model loaded.")

example_inputs = torch.randn(1,3,1000,device=device)

# Statistics before pruning
params_before = count_parameters(loaded_model)
size_before = model_size_mb(loaded_model)

# Compute FLOPs before pruning
flops_analyzer_before = FlopCountAnalysis(loaded_model, example_inputs)
flops_before = flops_analyzer_before.total()

print("Before pruning")
print(f"Parameters : {params_before:,}")
print(f"Model size : {size_before:.2f} MB")
print(f"FLOPs      : {flops_before:,}")


# Ignore final classifier
ignored_layers = [loaded_model.classifier[-1]]

# Build DepGraph pruner
importance = tp.importance.MagnitudeImportance(p=2)

pruner = tp.pruner.MagnitudePruner(
    loaded_model,
    example_inputs=example_inputs,
    importance=importance,
    pruning_ratio=PRUNING_RATIO,
    ignored_layers=ignored_layers,
)

# Apply structured pruning
print(
f"Applying {PRUNING_RATIO * 100:.0f}% "
    "structured pruning..."
)

pruner.step()

print("Pruning complete.")

# Statistics after pruning
params_after = count_parameters(loaded_model)
size_after = model_size_mb(loaded_model)

#Compute FLOPs after pruning
flops_analyzer_after = FlopCountAnalysis(loaded_model, example_inputs)
flops_after = flops_analyzer_after.total()

print()
print("After pruning")
print(f"Parameters : {params_after:,}")
print(f"Model size : {size_after:.2f} MB")
print(f"FLOPs      : {flops_after:,}")

print(
    f"Parameter reduction : "
    f"{100 * (1 - params_after / params_before):.2f}%"  # calculate parameter reduction
)
print(
    f"Model size reduction : "
    f"{100 * (1 - size_after / size_before):.2f}%" # calculate model size reduction
)
print(
    f"FLOPs reduction     : "
    f"{100 * (1 - flops_after / flops_before):.2f}%"  # calculate FLOPs reduction 
)

# Fine-tune pruned model
train_loader, val_loader, test_loader, classes, num_classes = get_data_loaders(
    data_dir="/var/scratch/bde2601/capture24/processed", 
    base_dir="/var/scratch/bde2601/capture24/baseline_results", 
    batch_size=128, 
    num_workers=4, 
    device=device, 
    include_test=True
)

print(f"Fine-tuning pruned model " f"for up to {FINE_TUNE_EPOCHS} epochs...")

# fine tune and apply early stopping 
loaded_model = fit( loaded_model, train_loader, val_loader, device, epochs=FINE_TUNE_EPOCHS, verbose=True)

# Validation evaluation
val_loss, val_accuracy = evaluate(loaded_model, val_loader, device)

print()
print("Validation results")
print(f"Loss     : {val_loss:.4f}")
print(f"Accuracy : {val_accuracy:.2f}%")

# Latency Benchmark
latency = benchmark_latency(
    loaded_model,
    input_size=(1, 3, 1000),
    device=device
)

print(f"Latency  : {latency:.3f} ms")

# Save pruned model
os.makedirs(PRUNED_DIR, exist_ok=True)

torch.save(loaded_model, PRUNED_PATH)
print(f"Saved full pruned model to: {PRUNED_PATH}")

