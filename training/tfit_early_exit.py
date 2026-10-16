import json
import os
import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from utility.loaders import get_data_loaders
from utility.utils import get_device


# configuration 
# -- DATABASE FILE PATH IF does not exist need to download and put in your path
USER = os.environ.get("USER", "")
DATA_DIR = f"/var/scratch/{USER}/capture24/processed"
# -- 

BASELINE_DIR = f"{ROOT_DIR}/zresults/baseline_results"
EARLY_EXIT_PATH = f"{ROOT_DIR}/zresults/early_exit_results/early_exit_ratio30_best.pt"
OUTPUT_DIR = f"{ROOT_DIR}/zresults/early_exit_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, "early_exit_ratio30_finetuned_best.pt")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "early_exit_finetune_history.json")

DEVICE = get_device()
BATCH_SIZE = 128
NUM_WORKERS = 4
EPOCHS = 5
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42


torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# data loaders

train_loader, val_loader, _, classes, num_classes = get_data_loaders(
    data_dir=DATA_DIR,
    base_dir=BASELINE_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    device=DEVICE,
    include_test=False,
)

print(f"Classes ({num_classes}): {list(classes)}", flush=True)

# load model

print("Loading frozen-backbone early-exit model", flush=True)
model = torch.load(EARLY_EXIT_PATH, map_location=DEVICE, weights_only=False).to(DEVICE)
print(f"Loaded: {EARLY_EXIT_PATH}", flush=True)

# Unfreeze all parameters for end-to-end fine-tuning
for param in model.parameters():
    param.requires_grad = True

trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"Using device: {DEVICE}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}", flush=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

# training

best_score = 0.0
history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    train_total = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)

        out1, out2, out3 = model(x_batch)
        loss = criterion(out1, y_batch) + criterion(out2, y_batch) + criterion(out3, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = y_batch.size(0)
        train_loss += loss.item() * n
        train_total += n

    train_loss /= train_total

    # validation

    model.eval()
    val_correct = [0, 0, 0]
    val_total = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)

            outs = model(x_batch)
            for i, output in enumerate(outs):
                val_correct[i] += (output.argmax(dim=1) == y_batch).sum().item()

            val_total += y_batch.size(0)

    val_acc = [c / val_total for c in val_correct]
    mean_score = sum(val_acc) / 3

    # logging

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_acc_exit1": val_acc[0],
        "val_acc_exit2": val_acc[1],
        "val_acc_exit3": val_acc[2],
        "mean_val_accuracy": mean_score,
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | Loss: {train_loss:.4f} | "
        f"Val Acc exit1: {val_acc[0]:.4f}  exit2: {val_acc[1]:.4f}  exit3: {val_acc[2]:.4f} | "
        f"Mean: {mean_score:.4f}",
        flush=True,
    )
    # checkpoint
    if mean_score > best_score:
        best_score = mean_score
        torch.save(model, MODEL_PATH)
        print(f"  -> Saved best model (mean Val Acc: {best_score:.4f})", flush=True)

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=4)

print("\nFine-tuning complete.", flush=True)
print(f"Best mean validation accuracy: {best_score:.4f}", flush=True)
print(f"Model saved to: {MODEL_PATH}", flush=True)