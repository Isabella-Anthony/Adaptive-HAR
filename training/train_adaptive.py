import json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from utility.early_exit_model import EarlyExitResNet1D
from utility.loaders import get_data_loaders
from utility.utils import get_device


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# configuration 
# -- DATABASE FILE PATH IF does not exist need to download and put in your path
USER = os.environ.get("USER", "")
DATA_DIR = f"/var/scratch/{USER}/capture24/processed"
# -- 
BASELINE_DIR = f"{ROOT_DIR}/zresults/baseline_results"
PRUNING_DIR = f"{ROOT_DIR}/zresults/pruning_results"
OUTPUT_DIR = f"{ROOT_DIR}/zresults/early_exit_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRUNED_PATH = os.path.join(PRUNING_DIR, "resnet1d_pruned_ratio30.pt")
MODEL_PATH = os.path.join(OUTPUT_DIR, "early_exit_ratio30_best.pt")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "early_exit_history.json")

DEVICE = get_device()
BATCH_SIZE = 128
NUM_WORKERS = 4
EPOCHS = 10
LR = 1e-3
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

# class weights

train_labels = train_loader.dataset.y[train_loader.dataset.indices]
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=train_labels,
)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f"Class Weights:\n{class_weights_t.cpu().numpy()}", flush=True)

criterion = nn.CrossEntropyLoss(weight=class_weights_t)


# model setup

backbone = torch.load(PRUNED_PATH, map_location=DEVICE, weights_only=False)
model = EarlyExitResNet1D(backbone=backbone, num_classes=num_classes).to(DEVICE)

# Freeze backbone, unfreeze only new exit heads
for p in model.parameters():
    p.requires_grad = False
trainable = list(model.exit1.parameters()) + list(model.exit2.parameters())
for p in trainable:
    p.requires_grad = True

print(f"Using device: {DEVICE}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"Trainable params: {sum(p.numel() for p in trainable):,}", flush=True)

optimizer = torch.optim.Adam(trainable, lr=LR)

# training

def set_train_mode():
    # Frozen backbone stays in eval so BatchNorm running stats do not drift
    model.eval()
    model.exit1.train()
    model.exit2.train()

best_f1_score = 0.0
history = []

for epoch in range(1, EPOCHS + 1):
    # train
    set_train_mode()
    train_loss, train_total = 0.0, 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)

        out1, out2, _ = model(x_batch)
        loss = criterion(out1, y_batch) + criterion(out2, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = y_batch.size(0)
        train_loss += loss.item() * n
        train_total += n

    train_loss /= train_total

    # validation of all 3 exits
    model.eval()
    all_val_targets = []
    val_preds = [[], [], []]

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)

            outs = model(x_batch)
            all_val_targets.extend(y_batch.cpu().tolist())

            for i, out in enumerate(outs):
                val_preds[i].extend(out.argmax(dim=1).cpu().tolist())

    targets = np.array(all_val_targets)
    val_f1 = [float(f1_score(targets, np.array(val_preds[i]), average="macro")) for i in range(3)]
    val_acc = [float((np.array(val_preds[i]) == targets).mean()) for i in range(3)]

    # Checkpoint on the mean Macro F1 of the two active heads
    mean_exit_f1 = (val_f1[0] + val_f1[1]) / 2

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_f1_exit1": val_f1[0],
        "val_f1_exit2": val_f1[1],
        "val_f1_exit3": val_f1[2],
        "val_acc_exit1": val_acc[0],
        "val_acc_exit2": val_acc[1],
        "val_acc_exit3": val_acc[2],
        "mean_exit1_2_f1": mean_exit_f1,
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | Loss: {train_loss:.4f} | "
        f"Val F1 [E1: {val_f1[0]:.4f}, E2: {val_f1[1]:.4f}, E3: {val_f1[2]:.4f}] | "
        f"Mean E1/2 F1: {mean_exit_f1:.4f}",
        flush=True,
    )

    # Save best performing checkpoint
    if mean_exit_f1 > best_f1_score:
        best_f1_score = mean_exit_f1
        torch.save(model, MODEL_PATH)
        print(f"  -> Saved best model (Mean Exit 1/2 Macro F1: {mean_exit_f1:.4f})", flush=True)

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=4)

print("\nTraining complete.", flush=True)
print(f"Best Mean Exit 1/2 Macro F1: {best_f1_score:.4f}", flush=True)
print(f"Model saved to: {MODEL_PATH}", flush=True)