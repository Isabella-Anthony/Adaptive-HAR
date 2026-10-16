import json
import os
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from utility.loaders import get_data_loaders
from utility.model import ResNet1D
from utility.utils import get_device, EarlyStopping

# configuration 
# -- DATABASE FILE PATH IF does not exist need to download and put in your path
USER = os.environ.get("USER", "")
DATA_DIR = f"/var/scratch/{USER}/capture24/processed"
# -- 

OUTPUT_DIR = f"{ROOT_DIR}/zresults/baseline_results"
SPLIT_DIR = os.path.join(OUTPUT_DIR, "splits")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, "resnet1d_baseline_best.pt")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history.json")

DEVICE = get_device()
BATCH_SIZE = 128
NUM_WORKERS = 4
EPOCHS = 30          
PATIENCE = 3          
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# data loaders
train_loader, val_loader, test_loader, classes, num_classes = get_data_loaders(
    data_dir=DATA_DIR,
    base_dir=OUTPUT_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    device=DEVICE,
    include_test=True,
)

print(f"Classes ({num_classes}): {list(classes)}", flush=True)

# class weights
train_labels = train_loader.dataset.y[train_loader.dataset.indices]
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=train_labels,
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f"Computed Balanced Class Weights:\n{class_weights.cpu().numpy()}", flush=True)

# model
model = ResNet1D(
    in_channels=3,
    num_classes=num_classes,
    width_mult=1.0,
).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# early stopping
early_stopping = EarlyStopping(
    patience=PATIENCE,
    min_delta=1e-3,
    mode="max",           # Maximize Macro F1
    checkpoint_path=MODEL_PATH,
)

# training and validation loop
history = []

for epoch in range(1, EPOCHS + 1):
    # train
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * y_batch.size(0)
        train_correct += (outputs.argmax(dim=1) == y_batch).sum().item()
        train_total += y_batch.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total

    # validation
    model.eval()
    val_loss = 0.0
    val_total = 0
    all_val_preds = []
    all_val_targets = []

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            val_loss += loss.item() * y_batch.size(0)
            val_total += y_batch.size(0)

            all_val_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_val_targets.extend(y_batch.cpu().tolist())

    val_loss /= val_total
    val_acc = (np.array(all_val_preds) == np.array(all_val_targets)).mean()
    val_f1 = f1_score(all_val_targets, all_val_preds, average="macro")

    # logging
    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
        f"Val Macro F1: {val_f1:.4f}",
        flush=True,
    )

    # check earlystopping and best fit
    is_best = early_stopping(
        score=val_f1,
        model=model,
        metadata={
            "epoch": epoch,
            "val_macro_f1": val_f1,
            "val_accuracy": val_acc,
            "class_names": list(classes),
        },
    )

    if is_best:
        print(f"  -> Saved new best model (Val Macro F1: {val_f1:.4f})", flush=True)

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=4)

    # if patience reached break loop
    if early_stopping.early_stop:
        print(f"\n[!] Early stopping triggered at epoch {epoch}. No improvement for {PATIENCE} epochs.", flush=True)
        break

print("\nTraining complete.")
print(f"Best Validation Macro F1 achieved: {early_stopping.best_score:.4f}", flush=True)
print(f"Best checkpoint saved to: {MODEL_PATH}", flush=True)