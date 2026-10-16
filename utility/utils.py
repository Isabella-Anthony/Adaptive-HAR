"""
Utility functions

Provides:
- device selection
- model parameter count
- model size
- evaluation
- latency benchmarking
"""

from __future__ import annotations
import io
import time
import torch
import torch.nn as nn
from pathlib import Path


# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

def save_checkpoint(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: str | Path, map_location=None) -> nn.Module:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    # Extract the state dict if the file is a metadata dictionary
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if saved with torch.nn.DataParallel
    clean_state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    model.load_state_dict(clean_state_dict)
    return model

def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


# --------------------------------------------------------------------------
# Model measurements
# --------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Return the total number of model parameters."""
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """Return serialized model size in MB."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)

    return len(buffer.getvalue()) / (1024 ** 2)





# --------------------------------------------------------------------------
# Train & evaluate
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
) -> tuple[float, float]:
    """
    Evaluate model.

    Returns:
        loss
        accuracy (%)
    """

    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(
            outputs,
            labels
        )

        total_loss += (
            loss.item()
            * inputs.size(0)
        )

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / total,
        100.0 * correct / total
    )

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)

    return total_loss / total

def fit(
    model, train_loader, val_loader, device,
    epochs: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    verbose: bool = True,
) -> nn.Module:

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            device
        )

        if verbose:
            print(
                f"epoch {epoch}/{epochs} - "
                f"train_loss: {train_loss:.4f} - "
                f"val_loss: {val_loss:.4f} - "
                f"val_acc: {val_acc:.2f}%"
            )

    return model

# --------------------------------------------------------------------------
# Latency
# --------------------------------------------------------------------------

@torch.no_grad()
def benchmark_latency(
    model,
    input_size=(1, 3, 1000),
    device=None,
    n_warmup=10,
    n_runs=50,
) -> float:
    """
    Measure average single-window inference latency in milliseconds.
    """

    device = device or get_device()

    model.eval()
    model.to(device)

    dummy = torch.randn(
        *input_size,
        device=device
    )

    # Warm-up
    for _ in range(n_warmup):
        model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(n_runs):
        model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    return (
        elapsed / n_runs
    ) * 1000

class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    Saves the best model state to disk and reloads it on early stop.
    """
    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "max", checkpoint_path: str = None):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_checkpoint = None

    def __call__(self, score: float, model: torch.nn.Module, metadata: dict = None) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, metadata)
            return True

        # Check for improvement based on mode ('max' for F1/Accuracy, 'min' for Loss)
        improved = (score > self.best_score + self.min_delta) if self.mode == "max" else (score < self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.save_checkpoint(score, model, metadata)
            self.counter = 0
            return True
        else:
            self.counter += 1
            print(f"  -> EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return False

    def save_checkpoint(self, score: float, model: torch.nn.Module, metadata: dict = None):
        payload = {
            "model_state_dict": model.state_dict(),
            "best_score": score,
        }
        if metadata:
            payload.update(metadata)

        if self.checkpoint_path:
            torch.save(payload, self.checkpoint_path)