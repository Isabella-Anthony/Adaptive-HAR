import os
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class CaptureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray):
        self.X = X
        self.y = y
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        i = self.indices[idx]
        x = np.array(self.X[i], dtype=np.float32, copy=True)
        y = self.y[i]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def get_data_loaders(
    data_dir: str = "/var/scratch/bde2601/capture24/processed",
    base_dir: str = "/var/scratch/bde2601/capture24/baseline_results",
    batch_size: int = 128,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    include_test: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], np.ndarray, int]:
    """
    Loads memory-mapped data and split indices, creating and saving splits 
    automatically if they do not already exist on disk.
    """
    split_dir = os.path.join(base_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)

    train_idx_path = os.path.join(split_dir, "train_indices.npy")
    val_idx_path = os.path.join(split_dir, "val_indices.npy")
    test_idx_path = os.path.join(split_dir, "test_indices.npy")
    classes_path = os.path.join(split_dir, "classes.npy")


    # Generate & Persist Splits if needed
    if not (os.path.exists(train_idx_path) and os.path.exists(val_idx_path) and os.path.exists(classes_path)):
        print("Splits not found. Generating new participant splits...", flush=True)
        y_raw_init = np.load(os.path.join(data_dir, "y.npy"), allow_pickle=True)
        pids = np.load(os.path.join(data_dir, "participant_ids.npy"), allow_pickle=True)

        encoder = LabelEncoder()
        encoder.fit(y_raw_init)
        np.save(classes_path, encoder.classes_)

        participants = np.unique(pids)
        train_val_pids, test_pids = train_test_split(participants, test_size=0.15, random_state=seed)
        train_pids, val_pids = train_test_split(train_val_pids, test_size=(0.15 / 0.85), random_state=seed)

        train_indices = np.flatnonzero(np.isin(pids, train_pids))
        val_indices = np.flatnonzero(np.isin(pids, val_pids))
        test_indices = np.flatnonzero(np.isin(pids, test_pids))

        np.save(os.path.join(split_dir, "train_pids.npy"), train_pids)
        np.save(os.path.join(split_dir, "val_pids.npy"), val_pids)
        np.save(os.path.join(split_dir, "test_pids.npy"), test_pids)
        np.save(train_idx_path, train_indices)
        np.save(val_idx_path, val_indices)
        np.save(test_idx_path, test_indices)
        print(f"splits generated and saved to {split_dir}", flush=True)


    # Load data & split yndices
    print("Loading data and split indices", flush=True)
    X = np.load(os.path.join(data_dir, "X.npy"), mmap_mode="r")
    y_raw = np.load(os.path.join(data_dir, "y.npy"), allow_pickle=True)

    classes = np.load(classes_path)
    train_indices = np.load(train_idx_path)
    val_indices = np.load(val_idx_path)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[item] for item in y_raw], dtype=np.int64)

    is_cuda = (device.type == "cuda") if device is not None else torch.cuda.is_available()

    # build dataLoaders
    # --------------------------------------------------------
    train_dataset = CaptureDataset(X, y, train_indices)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=is_cuda
    )

    val_dataset = CaptureDataset(X, y, val_indices)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=is_cuda
    )

    test_loader = None
    if include_test and os.path.exists(test_idx_path):
        test_indices = np.load(test_idx_path)
        test_dataset = CaptureDataset(X, y, test_indices)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=is_cuda
        )
        print(f"Loaded splits: {len(train_indices):,} train, {len(val_indices):,} val, {len(test_indices):,} test.")
    else:
        print(f"Loaded splits: {len(train_indices):,} train, {len(val_indices):,} val.")

    return train_loader, val_loader, test_loader, classes, len(classes)