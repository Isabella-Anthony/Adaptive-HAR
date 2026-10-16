import os
import sys
import glob
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

# Configuration
USER = os.environ.get("USER", "")
DEFAULT_SCRATCH = f"var/scratch/{USER}/capture24"

DATA_DIR = os.environ.get("DATA_DIR", DEFAULT_SCRATCH)
SAVE_DIR = os.environ.get("SAVE_DIR", os.path.join(DEFAULT_SCRATCH, "processed"))
LABEL_DICT_FILE = "annotation-label-dictionary.csv"

WINDOW_SIZE = 1000       
MIN_LABEL_PURITY = 0.80  

NUM_WORKERS = 4


def clean_participant_id(filepath: str) -> str:
    """Extract and sanitize participant ID (e.g. 'P075.csv.gz' -> 'P075')."""
    base = os.path.basename(filepath)
    pid = base.replace(".csv.gz", "").replace(".csv", "")
    return pid.split(" ")[0].strip()


def process_participant(filepath: str, label_df: pd.DataFrame):
    participant_id = clean_participant_id(filepath)

    try:
        # Load only necessary columns
        df = pd.read_csv(
            filepath,
            usecols=["time", "x", "y", "z", "annotation"],
            engine="c",
            low_memory=False,
        )
    except Exception as e:
        sys.stderr.write(f"[Warning] Failed to read {filepath}: {e}\n")
        return None, None, None

    # Merge labels on annotation string
    df = df.merge(
        label_df[["annotation", "label:WillettsSpecific2018"]],
        on="annotation",
        how="left",
    )
    df.rename(columns={"label:WillettsSpecific2018": "activity"}, inplace=True)

    windows = []
    labels = []
    participants = []

    # Non-overlapping sliding window
    for start in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = df.iloc[start : start + WINDOW_SIZE]

        # Ignore windows containing missing activity labels
        labeled = window["activity"].dropna()
        if len(labeled) == 0:
            continue

        # Find the most common activity 
        label_counts = labeled.value_counts()
        majority_label = label_counts.index[0]

        # Check label purity
        purity = label_counts.iloc[0] / WINDOW_SIZE

        if purity >= MIN_LABEL_PURITY:
            signal = window[["x", "y", "z"]].to_numpy(dtype=np.float32).T
            windows.append(signal)
            labels.append(majority_label)
            participants.append(participant_id)

    if not windows:
        return None, None, None

    return (
        np.array(windows, dtype=np.float32),
        np.array(labels),
        np.array(participants),
    )


def worker_task(filepath: str, label_df: pd.DataFrame):
    return process_participant(filepath, label_df)


def main():
    label_path = os.path.join(DATA_DIR, LABEL_DICT_FILE)
    if not os.path.exists(label_path):
        label_path = os.path.join(os.path.dirname(DATA_DIR), LABEL_DICT_FILE)
        if not os.path.exists(label_path):
            sys.exit(f"Error: Label dictionary not found at {label_path}")

    print(f"Loading label dictionary from {label_path}...")
    label_df = pd.read_csv(label_path)

    files = sorted(glob.glob(os.path.join(DATA_DIR, "P*.csv.gz")))
    if not files:
        files = sorted(glob.glob(os.path.join(DATA_DIR, "P*.csv")))

    if not files:
        sys.exit(f"Error: No participant files found in {DATA_DIR}")

    print(f"Found {len(files)} participant files.")
    print(f"Allocated Workers (CPUs): {NUM_WORKERS}")
    print(f"Reading from: {DATA_DIR}")
    print(f"Saving to   : {SAVE_DIR}")

    all_X = []
    all_y = []
    all_participants = []

    # Parallelize across files
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(worker_task, f, label_df) for f in files]
        for idx, f in enumerate(futures):
            X_p, y_p, p_ids = f.result()
            if X_p is not None and len(X_p) > 0:
                all_X.append(X_p)
                all_y.append(y_p)
                all_participants.append(p_ids)
            print(f"Completed [{idx + 1}/{len(files)}]", flush=True)

    if not all_X:
        sys.exit("Error: No valid windows produced. Check purity threshold and input data.")

    print("\nConcatenating processed arrays...")
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    participant_ids = np.concatenate(all_participants, axis=0)

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_x = os.path.join(SAVE_DIR, "X.npy")
    out_y = os.path.join(SAVE_DIR, "y.npy")
    out_pid = os.path.join(SAVE_DIR, "participant_ids.npy")

    np.save(out_x, X)
    np.save(out_y, y)
    np.save(out_pid, participant_ids)

    print(f"\nSaved successfully to {SAVE_DIR}:")
    print(f"  X shape: {X.shape} (Format: [Batch, Channels=3, Length={WINDOW_SIZE}])")
    print(f"  y shape: {y.shape}")
    print(f"  Total windows: {len(y):,}")
    print(f"  Unique participants: {len(np.unique(participant_ids))}")
    print("\nClass Distribution:")
    print(pd.Series(y).value_counts())


if __name__ == "__main__":
    main()