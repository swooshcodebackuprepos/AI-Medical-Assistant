import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

NPZ_PATH = Path("data/derived/mitdb100_beats.npz")

def main():
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"Missing: {NPZ_PATH}. Run extract_beats.py first.")

    data = np.load(NPZ_PATH, allow_pickle=True)

    # These key names are the most common pattern we used: adjust if your keys differ
    # We'll print keys so you can confirm quickly.
    print("NPZ keys:", list(data.keys()))

    # Try common key names:
    beats = None
    labels = None

    for k in ["X", "beats", "windows"]:
        if k in data:
            beats = data[k]
            break

    for k in ["y", "labels", "ann_labels"]:
        if k in data:
            labels = data[k]
            break

    if beats is None:
        raise KeyError("Could not find beats array. Check NPZ keys printed above.")

    print("Beats shape:", beats.shape)
    if labels is not None:
        print("Labels shape:", labels.shape)
        # quick label distribution
        unique, counts = np.unique(labels, return_counts=True)
        print("Label counts:", dict(zip(unique.tolist(), counts.tolist())))

    # Plot a few random beats
    n = min(12, len(beats))
    idx = np.random.choice(len(beats), size=n, replace=False)

    plt.figure(figsize=(12, 6))
    for i, j in enumerate(idx, start=1):
        plt.subplot(3, 4, i)
        plt.plot(beats[j])
        title = f"#{j}"
        if labels is not None:
            title += f" | {labels[j]}"
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
