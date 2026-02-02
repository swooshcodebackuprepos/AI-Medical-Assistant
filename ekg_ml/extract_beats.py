from pathlib import Path
import numpy as np
import wfdb

BASE = Path(__file__).parent
DATA_DIR = BASE / "data" / "mitdb"
RECORD = "100"

OUT_DIR = BASE / "data" / "derived"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_NPZ = OUT_DIR / "mitdb100_beats.npz"

FS = 360
PRE = int(0.25 * FS)   # 250 ms
POST = int(0.45 * FS)  # 450 ms

NORMAL = {"N", "L", "R", "e", "j"}
VENTRICULAR = {"V", "E"}

def beat_class(sym: str) -> int:
    if sym in NORMAL:
        return 0  # N
    if sym in VENTRICULAR:
        return 1  # V
    return 2      # O

def main():
    record_path = DATA_DIR / RECORD
    print("Loading record:", record_path)

    rec = wfdb.rdrecord(str(record_path))
    ann = wfdb.rdann(str(record_path), "atr")

    sig = rec.p_signal[:, 0].astype(np.float32)
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)

    X, y, syms = [], [], []

    for samp, sym in zip(ann.sample, ann.symbol):
        start = samp - PRE
        end = samp + POST
        if start < 0 or end > len(sig):
            continue
        X.append(sig[start:end])
        y.append(beat_class(sym))
        syms.append(sym)

    X = np.stack(X)              # (num_beats, window_len)
    y = np.array(y, dtype=np.int64)
    syms = np.array(syms)

    print("Windows:", X.shape)
    unique, counts = np.unique(y, return_counts=True)
    print("Class counts:", dict(zip(unique.tolist(), counts.tolist())))

    np.savez_compressed(OUT_NPZ, X=X, y=y, symbols=syms, fs=FS, pre=PRE, post=POST)
    print("Saved:", OUT_NPZ)

if __name__ == "__main__":
    main()
