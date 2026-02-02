# Proprietary Software
# Copyright (c) 2026 Nigel Phillips
# All rights reserved.
# Unauthorized copying, modification, distribution, or use is prohibited.

from __future__ import annotations

from pathlib import Path
from collections import Counter
import numpy as np
import wfdb


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "mitdb"
DERIVED_DIR = BASE_DIR / "data" / "derived"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DERIVED_DIR.mkdir(parents=True, exist_ok=True)


# MIT-BIH Arrhythmia Database common record IDs (you can extend this list)
RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107",
    "108", "109", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "121", "122", "123", "124",
]


# Simple mapping from annotation symbol -> class id
# This is NOT full AAMI yet, but it's a better start than single-record.
# 0 = Normal (N-like)
# 1 = Other/Rare (everything else for now)
# 2 = Ventricular-ish (V-like)
NORMAL_SET = {"N", "L", "R", "e", "j"}
VENT_SET = {"V", "E"}  # ventricular ectopic beats / ventricular escape-ish
# everything else -> Other/Rare


def sym_to_class(sym: str) -> int:
    s = sym.strip()
    if s in NORMAL_SET:
        return 0
    if s in VENT_SET:
        return 2
    return 1


def decode_sym(x) -> str:
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def main():
    fs = 360
    pre = 90
    post = 162
    win = pre + post

    X_all = []
    y_all = []

    print("Downloading records (if needed) and extracting beats...")
    for rec in RECORDS:
        print("Record:", rec)

        # Download record + annotations into DATA_DIR
        wfdb.dl_database("mitdb", dl_dir=str(DATA_DIR), records=[rec])

        # Read record and annotation (atr)
        rec_path = DATA_DIR / rec
        record = wfdb.rdrecord(str(rec_path))
        ann = wfdb.rdann(str(rec_path), "atr")

        # Use channel 0 (often MLII)
        sig = record.p_signal[:, 0].astype(np.float32)

        # Annotation sample indices + symbols
        samp = ann.sample
        syms = [decode_sym(s) for s in ann.symbol]

        # Extract beats at annotated positions
        for idx, sym in zip(samp, syms):
            a = idx - pre
            b = idx + post
            if a < 0 or b > len(sig):
                continue
            beat = sig[a:b]
            if beat.shape[0] != win:
                continue

            # Normalize each beat (helps RF baseline)
            beat = beat - np.mean(beat)
            sd = np.std(beat) + 1e-8
            beat = beat / sd

            cls = sym_to_class(sym)

            X_all.append(beat)
            y_all.append(cls)

    X = np.stack(X_all).astype(np.float32)
    y = np.asarray(y_all, dtype=int)

    counts = Counter(y.tolist())
    print("Label counts:", dict(sorted(counts.items(), key=lambda kv: kv[0])))

    # Human labels for the class ids (these will be correct now)
    symbols = np.array(["Normal (N)", "Other/Rare", "Ventricular-ish (V)"], dtype=object)

    out = DERIVED_DIR / "mitdb_multi_beats.npz"
    np.savez_compressed(out, X=X, y=y, symbols=symbols, fs=fs, pre=pre, post=post)
    print("Saved:", out)
    print("X shape:", X.shape, "y shape:", y.shape)


if __name__ == "__main__":
    main()
