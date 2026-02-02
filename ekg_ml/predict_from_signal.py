# Proprietary Software
# Copyright (c) 2026 Nigel Phillips
# All rights reserved.
# Unauthorized copying, modification, distribution, or use is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import joblib
from scipy.signal import find_peaks


MODEL_PATH_DEFAULT = Path(__file__).parent / "models" / "baseline_rf.joblib"


@dataclass
class Prediction:
    label_id: int
    label_name: str
    confidence: float


def _to_str(x) -> str:
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def load_bundle(model_path: Path = MODEL_PATH_DEFAULT) -> dict:
    return joblib.load(model_path)


def load_model(model_path: Path = MODEL_PATH_DEFAULT):
    """
    Returns:
      model
      class_ids (list[int])  -> same ordering as target_names
      target_names (list[str])
      meta (dict)
    """
    bundle = load_bundle(model_path)
    model = bundle["model"]

    # Prefer the trained target_names/classes we now save
    class_ids = bundle.get("classes", None)
    target_names = bundle.get("target_names", None)

    if class_ids is None:
        # fallback to model.classes_ if present
        class_ids = getattr(model, "classes_", None)
        if class_ids is None:
            class_ids = list(range(getattr(model, "n_classes_", 0)))
    class_ids = [int(x) for x in list(class_ids)]

    if target_names is None:
        # fallback to old style symbols if present
        symbols = bundle.get("symbols", None)
        if symbols is None:
            target_names = [f"Class {c}" for c in class_ids]
        else:
            sym_list = [_to_str(s).strip() for s in list(symbols)]
            # map by class id if possible
            if len(sym_list) > max(class_ids):
                target_names = [sym_list[c] for c in class_ids]
            else:
                target_names = [f"Class {c}" for c in class_ids]
    else:
        target_names = [_to_str(s).strip() for s in list(target_names)]

    meta = bundle.get("meta", {})
    return model, class_ids, target_names, meta


def predict_beat_vector(
    beat_vec: np.ndarray,
    model_path: Path = MODEL_PATH_DEFAULT,
    top_k: int = 3
) -> List[Prediction]:
    """
    beat_vec: shape (beat_len,) e.g., (252,)
    """
    model, class_ids, target_names, _meta = load_model(model_path)

    x = np.asarray(beat_vec, dtype=np.float32).reshape(1, -1)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        # proba order corresponds to model.classes_
        model_class_ids = [int(v) for v in list(getattr(model, "classes_", class_ids))]
        id_to_name = {cid: target_names[i] if i < len(target_names) else f"Class {cid}"
                      for i, cid in enumerate(class_ids)}

        order = np.argsort(-proba)
        preds: List[Prediction] = []
        for idx in order[:top_k]:
            cid = int(model_class_ids[idx])
            name = id_to_name.get(cid, f"Class {cid}")
            preds.append(Prediction(label_id=cid, label_name=name, confidence=float(proba[idx])))
        return preds

    # fallback: no probabilities
    pred = int(model.predict(x)[0])
    name = target_names[class_ids.index(pred)] if pred in class_ids else f"Class {pred}"
    return [Prediction(label_id=pred, label_name=name, confidence=1.0)]


def extract_beats_from_1d_signal(
    signal_1d: np.ndarray,
    fs: int = 360,
    pre: int = 90,
    post: int = 162,
    peak_distance_ms: int = 250,
    prominence: float = 0.4,
) -> np.ndarray:
    """
    Starter R-peak based beat extraction from a 1D ECG-like waveform.
    Returns beats array of shape (n_beats, pre+post).
    """
    sig = np.asarray(signal_1d, dtype=np.float32)

    # normalize for stability
    s = sig - np.mean(sig)
    sd = np.std(s) + 1e-8
    s = s / sd

    distance = int((peak_distance_ms / 1000.0) * fs)
    peaks, _ = find_peaks(s, distance=distance, prominence=prominence)

    beats = []
    L = len(s)
    win = pre + post

    for p in peaks:
        a = p - pre
        b = p + post
        if a < 0 or b > L:
            continue
        beat = s[a:b]
        if beat.shape[0] == win:
            beats.append(beat)

    if not beats:
        return np.zeros((0, win), dtype=np.float32)

    return np.stack(beats).astype(np.float32)


def predict_from_1d_signal(
    signal_1d: np.ndarray,
    fs: int = 360,
    pre: int = 90,
    post: int = 162,
    model_path: Path = MODEL_PATH_DEFAULT,
    top_k: int = 3
) -> Tuple[List[Prediction], dict]:
    """
    Extract beats from a continuous 1D signal and aggregate predictions by mean probability.
    """
    model, class_ids, target_names, meta0 = load_model(model_path)

    # If meta has pre/post/fs, prefer them unless caller overrides
    if meta0:
        fs = int(meta0.get("fs", fs) or fs)
        pre = int(meta0.get("pre", pre) or pre)
        post = int(meta0.get("post", post) or post)

    beats = extract_beats_from_1d_signal(signal_1d, fs=fs, pre=pre, post=post)
    meta = {"beats_found": int(beats.shape[0]), "beat_len": int(pre + post), "fs": fs, "pre": pre, "post": post}

    if beats.shape[0] == 0:
        return [Prediction(label_id=-1, label_name="no_beats_found", confidence=0.0)], meta

    if not hasattr(model, "predict_proba"):
        preds = model.predict(beats)
        vals, counts = np.unique(preds, return_counts=True)
        best = int(vals[np.argmax(counts)])
        name = target_names[class_ids.index(best)] if best in class_ids else f"Class {best}"
        conf = float(np.max(counts) / len(preds))
        return [Prediction(label_id=best, label_name=name, confidence=conf)], meta

    proba = model.predict_proba(beats)  # (n_beats, n_classes)
    mean_proba = np.mean(proba, axis=0)

    model_class_ids = [int(v) for v in list(getattr(model, "classes_", class_ids))]
    id_to_name = {cid: target_names[i] if i < len(target_names) else f"Class {cid}"
                  for i, cid in enumerate(class_ids)}

    order = np.argsort(-mean_proba)
    out: List[Prediction] = []
    for idx in order[:top_k]:
        cid = int(model_class_ids[idx])
        name = id_to_name.get(cid, f"Class {cid}")
        out.append(Prediction(label_id=cid, label_name=name, confidence=float(mean_proba[idx])))

    return out, meta


if __name__ == "__main__":
    # Self-test: load one beat from the NPZ and predict
    npz = Path(__file__).parent / "data" / "derived" / "mitdb100_beats.npz"
    if npz.exists():
        d = np.load(npz, allow_pickle=True)
        X = d["X"]
        print("Loaded beats:", X.shape)
        preds = predict_beat_vector(X[0])
        print("Top predictions for beat[0]:")
        for p in preds:
            print(f"  {p.label_id} {p.label_name}  conf={p.confidence:.3f}")
    else:
        print("No NPZ found. Run extract_beats.py first.")
