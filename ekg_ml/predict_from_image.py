# Proprietary Software
# Copyright (c) 2026 Nigel Phillips
# All rights reserved.
# Unauthorized copying, modification, distribution, or use is prohibited.

import argparse
from pathlib import Path

import cv2
import numpy as np
import joblib


DEFAULT_MODEL = Path(__file__).parent / "models" / "baseline_rf.joblib"


def load_bundle(model_path: Path):
    bundle = joblib.load(model_path)
    # bundle expected keys: model, classes, target_names (or symbols/whatever)
    model = bundle["model"]
    classes = bundle.get("classes", None)
    target_names = bundle.get("target_names", None)

    if classes is None:
        # fallback: use model.classes_ if available
        classes = getattr(model, "classes_", None)

    if target_names is None:
        # fallback: create generic names
        if classes is None:
            target_names = None
        else:
            target_names = [str(c) for c in classes]

    return model, classes, target_names


def image_to_waveform(image_bgr: np.ndarray) -> np.ndarray:
    """
    Very simple screenshot->waveform extraction:
    - convert to grayscale
    - threshold to isolate dark ink
    - for each x column, estimate trace y using median of ink pixels
    - return waveform as y-values (inverted so "up" is positive)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Invert: ECG trace usually darker than background
    # Threshold to keep dark pixels
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    h, w = mask.shape
    ys = np.full(w, np.nan, dtype=np.float32)

    for x in range(w):
        col = mask[:, x]
        ink_y = np.where(col > 0)[0]
        if ink_y.size > 0:
            ys[x] = np.median(ink_y)

    # fill gaps
    valid = np.isfinite(ys)
    if valid.sum() < max(20, w * 0.02):
        raise RuntimeError(
            "Could not extract ECG trace from image. "
            "Try a tighter crop (just one strip), higher resolution, or higher contrast."
        )

    xs = np.arange(w)
    ys_interp = np.interp(xs, xs[valid], ys[valid]).astype(np.float32)

    # Convert y position to waveform amplitude:
    # image origin y=0 at top, so invert
    wave = (h - ys_interp)

    # Remove slow baseline drift (simple detrend)
    wave = wave - np.mean(wave)

    # Normalize
    std = np.std(wave) + 1e-8
    wave = wave / std

    return wave.astype(np.float32)


def window_waveform(wave: np.ndarray, window_size: int = 252, stride: int = 84) -> np.ndarray:
    """
    Produce overlapping windows of length 252.
    Stride 84 means ~3 windows per 252 block.
    """
    n = len(wave)
    if n < window_size:
        # pad
        pad = window_size - n
        wave = np.pad(wave, (0, pad), mode="edge")
        n = len(wave)

    windows = []
    for start in range(0, n - window_size + 1, stride):
        windows.append(wave[start:start + window_size])

    X = np.stack(windows, axis=0).astype(np.float32)
    return X


def predict_from_image(image_path: Path, model_path: Path, topk: int = 3):
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    model, classes, target_names = load_bundle(model_path)

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    wave = image_to_waveform(img)

    # Turn into 252-length windows for the beat model
    X = window_waveform(wave, window_size=252, stride=84)

    # Model expects shape (n_samples, 252)
    if X.shape[1] != 252:
        raise RuntimeError(f"Internal error: expected 252 features, got {X.shape[1]}")

    probs_all = model.predict_proba(X)  # shape: (n_windows, n_classes)
    probs = probs_all.mean(axis=0)      # aggregate across windows

    # Determine labels/classes
    if classes is None:
        classes = list(range(len(probs)))

    # Sort top-k
    idx = np.argsort(probs)[::-1][:topk]
    results = []
    for i in idx:
        cls = int(classes[i]) if i < len(classes) else int(i)
        name = target_names[cls] if (target_names is not None and cls < len(target_names)) else str(cls)
        results.append({
            "class_id": cls,
            "label": name,
            "confidence": float(probs[i]),
        })

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to screenshot image (png/jpg)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to trained joblib model bundle")
    ap.add_argument("--topk", type=int, default=3, help="Number of top predictions to show")
    args = ap.parse_args()

    image_path = Path(args.image).resolve()
    model_path = Path(args.model).resolve()

    preds = predict_from_image(image_path, model_path, topk=args.topk)

    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    print("Top predictions:")
    for p in preds:
        print(f"  {p['class_id']} {p['label']}  conf={p['confidence']:.3f}")


if __name__ == "__main__":
    main()
