from __future__ import annotations

from pathlib import Path
from collections import Counter
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib


# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "derived" / "mitdb100_beats.npz"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Helpers
# -------------------------
def load_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    X = d["X"]
    y = d["y"]
    symbols = d.get("symbols", None)
    fs = int(d.get("fs", 0)) if "fs" in d else 0
    pre = int(d.get("pre", 0)) if "pre" in d else 0
    post = int(d.get("post", 0)) if "post" in d else 0
    return X, y, symbols, fs, pre, post


def to_str_list(symbols) -> list[str] | None:
    if symbols is None:
        return None
    try:
        out: list[str] = []
        for s in list(symbols):
            if isinstance(s, (bytes, bytearray)):
                out.append(s.decode("utf-8", errors="ignore"))
            else:
                out.append(str(s))
        return out
    except Exception:
        return None


def build_target_names(unique_classes: list[int], raw_symbols) -> list[str]:
    """
    IMPORTANT FIX:
    We only accept NPZ symbols if the labels for the *actual used class ids*
    are distinct. If class 0/1/2 all map to 'N', we reject and use fallback.
    """
    # Fallback mapping (works even if your NPZ labels are broken)
    fallback = {
        0: "Normal (N)",
        1: "Other/Rare",
        2: "Ventricular-ish (V)",
        3: "Supraventricular-ish (S)",
        4: "Fusion (F)",
        5: "Unknown/Noise (Q)",
    }

    sym_list = to_str_list(raw_symbols)

    if sym_list is not None and len(sym_list) > 0:
        max_id = max(unique_classes)

        # Only attempt if we can index by class id
        if len(sym_list) > max_id:
            candidate = [sym_list[c].strip() for c in unique_classes]

            # Reject if any are blank
            if all(name != "" for name in candidate):
                # Reject if duplicates exist (your current problem)
                if len(set(candidate)) == len(candidate):
                    return candidate

    # Use fallback if NPZ symbols are missing/broken/duplicated
    return [fallback.get(c, f"Class {c}") for c in unique_classes]


# -------------------------
# Main
# -------------------------
def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset: {DATA_PATH}\nRun: python extract_beats.py"
        )

    X, y, symbols, fs, pre, post = load_npz(DATA_PATH)

    print("NPZ loaded from:", DATA_PATH)
    print("Beats shape:", getattr(X, "shape", None))
    print("Labels shape:", getattr(y, "shape", None))

    # Ensure y is 1D ints
    y = np.asarray(y).astype(int).reshape(-1)
    X = np.asarray(X)

    if X.ndim != 2:
        X = np.squeeze(X)
    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D (n_beats, beat_len). Got: {X.shape}")

    counts = Counter(y.tolist())
    print("Label counts:", dict(sorted(counts.items(), key=lambda kv: kv[0])))

    unique_classes = sorted(np.unique(y).tolist())
    target_names = build_target_names(unique_classes, symbols)

    print("Resolved target_names:", target_names)

    # Stratify only if every class has >=2 samples
    min_count = min(counts.values()) if counts else 0
    stratify = y if min_count >= 2 else None
    if stratify is None:
        print(
            f"WARNING: At least one class has only {min_count} sample(s). "
            "Disabling stratify for train/test split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    print("Training baseline classifier...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=unique_classes,
            target_names=target_names,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=unique_classes))

    model_path = MODEL_DIR / "baseline_rf.joblib"
    bundle = {
        "model": clf,
        "classes": unique_classes,
        "target_names": target_names,
        "meta": {
            "fs": fs,
            "pre": pre,
            "post": post,
            "beat_len": int(X.shape[1]),
            "data_path": str(DATA_PATH),
        },
    }
    joblib.dump(bundle, model_path)

    print(f"\nSaved model: {model_path}")
    print("Saved target_names:", target_names)


if __name__ == "__main__":
    main()
