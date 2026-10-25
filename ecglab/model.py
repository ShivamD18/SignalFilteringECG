"""Train and evaluate heartbeat classifiers on MIT-BIH with an inter-patient protocol.

Why inter-patient?  If beats from the *same person* appear in both the
training and test sets (a random beat-level split), a classifier can simply
memorise each patient's beat shapes and report ~99 % accuracy. That number
collapses on new patients. Following de Chazal et al. (2004) we train on the
22 DS1 records and test on 22 *different* people in DS2, and pick models with
cross-validation that is grouped by patient. ``intra_patient_demo`` reproduces
the inflated figure so the difference can be seen.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data
from .detect import refine_peaks
from .features import FEATURE_NAMES, extract_features
from .filters import TRAINING_FILTER, preprocess

CACHE_DIR = data.DATA_DIR / "cache"
MODELS_DIR = data.ROOT / "models"
CLASSES = data.CLASSES


@dataclass
class BeatDataset:
    X: np.ndarray
    y: np.ndarray          # index into CLASSES
    groups: np.ndarray     # record name of each beat (one record = one patient)
    positions: np.ndarray  # R-peak sample index


def record_features(name: str):
    """Features for every labelled N/S/V/F beat of one record, aligned on the annotations."""
    rec = data.load_record(name)
    ecg = preprocess(rec.signal, rec.fs, TRAINING_FILTER)
    r = refine_peaks(ecg, rec.beat_samples, rec.fs)
    X, idx = extract_features(ecg, r, rec.fs)
    cls = rec.beat_classes[idx]
    keep = np.isin(cls, CLASSES)
    y = np.array([CLASSES.index(c) for c in cls[keep]], dtype=np.int64)
    return X[keep], y, r[idx][keep]


def load_dataset(records, use_cache: bool = True) -> BeatDataset:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for name in records:
        path = CACHE_DIR / f"{name}.npz"
        if use_cache and path.exists():
            z = np.load(path)
            X, y, pos = z["X"], z["y"], z["pos"]
            if X.shape[1] != len(FEATURE_NAMES):
                use_cache = False
        if not (use_cache and path.exists()):
            X, y, pos = record_features(name)
            np.savez_compressed(path, X=X, y=y, pos=pos)
        parts.append((X, y, np.full(len(y), name), pos))
    return BeatDataset(*(np.concatenate([p[i] for p in parts]) for i in range(4)))


def class_weights(y: np.ndarray, power: float = 0.5) -> np.ndarray:
    """Per-sample weights ~ (1 / class frequency) ** power.

    power=1 is sklearn's "balanced"; 0.5 softens it so the very rare fusion
    class does not dominate training.
    """
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    w = (counts.sum() / np.maximum(counts, 1)) ** power
    w = w / np.sum(w * counts) * counts.sum()
    return w[y]


def candidate_models() -> dict:
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, max_features="sqrt", n_jobs=-1, random_state=0),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
            l2_regularization=1.0, random_state=0),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(32, 16), alpha=1e-2, learning_rate_init=1e-3, max_iter=200,
            early_stopping=False, random_state=0)),
    }


# Class-weight strength per model, chosen by DS1 grouped cross-validation (macro-F1).
WEIGHT_POWER = {"logistic_regression": 0.5, "random_forest": 1.0, "gradient_boosting": 1.0, "mlp": 0.5}


def fit(name: str, X, y):
    model = candidate_models()[name]
    w = class_weights(y, WEIGHT_POWER[name])
    if hasattr(model, "steps"):
        model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    else:
        model.fit(X, y, sample_weight=w)
    return model


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """AAMI EC57-style metrics plus the binary normal-vs-abnormal view."""
    k = len(CLASSES)
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    per_class = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        se = tp / max(cm[i].sum(), 1)
        ppv = tp / max(cm[:, i].sum(), 1)
        per_class[c] = {"support": int(cm[i].sum()), "sensitivity": float(se), "ppv": float(ppv),
                        "f1": float(2 * se * ppv / (se + ppv)) if se + ppv else 0.0}
    abn_t, abn_p = y_true != 0, y_pred != 0
    tp = int(np.sum(abn_t & abn_p)); tn = int(np.sum(~abn_t & ~abn_p))
    fp = int(np.sum(~abn_t & abn_p)); fn = int(np.sum(abn_t & ~abn_p))
    se, sp = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    return {
        "confusion_matrix": cm.tolist(),
        "accuracy": float(np.trace(cm) / cm.sum()),
        "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
        "per_class": per_class,
        "binary": {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "sensitivity": se, "specificity": sp,
                   "ppv": ppv, "f1": 2 * se * ppv / (se + ppv) if se + ppv else 0.0,
                   "balanced_accuracy": (se + sp) / 2},
    }


def cross_validate(name: str, ds: BeatDataset, n_splits: int = 5) -> dict:
    """Patient-grouped CV on the training set: every fold tests on unseen people."""
    pred = np.zeros_like(ds.y)
    for train, test in GroupKFold(n_splits=n_splits).split(ds.X, ds.y, ds.groups):
        model = fit(name, ds.X[train], ds.y[train])
        pred[test] = model.predict(ds.X[test])
    return evaluate(ds.y, pred)


def intra_patient_demo(ds: BeatDataset, name: str) -> dict:
    """The *wrong* way: random beat split, same patients on both sides."""
    tr, te = train_test_split(np.arange(len(ds.y)), test_size=0.2, stratify=ds.y, random_state=0)
    model = fit(name, ds.X[tr], ds.y[tr])
    return evaluate(ds.y[te], model.predict(ds.X[te]))


def per_record(ds: BeatDataset, pred: np.ndarray) -> list[dict]:
    """Per-patient results: performance varies enormously from person to person."""
    rows = []
    for name in dict.fromkeys(ds.groups):
        sel = ds.groups == name
        t, p = ds.y[sel], pred[sel]
        abn = t != 0
        rows.append({
            "record": str(name), "beats": int(sel.sum()),
            "counts": {c: int(np.sum(t == i)) for i, c in enumerate(CLASSES)},
            "accuracy": float(np.mean(t == p)),
            "abnormal_sensitivity": float(np.mean(p[abn] != 0)) if abn.any() else None,
            "normal_specificity": float(np.mean(p[~abn] == 0)) if (~abn).any() else None,
        })
    return rows


def selection_score(m: dict) -> float:
    """Macro-F1 over the four classes: rewards catching the rare abnormal classes."""
    return m["macro_f1"]


def train_and_evaluate(models: list[str] | None = None, verbose: bool = True) -> dict:
    """Full experiment: grouped CV on DS1, refit on DS1, final test on DS2."""
    import joblib

    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    train, test = load_dataset(data.DS1), load_dataset(data.DS2)
    log(f"DS1 train: {len(train.y)} beats  {np.bincount(train.y).tolist()}  |  "
        f"DS2 test: {len(test.y)} beats  {np.bincount(test.y).tolist()}  ({time.time() - t0:.0f}s)")

    names = models or list(candidate_models())
    results: dict = {"classes": list(CLASSES), "features": FEATURE_NAMES, "models": {}}
    fitted = {}
    for name in names:
        t = time.time()
        cv = cross_validate(name, train)
        model = fit(name, train.X, train.y)
        pred = model.predict(test.X)
        test_m = evaluate(test.y, pred)
        fitted[name] = model
        results["models"][name] = {"cv_ds1": cv, "test_ds2": test_m, "weight_power": WEIGHT_POWER[name],
                                   "test_ds2_per_record": per_record(test, pred)}
        log(f"{name:20s} CV macro-F1 {cv['macro_f1']:.3f} | DS2 acc {test_m['accuracy']:.3f} "
            f"macro-F1 {test_m['macro_f1']:.3f} | abnormal Se {test_m['binary']['sensitivity']:.3f} "
            f"Sp {test_m['binary']['specificity']:.3f} | "
            + " ".join(f"{c}:Se{v['sensitivity']:.2f}/P{v['ppv']:.2f}" for c, v in test_m["per_class"].items())
            + f" ({time.time() - t:.0f}s)")

    best = max(names, key=lambda n: selection_score(results["models"][n]["cv_ds1"]))
    results["selected_model"] = best
    log(f"Selected by DS1 cross-validation: {best}")

    results["intra_patient_demo"] = intra_patient_demo(
        BeatDataset(*(np.concatenate([getattr(train, f), getattr(test, f)])
                      for f in ("X", "y", "groups", "positions"))), best)
    keep = test.groups != "232"
    results["test_ds2_without_232"] = evaluate(test.y[keep], fitted[best].predict(test.X[keep]))
    log(f"Intra-patient (leaky) accuracy: {results['intra_patient_demo']['accuracy']:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": fitted[best], "name": best, "features": FEATURE_NAMES,
                 "classes": CLASSES, "filter": TRAINING_FILTER.to_dict()}, MODELS_DIR / "beat_classifier.joblib")
    (MODELS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    results["_fitted"] = fitted
    return results


WEB_MODEL = data.ROOT / "docs" / "data" / "model.json"


class JsonModel:
    """Pure-NumPy inference for the exported ``docs/data/model.json``.

    This is the exact model the website runs, so the Python report works right
    after cloning the repository: no retraining and no pickle files.
    """

    def __init__(self, spec: dict):
        self.spec = spec

    def _raw(self, X: np.ndarray) -> np.ndarray:
        s = self.spec
        if s["type"] in ("logistic_regression", "mlp"):
            Z = (X - np.asarray(s["scaler"]["mean"])) / np.asarray(s["scaler"]["scale"])
            if s["type"] == "logistic_regression":
                return Z @ np.asarray(s["coef"]).T + np.asarray(s["intercept"])
            for i, layer in enumerate(s["layers"]):
                Z = Z @ np.asarray(layer["W"]) + np.asarray(layer["b"])
                if i < len(s["layers"]) - 1:
                    Z = np.maximum(Z, 0)
            return Z
        if s["type"] == "gradient_boosting":
            raw = np.tile(np.asarray(s["baseline"]), (len(X), 1))
            rows = np.arange(len(X))
            for iteration in s["trees"]:
                for k, tree in enumerate(iteration):
                    f, t, l, r = (np.asarray(tree[key]) for key in ("f", "t", "l", "r"))
                    node = np.zeros(len(X), dtype=np.int64)
                    while (f[node] >= 0).any():
                        split = f[node] >= 0
                        go_left = X[rows, np.maximum(f[node], 0)] <= t[node]
                        node = np.where(split, np.where(go_left, l[node], r[node]), node)
                    raw[:, k] += t[node]
            return raw
        raise ValueError(f"unknown model type {s['type']!r}")

    def predict_proba(self, X) -> np.ndarray:
        z = self._raw(np.asarray(X, dtype=np.float64))
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


def load_classifier(path: Path | None = None) -> dict:
    """Load the trained classifier: the joblib file from `train` if present, else the exported JSON model."""
    path = path or MODELS_DIR / "beat_classifier.joblib"
    if path.exists():
        import joblib

        return joblib.load(path)
    if WEB_MODEL.exists():
        spec = json.loads(WEB_MODEL.read_text())
        return {"model": JsonModel(spec), "name": spec["type"], "features": spec["features"],
                "classes": tuple(spec["classes"]), "filter": spec["filter"]}
    raise FileNotFoundError(f"No model found. Run `python -m ecglab train` (expected {path}).")
