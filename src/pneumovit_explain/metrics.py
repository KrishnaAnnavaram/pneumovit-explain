"""Screening metrics, calibration, operating thresholds and patient-cluster bootstrap CIs."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

EPS = 1e-6


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def fit_temperature(val_logits: np.ndarray, val_y: np.ndarray) -> float:
    """Temperature T > 0 that minimises the validation log loss of sigmoid(logit / T)."""
    z, y = np.asarray(val_logits, float), np.asarray(val_y, float)

    def nll(log_t):
        p = np.clip(sigmoid(z / np.exp(log_t)), EPS, 1 - EPS)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    res = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    return float(np.exp(res.x))


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """The highest threshold whose sensitivity on (y, p) is at least `target`."""
    pos = np.sort(np.asarray(p)[np.asarray(y) == 1])
    if pos.size == 0:
        raise ValueError("no positive cases to set a threshold")
    k = int(np.floor((1 - target) * pos.size))  # number of positives that can fall below the threshold
    return float(pos[k])


def confusion(y, p, threshold: float) -> dict:
    y = np.asarray(y)
    pred = np.asarray(p) >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    fp = int(np.sum(pred & (y == 0)))
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp}


def _rates(c: dict) -> dict:
    def div(a, b):
        return a / b if b else float("nan")
    return {"sensitivity": div(c["tp"], c["tp"] + c["fn"]), "specificity": div(c["tn"], c["tn"] + c["fp"]),
            "ppv": div(c["tp"], c["tp"] + c["fp"]), "npv": div(c["tn"], c["tn"] + c["fn"]),
            "accuracy": div(c["tp"] + c["tn"], sum(c.values()))}


def ece(y, p, n_bins: int = 10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    idx = np.clip(np.digitize(p, np.linspace(0, 1, n_bins + 1)[1:-1]), 0, n_bins - 1)
    return float(sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean())
                     for b in range(n_bins) if (idx == b).any()))


def summary(y, p, threshold: float) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    c = confusion(y, p, threshold)
    out = {"auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
           "brier": float(np.mean((p - y) ** 2)), "ece": ece(y, p), "threshold": float(threshold),
           "confusion": c, **_rates(c)}
    return out


def bootstrap(y, p, threshold: float, groups, n_boot: int = 1000, seed: int = 0) -> dict:
    """95% percentile CIs that resample whole patients (images of one patient are not independent)."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    units = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    names = ("auc", "sensitivity", "specificity")
    stats = {k: [] for k in names}
    for _ in range(n_boot):
        idx = np.concatenate([units[i] for i in rng.integers(0, len(units), len(units))])
        if len(np.unique(y[idx])) < 2:
            continue
        stats["auc"].append(roc_auc_score(y[idx], p[idx]))
        r = _rates(confusion(y[idx], p[idx], threshold))
        stats["sensitivity"].append(r["sensitivity"])
        stats["specificity"].append(r["specificity"])
    point = summary(y, p, threshold)
    return {k: {"value": point[k], "low": float(np.nanquantile(v, 0.025)), "high": float(np.nanquantile(v, 0.975))}
            for k, v in stats.items()}
