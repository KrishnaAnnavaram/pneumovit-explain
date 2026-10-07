"""Classifiers behind one interface. The light classifier runs on CPU with scikit-learn and needs no download.

Interface (also used by `deep.TorchClassifier`):
* fit(train_images, train_y, val_images, val_y) - selection and early stopping use the val part only
* logits(images) -> raw scores for PNEUMONIA (before temperature scaling)
* embed(images) -> vectors for the nearest-neighbour index
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


class Classifier(Protocol):
    name: str

    def fit(self, images, y, val_images, val_y): ...

    def logits(self, images) -> np.ndarray: ...

    def embed(self, images) -> np.ndarray: ...


ZONES = [f"{side} {level}" for level in ("upper", "middle", "lower") for side in ("right", "left")]


def zone_slices(size: int) -> dict[str, tuple[slice, slice]]:
    """Six lung zones. Radiographic convention: the patient's right side is on the left of the image."""
    rows = {"upper": slice(int(size * 0.2), int(size * 0.42)), "middle": slice(int(size * 0.42), int(size * 0.58)),
            "lower": slice(int(size * 0.58), int(size * 0.8))}
    cols = {"right": slice(int(size * 0.15), int(size * 0.47)), "left": slice(int(size * 0.53), int(size * 0.85))}
    return {f"{s} {lv}": (rows[lv], cols[s]) for lv in ("upper", "middle", "lower") for s in ("right", "left")}


def xray_features(img: np.ndarray) -> np.ndarray:
    """Intensity histogram, per-zone mean/std/texture, gradient histogram and an 8x8 thumbnail."""
    size = img.shape[0]
    hist = np.histogram(img, bins=16, range=(0, 1))[0] / img.size
    zones = []
    gy, gx = np.gradient(img)
    grad = np.hypot(gx, gy)
    for rs, cs in zone_slices(size).values():
        z = img[rs, cs]
        zones += [z.mean(), z.std(), grad[rs, cs].mean()]
    ghist = np.histogram(grad, bins=8, range=(0, 0.2))[0] / grad.size
    h = size - size % 8
    thumb = img[:h, :h].reshape(8, h // 8, 8, h // 8).mean(axis=(1, 3)).ravel()
    return np.concatenate([hist, zones, ghist, thumb]).astype(np.float32)


class LightClassifier:
    """Logistic regression on handcrafted X-ray features. The regularisation C is chosen on the val part."""

    name = "light_logreg"

    def __init__(self, seed: int = 0, grid=(0.01, 0.1, 1.0, 10.0)):
        self.seed = seed
        self.grid = grid

    def _X(self, images) -> np.ndarray:
        return np.stack([xray_features(im) for im in images])

    def fit(self, images, y, val_images, val_y):
        X, Xv = self._X(images), self._X(val_images)
        self.scaler_ = StandardScaler().fit(X)
        Xs, Xvs = self.scaler_.transform(X), self.scaler_.transform(Xv)
        self.selection_ = []
        best = None
        for C in self.grid:
            m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=self.seed).fit(Xs, y)
            auc = roc_auc_score(val_y, m.decision_function(Xvs))
            self.selection_.append({"C": C, "val_auc": float(auc)})
            if best is None or auc > best[0]:
                best = (auc, m)
        self.val_auc_, self.model_ = best
        return self

    def logits(self, images) -> np.ndarray:
        return self.model_.decision_function(self.scaler_.transform(self._X(images)))

    def embed(self, images) -> np.ndarray:
        return self.scaler_.transform(self._X(images))
