"""Train, calibrate and select on the validation part. Score the test part once. Package the result.

Order of use of the three parts:
* train - fit the model and build the nearest-neighbour index
* val   - choose hyperparameters / the best epoch, fit the temperature, choose the threshold
* test  - one pass at the end, after every choice is fixed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import BANNER, __version__, metrics
from .dataset import Index, load_gray
from .explain import knn, narrative, saliency
from .models import Classifier, LightClassifier


def load_images(frame: pd.DataFrame, size: int) -> list[np.ndarray]:
    return [load_gray(p, size) for p in frame["path"]]


@dataclass
class Bundle:
    """Everything that the CLI and the app need. The app loads this object once and caches it."""

    classifier: Classifier
    temperature: float
    threshold: float
    target_sensitivity: float
    index: knn.TrainIndex
    image_size: int
    val_summary: dict
    version: str = __version__
    notes: list[str] = field(default_factory=list)

    def probability(self, images: list[np.ndarray]) -> np.ndarray:
        return metrics.sigmoid(self.classifier.logits(images) / self.temperature)

    def analyze(self, image: np.ndarray, k: int = 10, client=None, send_image: bool = False,
                patch: int = 8, stride: int = 4) -> dict:
        """Prediction plus three explanations. The image stays in memory. Nothing is written to disk."""
        img = np.asarray(image, dtype=np.float32)
        if img.ndim == 3:
            img = img.mean(axis=-1)
        if img.max() > 1.0:
            img = img / 255.0
        if img.shape != (self.image_size, self.image_size):
            from PIL import Image
            img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        p = float(self.probability([img])[0])
        neighbours = self.index.query(self.classifier.embed([img])[0], k)
        heat = saliency.occlusion(self.classifier.logits, img, patch, stride)
        zone, share = saliency.top_zone(heat)
        result = {
            "probability": p, "threshold": self.threshold,
            "flag": "above threshold: review for pneumonia" if p >= self.threshold else "below threshold",
            "neighbours": [nb.__dict__ for nb in neighbours],
            "neighbours_summary": knn.neighbour_summary(neighbours),
            "saliency_zone": zone, "saliency_zone_share": share, "heat": heat, "image": img,
            "banner": BANNER,
        }
        result["summary"] = narrative.summarise(result, client, img, send_image)
        return result


def train_and_select(index: Index, classifier: Classifier, settings) -> tuple[Bundle, dict]:
    """Fit on train, select and calibrate on val. The test part is not read here."""
    size = settings.image_size
    tr, va = index.part("train"), index.part("val")
    Xtr, Xva = load_images(tr, size), load_images(va, size)
    classifier.fit(Xtr, tr["y"].to_numpy(), Xva, va["y"].to_numpy())
    z_val = classifier.logits(Xva)
    temperature = metrics.fit_temperature(z_val, va["y"].to_numpy())
    p_val = metrics.sigmoid(z_val / temperature)
    threshold = metrics.threshold_for_sensitivity(va["y"].to_numpy(), p_val, settings.target_sensitivity)
    train_index = knn.TrainIndex(classifier.embed(Xtr), tr["y"].to_numpy(), list(tr["file"]))
    val_summary = metrics.summary(va["y"].to_numpy(), p_val, threshold)
    bundle = Bundle(classifier, temperature, threshold, settings.target_sensitivity, train_index, size, val_summary)
    info = {"train_images": len(tr), "val_images": len(va), "temperature": temperature, "threshold": threshold,
            "selection": getattr(classifier, "selection_", None) or getattr(getattr(classifier, "log_", None),
                                                                            "history", None)}
    return bundle, info


def evaluate_test(bundle: Bundle, index: Index, settings) -> dict:
    """The single pass over the test part."""
    te = index.part("test")
    p = bundle.probability(load_images(te, bundle.image_size))
    y, groups = te["y"].to_numpy(), te["patient"].to_numpy()
    at_operating = metrics.summary(y, p, bundle.threshold)
    at_half = metrics.summary(y, p, 0.5)
    ci = metrics.bootstrap(y, p, bundle.threshold, groups, settings.n_bootstrap, settings.seed)
    subtypes = {}
    for sub, part in te.groupby("subtype"):
        mask = te["subtype"].to_numpy() == sub
        flagged = p[mask] >= bundle.threshold
        subtypes[sub] = {"images": int(mask.sum()), "flagged_share": float(flagged.mean())}
    return {"test_images": int(len(te)), "test_patients": int(te["patient"].nunique()),
            "prevalence": float(y.mean()), "at_operating_threshold": at_operating, "at_0_5": at_half,
            "ci": ci, "by_subtype": subtypes}


def sanity_check(bundle: Bundle, index: Index, settings, n_images: int = 5) -> dict:
    """Model-randomisation test: refit the light model on shuffled labels and compare saliency maps."""
    if not isinstance(bundle.classifier, LightClassifier):
        return {"skipped": "only for the light classifier"}
    tr, va = index.part("train"), index.part("val")
    rng = np.random.default_rng(settings.seed)
    Xtr, Xva = load_images(tr, settings.image_size), load_images(va, settings.image_size)
    rnd = LightClassifier(seed=settings.seed).fit(Xtr, rng.permutation(tr["y"].to_numpy()), Xva,
                                                  rng.permutation(va["y"].to_numpy()))
    rhos = []
    for img in Xva[:n_images]:
        rhos.append(saliency.randomisation_check(saliency.occlusion(bundle.classifier.logits, img),
                                                 saliency.occlusion(rnd.logits, img)))
    return {"images": len(rhos), "mean_spearman": float(np.mean(rhos)), "max_spearman": float(np.max(rhos))}


def save_bundle(bundle: Bundle, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: str | Path) -> Bundle:
    """joblib runs code on load: open only bundles that you made."""
    obj = joblib.load(path)
    if not isinstance(obj, Bundle):
        raise TypeError(f"{path} is not a pneumovit-explain bundle")
    return obj


def zone_agreement(bundle: Bundle, index: Index, zones: dict[str, str]) -> dict:
    """Share of test images with a known finding zone where the top saliency zone is that zone.

    Only synthetic data has known zones. Chance level is 1/6 (six lung zones).
    """
    te = index.part("test")
    te = te[te["file"].map(lambda f: zones.get(f, "-") != "-")]
    hits = []
    for _, row in te.iterrows():
        heat = saliency.occlusion(bundle.classifier.logits, load_gray(row["path"], bundle.image_size))
        hits.append(saliency.top_zone(heat)[0] == zones[row["file"]])
    return {"images": len(hits), "agreement": float(np.mean(hits)) if hits else float("nan"), "chance": 1 / 6}
