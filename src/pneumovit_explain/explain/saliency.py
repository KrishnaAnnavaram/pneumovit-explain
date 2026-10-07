"""Occlusion saliency for any classifier, a zone summary, and a model-randomisation sanity check."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr

from ..models import zone_slices

LogitFn = Callable[[list[np.ndarray]], np.ndarray]


def occlusion(logit_fn: LogitFn, img: np.ndarray, patch: int = 8, stride: int = 4) -> np.ndarray:
    """Drop of the PNEUMONIA logit when each patch is replaced by a blurred copy (same size as img).

    The blurred copy removes local detail (for example an opacity) but keeps the local
    brightness, so the occlusion does not create an artificial dark or bright square.
    Positive values mark regions that push the model toward PNEUMONIA.
    """
    h, w = img.shape
    fill = ndimage.gaussian_filter(img, sigma=patch)
    base = float(logit_fn([img])[0])
    batch, boxes = [], []
    for y0 in range(0, h - patch + 1, stride):
        for x0 in range(0, w - patch + 1, stride):
            im = img.copy()
            im[y0:y0 + patch, x0:x0 + patch] = fill[y0:y0 + patch, x0:x0 + patch]
            batch.append(im)
            boxes.append((y0, x0))
    scores = logit_fn(batch)
    heat = np.zeros((h, w))
    count = np.zeros((h, w))
    for (y0, x0), s in zip(boxes, scores):
        heat[y0:y0 + patch, x0:x0 + patch] += base - s
        count[y0:y0 + patch, x0:x0 + patch] += 1
    return heat / np.maximum(count, 1)


def zone_scores(heat: np.ndarray) -> dict[str, float]:
    """Mean positive evidence in each of the six lung zones."""
    pos = np.clip(heat, 0, None)
    return {z: float(pos[rs, cs].mean()) for z, (rs, cs) in zone_slices(heat.shape[0]).items()}


def top_zone(heat: np.ndarray) -> tuple[str, float]:
    scores = zone_scores(heat)
    total = sum(scores.values())
    zone = max(scores, key=scores.get)
    return zone, (scores[zone] / total if total > 0 else 0.0)


def randomisation_check(heat_trained: np.ndarray, heat_random: np.ndarray) -> float:
    """Spearman correlation of two saliency maps. A high value means the map ignores what the model learned."""
    rho = spearmanr(heat_trained.ravel(), heat_random.ravel()).correlation
    return float(0.0 if np.isnan(rho) else rho)


def save_overlay(img: np.ndarray, heat: np.ndarray, path: str | Path) -> Path:
    """Raw grey image (no re-normalisation) next to the same image with positive evidence in red."""
    grey = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([grey] * 3, axis=-1).astype(float)
    pos = np.clip(heat, 0, None)
    pos = pos / pos.max() if pos.max() > 0 else pos
    over = rgb.copy()
    over[..., 0] = rgb[..., 0] * (1 - pos) + 255 * pos
    over[..., 1:] = rgb[..., 1:] * (1 - 0.7 * pos[..., None])
    panel = np.concatenate([rgb, over], axis=1).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel).resize((panel.shape[1] * 4, panel.shape[0] * 4), Image.Resampling.NEAREST).save(path)
    return path
