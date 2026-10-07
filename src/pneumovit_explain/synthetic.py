"""Synthetic chest-X-ray-like images in the Kermany folder layout, with several images per patient.

Pneumonia images get focal (bacterial) or diffuse (viral) opacities inside a lung field. The
generator is for offline tests and the demo only. It is not a model of real radiographs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage


def draw_xray(size: int, rng: np.random.Generator, kind: str) -> tuple[np.ndarray, str]:
    """Return (uint8 image, zone of the main finding or '-')."""
    yy, xx = np.mgrid[0:size, 0:size] / size
    img = np.zeros((size, size)) + 0.05
    torso = ((xx - 0.5) / 0.45) ** 2 + ((yy - 0.55) / 0.5) ** 2 <= 1
    img[torso] = 0.55
    lungs = []
    for cx in (0.32, 0.68):
        lung = ((xx - cx - rng.normal(0, 0.01)) / 0.14) ** 2 + ((yy - 0.5) / 0.3) ** 2 <= 1
        img[lung] = 0.22
        lungs.append(lung)
    img[(np.abs(xx - 0.5) < 0.04) & torso] = 0.8  # spine
    ribs = (np.sin(yy * 40 + rng.uniform(0, 6)) > 0.85) & (lungs[0] | lungs[1])
    img[ribs] += 0.08
    zone = "-"
    if kind in {"bacteria", "virus"}:
        side = int(rng.integers(0, 2))
        level = rng.choice(["upper", "middle", "lower"])
        if kind == "bacteria":  # a focal finding has a zone; diffuse viral haze has none
            zone = f"{'right' if side == 0 else 'left'} {level}"  # patient right = image left
        cy = {"upper": 0.33, "middle": 0.5, "lower": 0.66}[level]
        cx = 0.32 if side == 0 else 0.68
        if kind == "bacteria":
            blob = np.exp(-(((xx - cx) / 0.07) ** 2 + ((yy - cy) / 0.08) ** 2))
            img += rng.uniform(0.06, 0.25) * blob * (lungs[side] | (blob > 0.5))
        else:
            haze = ndimage.gaussian_filter(rng.random((size, size)), size / 20)
            haze = (haze - haze.min()) / (np.ptp(haze) + 1e-9)
            img += rng.uniform(0.04, 0.14) * haze * (lungs[0] | lungs[1])
    # soft-tissue shadows and exposure differences that appear in every class
    for _ in range(int(rng.integers(0, 3))):
        sy, sx = rng.uniform(0.3, 0.75), rng.uniform(0.2, 0.8)
        img += rng.uniform(0.03, 0.12) * np.exp(-(((xx - sx) / 0.12) ** 2 + ((yy - sy) / 0.1) ** 2))
    img = img * rng.uniform(0.8, 1.2) + rng.normal(0, 0.06, img.shape)
    img = ndimage.gaussian_filter(img, rng.uniform(0.6, 1.4))
    return (np.clip(img, 0, 1) * 255).astype(np.uint8), zone


def generate(root: str | Path, n_patients: int = 120, size: int = 64, pneumonia_share: float = 0.7,
             seed: int = 42) -> pd.DataFrame:
    """Write <root>/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg with Kermany-style names. Return the metadata.

    Each patient gets 1 to 4 images with small shifts. Like the public set, the official
    `val/` part is tiny, and some patients appear in two official parts.
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    rows = []
    for p in range(1, n_patients + 1):
        sick = rng.random() < pneumonia_share
        kind = rng.choice(["bacteria", "virus"], p=[0.65, 0.35]) if sick else "normal"
        label = "PNEUMONIA" if sick else "NORMAL"
        official = rng.choice(["train", "test", "val"], p=[0.88, 0.11, 0.01])
        for k in range(1, int(rng.integers(1, 5)) + 1):
            img, zone = draw_xray(size, rng, kind)
            shift = rng.integers(-2, 3, size=2)
            img = np.roll(img, tuple(shift), axis=(0, 1))
            part = official if rng.random() > 0.05 else "train"  # a few patients span two parts
            name = f"person{p}_{kind}_{k}.jpeg" if sick else f"IM-{p:04d}-{k:04d}.jpeg"
            folder = root / part / label
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(folder / name, quality=95)
            rows.append({"file": name, "label": label, "subtype": kind, "zone": zone, "official_split": part})
    meta = pd.DataFrame(rows)
    meta.to_csv(root / "metadata.csv", index=False)  # true finding zones, for the explanation check
    return meta
