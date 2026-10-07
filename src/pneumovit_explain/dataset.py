"""Index of the chest X-ray folder with patient IDs, and a patient-aware train/val/test re-split.

Expected layout (Kermany et al.): <root>/{train,val,test}/{NORMAL,PNEUMONIA}/<file>. The
official `val/` part has only 16 images, so all parts are pooled and split again by patient.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedGroupKFold

from . import CLASSES

SUFFIXES = {".jpeg", ".jpg", ".png"}
PNEUMONIA_RX = re.compile(r"^(person\d+)_(bacteria|virus)_\d+", re.I)
NORMAL_RX = re.compile(r"^((?:NORMAL2-)?IM-\d+)-\d+", re.I)


class DatasetError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors[:10]))
        self.errors = errors


def parse_patient(file_name: str, label: str) -> tuple[str, str]:
    """Return (patient id, subtype) from a Kermany file name. Unknown names get their own ID."""
    stem = Path(file_name).stem
    m = PNEUMONIA_RX.match(stem)
    if m:
        return f"P:{m.group(1).lower()}", m.group(2).lower()
    m = NORMAL_RX.match(stem)
    if m:
        return f"N:{m.group(1).upper()}", "normal"
    return f"U:{label}:{stem}", "normal" if label == "NORMAL" else "unknown"


@dataclass
class Index:
    frame: pd.DataFrame  # path, file, label, y, patient, subtype, official_split, split
    warnings: list[str] = field(default_factory=list)

    def part(self, name: str) -> pd.DataFrame:
        return self.frame[self.frame["split"] == name].reset_index(drop=True)


def load_gray(path, size: int | None = None) -> np.ndarray:
    """Decode to grey float32 in [0, 1], optionally resized to size x size."""
    with Image.open(path) as im:
        im = im.convert("L")
        if size is not None:
            im = im.resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


def scan(root: str | Path, check_decode: bool = True) -> Index:
    root = Path(root)
    if not root.is_dir():
        raise DatasetError([f"{root} is not a folder. See data/README.md or run `pneumovit-explain synth`."])
    rows, errors, warnings = [], [], []
    for official in ("train", "val", "test"):
        for label in CLASSES:
            folder = root / official / label
            if not folder.is_dir():
                continue
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() not in SUFFIXES:
                    continue
                if check_decode:
                    try:
                        with Image.open(p) as im:
                            im.verify()
                    except (UnidentifiedImageError, OSError):
                        errors.append(f"cannot decode {p}")
                        continue
                patient, subtype = parse_patient(p.name, label)
                rows.append({"path": str(p), "file": p.name, "label": label, "y": int(label == "PNEUMONIA"),
                             "patient": patient, "subtype": subtype, "official_split": official})
    if not rows:
        errors.append(f"no images under {root}/{{train,val,test}}/{{NORMAL,PNEUMONIA}}")
    if errors:
        raise DatasetError(errors)
    frame = pd.DataFrame(rows)
    if frame["y"].nunique() < 2:
        raise DatasetError(["both classes are needed"])
    unknown = int(frame["patient"].str.startswith("U:").sum())
    if unknown:
        warnings.append(f"{unknown} file names have no patient ID (each one is its own patient)")
    spans = frame.groupby("patient")["official_split"].nunique()
    if (spans > 1).any():
        warnings.append(f"{int((spans > 1).sum())} patients appear in more than one official split")
    mixed = frame.groupby("patient")["y"].nunique()
    if (mixed > 1).any():
        warnings.append(f"{int((mixed > 1).sum())} patient IDs have both labels")
    return Index(frame, warnings)


def patient_split(index: Index, val_fraction: float = 0.15, test_fraction: float = 0.2, seed: int = 42) -> Index:
    """Assign `split` = train / val / test so that each patient is in exactly one part (label-stratified)."""
    f = index.frame.copy()
    y, groups = f["y"].to_numpy(), f["patient"].to_numpy()

    def take(mask, fraction, seed_):
        idx = np.flatnonzero(mask)
        n_splits = max(2, int(round(1 / fraction)))
        if len(np.unique(groups[idx])) < n_splits:
            raise DatasetError([f"too few patients ({len(np.unique(groups[idx]))}) for the split"])
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed_)
        _, picked = next(sgkf.split(idx, y[idx], groups[idx]))
        return idx[picked]

    f["split"] = "train"
    test_idx = take(np.ones(len(f), bool), test_fraction, seed)
    f.loc[test_idx, "split"] = "test"
    rest = (f["split"] == "train").to_numpy()
    val_idx = take(rest, val_fraction / (1 - test_fraction), seed + 1)
    f.loc[val_idx, "split"] = "val"
    check = f.groupby("patient")["split"].nunique()
    assert (check == 1).all(), "a patient crosses the split"
    return Index(f, list(index.warnings))
