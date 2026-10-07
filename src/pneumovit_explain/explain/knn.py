"""Nearest-neighbour index over TRAINING images of both classes. The query never filters by the prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import CLASSES


@dataclass
class Neighbour:
    file: str
    label: str
    similarity: float


class TrainIndex:
    """Cosine search with NumPy (FAISS is an optional extra for large indexes)."""

    def __init__(self, vectors: np.ndarray, labels: np.ndarray, files: list[str], use_faiss: bool = False):
        v = np.asarray(vectors, dtype=np.float32)
        self.vectors = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        self.labels = np.asarray(labels, dtype=int)
        self.files = list(files)
        self._faiss = None
        if use_faiss:  # pragma: no cover - optional extra
            import faiss
            self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
            self._faiss.add(self.vectors)

    def __len__(self) -> int:
        return len(self.files)

    def query(self, vector: np.ndarray, k: int = 10) -> list[Neighbour]:
        q = np.asarray(vector, dtype=np.float32).ravel()
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        k = min(k, len(self))
        if self._faiss is not None:  # pragma: no cover
            sims, idx = self._faiss.search(q[None], k)
            pairs = zip(idx[0], sims[0])
        else:
            sims = self.vectors @ q
            top = np.argsort(-sims)[:k]
            pairs = zip(top, sims[top])
        return [Neighbour(self.files[i], CLASSES[self.labels[i]], float(s)) for i, s in pairs]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vectors=self.vectors, labels=self.labels, files=np.asarray(self.files))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrainIndex":
        data = np.load(path, allow_pickle=False)
        return cls(data["vectors"], data["labels"], [str(f) for f in data["files"]])


def neighbour_summary(neighbours: list[Neighbour]) -> dict:
    n_pos = sum(nb.label == "PNEUMONIA" for nb in neighbours)
    return {"k": len(neighbours), "pneumonia": n_pos, "normal": len(neighbours) - n_pos,
            "pneumonia_share": n_pos / len(neighbours) if neighbours else float("nan")}
