"""Deep classifiers (optional extra: pip install 'pneumovit-explain[deep]'). torch and timm load lazily.

* `vit_scratch`: a Vision Transformer written here and trained from random weights (the ablation).
* `vit_small_dino`, `densenet121`, `convnext_tiny`: pretrained timm backbones (the main models).

Training uses class weights, a cosine learning-rate schedule over the real number of epochs,
early stopping on validation AUC, and saves the best-validation weights to ONE file. The
service and the app load that same file.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

TIMM_BACKBONES = {
    "vit_small_dino": "vit_small_patch16_224.dino",
    "densenet121": "densenet121.tv_in1k",
    "convnext_tiny": "convnext_tiny.fb_in1k",
}


def require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError("deep models need: pip install 'pneumovit-explain[deep]'") from exc
    return torch


def pick_device(device: str = "auto"):
    """'auto' gives CUDA when it exists, else CPU. A CPU-only machine never gets a CUDA device."""
    torch = require_torch()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


@dataclass
class ViTConfig:
    image_size: int = 224
    patch: int = 16
    dim: int = 384
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1


def build_vit(cfg: ViTConfig):
    """A pre-norm Vision Transformer for 1-channel images with a CLS token and learned positions."""
    torch = require_torch()
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(cfg.dim)
            self.attn = nn.MultiheadAttention(cfg.dim, cfg.heads, dropout=cfg.dropout, batch_first=True)
            self.norm2 = nn.LayerNorm(cfg.dim)
            hidden = int(cfg.dim * cfg.mlp_ratio)
            self.mlp = nn.Sequential(nn.Linear(cfg.dim, hidden), nn.GELU(), nn.Dropout(cfg.dropout),
                                     nn.Linear(hidden, cfg.dim), nn.Dropout(cfg.dropout))
            self.last_attention = None

        def forward(self, x):
            h = self.norm1(x)
            out, weights = self.attn(h, h, h, need_weights=True, average_attn_weights=True)
            self.last_attention = weights.detach()  # (B, tokens, tokens), mean over heads
            x = x + out
            return x + self.mlp(self.norm2(x))

    class ViT(nn.Module):
        def __init__(self):
            super().__init__()
            n = (cfg.image_size // cfg.patch) ** 2
            self.patch_embed = nn.Conv2d(1, cfg.dim, cfg.patch, cfg.patch)
            self.cls = nn.Parameter(torch.zeros(1, 1, cfg.dim))
            self.pos = nn.Parameter(torch.randn(1, n + 1, cfg.dim) * 0.02)
            self.drop = nn.Dropout(cfg.dropout)
            self.blocks = nn.ModuleList([Block() for _ in range(cfg.depth)])
            self.norm = nn.LayerNorm(cfg.dim)
            self.head = nn.Linear(cfg.dim, 1)

        def features(self, x):
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1) + self.pos
            x = self.drop(x)
            for blk in self.blocks:
                x = blk(x)
            return self.norm(x)[:, 0]

        def forward(self, x):
            return self.head(self.features(x)).squeeze(1)

    return ViT()


def attention_rollout(model, discard_ratio: float = 0.0) -> np.ndarray:
    """Attention rollout (Abnar and Zuidema, 2020) over all blocks of the last forward pass.

    Returns the CLS-to-patch relevance as a square grid per image: (B, side, side).
    """
    torch = require_torch()
    mats = [b.last_attention for b in model.blocks]
    if any(m is None for m in mats):
        raise RuntimeError("run a forward pass first")
    result = torch.eye(mats[0].shape[-1]).expand_as(mats[0]).clone()
    for a in mats:
        if discard_ratio > 0:
            flat = a.flatten(1)
            k = int(flat.shape[1] * discard_ratio)
            low = flat.topk(k, dim=1, largest=False).indices
            flat.scatter_(1, low, 0)
            a = flat.view_as(a)
        a = a + torch.eye(a.shape[-1])
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ result
    cls = result[:, 0, 1:]
    side = int(np.sqrt(cls.shape[-1]))
    grid = cls.reshape(-1, side, side)
    grid = grid / grid.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    return grid.cpu().numpy()


@dataclass
class TrainConfig:
    epochs: int = 30
    patience: int = 5
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.05
    image_size: int = 224


@dataclass
class TrainLog:
    best_epoch: int = -1
    best_val_auc: float = -1.0
    history: list[dict] = field(default_factory=list)


def _to_tensor(images, size: int, channels: int, mean: float, std: float, train: bool, rng):
    torch = require_torch()
    from PIL import Image
    out = []
    for im in images:
        arr = np.asarray(Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)).resize(
            (size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        if train:  # small, anatomy-preserving augmentation (no flips: left and right matter)
            arr = np.roll(arr, tuple(rng.integers(-size // 32, size // 32 + 1, 2)), axis=(0, 1))
            arr = np.clip(arr * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05), 0, 1)
        out.append((arr - mean) / std)
    x = torch.from_numpy(np.stack(out)[:, None])
    return x.repeat(1, channels, 1, 1) if channels > 1 else x


class TorchClassifier:
    def __init__(self, name: str = "vit_scratch", seed: int = 0, device: str = "auto",
                 train_config: TrainConfig | None = None, vit_config: ViTConfig | None = None,
                 pretrained: bool = True):
        self.name = name
        self.seed = seed
        self.device = device
        self.tc = train_config or TrainConfig()
        self.vit_config = vit_config or ViTConfig(image_size=self.tc.image_size)
        self.pretrained = pretrained

    def _build(self):
        torch = require_torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if self.name == "vit_scratch":
            self.channels_, self.mean_, self.std_ = 1, 0.5, 0.25
            return build_vit(self.vit_config)
        if self.name not in TIMM_BACKBONES:
            raise ValueError(f"unknown deep model {self.name!r}")
        import timm
        model = timm.create_model(TIMM_BACKBONES[self.name], pretrained=self.pretrained, num_classes=1)
        dc = timm.data.resolve_model_data_config(model)
        self.channels_ = 3
        self.mean_, self.std_ = float(np.mean(dc["mean"])), float(np.mean(dc["std"]))
        self.tc.image_size = int(dc["input_size"][-1])
        return model

    def fit(self, images, y, val_images, val_y, checkpoint: str | Path | None = None):
        torch = require_torch()
        from sklearn.metrics import roc_auc_score
        dev = pick_device(self.device)
        self.model_ = self._build().to(dev)
        y = np.asarray(y, dtype=np.float32)
        pos_weight = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)], device=dev)  # class weights
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.tc.lr, weight_decay=self.tc.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(self.tc.epochs, 1))
        rng = np.random.default_rng(self.seed)
        self.log_ = TrainLog()
        best_state, stale = None, 0
        for epoch in range(self.tc.epochs):
            self.model_.train()
            order = rng.permutation(len(images))
            losses = []
            for s in range(0, len(order), self.tc.batch_size):
                idx = order[s:s + self.tc.batch_size]
                xb = _to_tensor([images[i] for i in idx], self.tc.image_size, self.channels_, self.mean_,
                                self.std_, True, rng).to(dev)
                yb = torch.from_numpy(y[idx]).to(dev)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self.model_(xb).reshape(-1), yb)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu()))
            sched.step()
            val_auc = float(roc_auc_score(val_y, self.logits(val_images)))
            self.log_.history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_auc": val_auc,
                                      "lr": float(sched.get_last_lr()[0])})
            if val_auc > self.log_.best_val_auc:
                self.log_.best_val_auc, self.log_.best_epoch = val_auc, epoch
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in self.model_.state_dict().items()})
                stale = 0
            else:
                stale += 1
                if stale >= self.tc.patience:
                    break
        self.model_.load_state_dict(best_state)
        if checkpoint is not None:
            self.save(checkpoint)
        return self

    def logits(self, images) -> np.ndarray:
        torch = require_torch()
        dev = pick_device(self.device)
        self.model_.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(images), 64):
                xb = _to_tensor(images[s:s + 64], self.tc.image_size, self.channels_, self.mean_, self.std_,
                                False, None).to(dev)
                out.append(self.model_(xb).reshape(-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def embed(self, images) -> np.ndarray:
        torch = require_torch()
        dev = pick_device(self.device)
        self.model_.eval()
        feats = []
        with torch.no_grad():
            for s in range(0, len(images), 64):
                xb = _to_tensor(images[s:s + 64], self.tc.image_size, self.channels_, self.mean_, self.std_,
                                False, None).to(dev)
                f = self.model_.features(xb) if self.name == "vit_scratch" else self.model_.forward_head(
                    self.model_.forward_features(xb), pre_logits=True)
                feats.append(f.cpu().numpy())
        return np.concatenate(feats)

    def __getstate__(self):
        """Pickle the weights, not the module (the ViT class is built inside a function)."""
        state = self.__dict__.copy()
        model = state.pop("model_", None)
        if model is not None:
            state["_state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        return state

    def __setstate__(self, state):
        weights = state.pop("_state_dict", None)
        self.__dict__.update(state)
        if weights is not None:
            self.pretrained = False
            self.model_ = self._build()
            self.model_.load_state_dict(weights)
            self.model_.to(pick_device(self.device))

    def rollout(self, image) -> np.ndarray:
        if self.name != "vit_scratch":
            raise ValueError("attention rollout needs the vit_scratch model")
        self.logits([image])
        return attention_rollout(self.model_)[0]

    def save(self, path: str | Path) -> Path:
        torch = require_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"name": self.name, "state_dict": self.model_.state_dict(), "train_config": asdict(self.tc),
                    "vit_config": asdict(self.vit_config), "log": asdict(self.log_)}, path)
        path.with_suffix(".json").write_text(json.dumps({"checkpoint": path.name, "best_epoch": self.log_.best_epoch,
                                                         "best_val_auc": self.log_.best_val_auc}, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "TorchClassifier":
        torch = require_torch()
        blob = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(blob["name"], device=device, train_config=TrainConfig(**blob["train_config"]),
                  vit_config=ViTConfig(**blob["vit_config"]), pretrained=False)
        obj.model_ = obj._build()
        obj.model_.load_state_dict(blob["state_dict"])
        obj.model_.to(pick_device(device))
        obj.log_ = TrainLog(**blob["log"])
        return obj
