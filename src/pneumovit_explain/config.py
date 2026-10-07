"""Settings from environment variables (and an optional local .env file). Keys are read, never printed."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

LLM_PROVIDERS = ("none", "fake", "gemini", "openai")


class ConfigError(ValueError):
    """A setting has an invalid value."""


def _read_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                v = v.strip().strip("'\"")
                if v:
                    out[k.strip()] = v
    return out


def _num(env, key, default, cast, low, high=None):
    raw = env.get(key)
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} has an invalid value") from exc
    if value < low or (high is not None and value > high):
        raise ConfigError(f"{key} must be in [{low}, {high}]")
    return value


@dataclass(frozen=True)
class Settings:
    data_dir: Path | None = None
    output_dir: Path = Path("outputs")
    model_dir: Path = Path("models")
    seed: int = 42
    image_size: int = 64
    val_fraction: float = 0.15
    test_fraction: float = 0.2
    target_sensitivity: float = 0.95
    n_bootstrap: int = 1000
    k_neighbours: int = 10
    device: str = "auto"
    llm_provider: str = "none"
    llm_model: str = ""
    llm_base_url: str = ""
    llm_send_image: bool = False
    api_keys: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.llm_provider not in LLM_PROVIDERS:
            raise ConfigError(f"PNEUMOVIT_LLM_PROVIDER must be one of {LLM_PROVIDERS}")
        if not 0.05 <= self.val_fraction <= 0.4 or not 0.05 <= self.test_fraction <= 0.4:
            raise ConfigError("val and test fractions must be in [0.05, 0.4]")
        if not 0.5 <= self.target_sensitivity < 1.0:
            raise ConfigError("target sensitivity must be in [0.5, 1)")

    def with_overrides(self, **changes) -> "Settings":
        return replace(self, **{k: v for k, v in changes.items() if v is not None})

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, dotenv: Path | None = Path(".env")) -> "Settings":
        merged: dict[str, str] = {}
        if env is None:
            if dotenv is not None:
                merged.update(_read_dotenv(dotenv))
            merged.update(os.environ)
        else:
            merged.update(env)
        data = merged.get("PNEUMOVIT_DATA_DIR")
        return cls(
            data_dir=Path(data) if data else None,
            output_dir=Path(merged.get("PNEUMOVIT_OUTPUT_DIR") or "outputs"),
            model_dir=Path(merged.get("PNEUMOVIT_MODEL_DIR") or "models"),
            seed=_num(merged, "PNEUMOVIT_SEED", 42, int, 0),
            image_size=_num(merged, "PNEUMOVIT_IMAGE_SIZE", 64, int, 32, 1024),
            val_fraction=_num(merged, "PNEUMOVIT_VAL_FRACTION", 0.15, float, 0.05, 0.4),
            test_fraction=_num(merged, "PNEUMOVIT_TEST_FRACTION", 0.2, float, 0.05, 0.4),
            target_sensitivity=_num(merged, "PNEUMOVIT_TARGET_SENSITIVITY", 0.95, float, 0.5, 0.999),
            n_bootstrap=_num(merged, "PNEUMOVIT_BOOTSTRAP", 1000, int, 50),
            k_neighbours=_num(merged, "PNEUMOVIT_K_NEIGHBOURS", 10, int, 1, 100),
            device=merged.get("PNEUMOVIT_DEVICE") or "auto",
            llm_provider=(merged.get("PNEUMOVIT_LLM_PROVIDER") or "none").lower(),
            llm_model=merged.get("PNEUMOVIT_LLM_MODEL") or "",
            llm_base_url=merged.get("PNEUMOVIT_LLM_BASE_URL") or "",
            llm_send_image=(merged.get("PNEUMOVIT_LLM_SEND_IMAGE") or "false").lower() in {"1", "true", "yes"},
            api_keys={k: merged[k] for k in ("GOOGLE_API_KEY", "OPENAI_API_KEY") if merged.get(k)},
        )
