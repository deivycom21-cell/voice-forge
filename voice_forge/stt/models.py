"""Whisper model catalog, download and local cache.

``download`` is the ONLY networked operation in voice-forge.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voice_forge.config import APP_NAME

CATALOG: dict[str, tuple[str, int]] = {
    # name: (huggingface repo, approx. download size in MB)
    "tiny": ("Systran/faster-whisper-tiny", 75),
    "tiny.en": ("Systran/faster-whisper-tiny.en", 75),
    "base": ("Systran/faster-whisper-base", 145),
    "base.en": ("Systran/faster-whisper-base.en", 145),
    "small": ("Systran/faster-whisper-small", 485),
}
REQUIRED_FILES = ("model.bin", "config.json", "tokenizer.json")
ALLOW_PATTERNS = [
    "config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
    "preprocessor_config.json",
]


class ModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelInfo:
    name: str
    repo: str
    approx_mb: int
    path: Path
    downloaded: bool
    size_mb: float


def models_dir() -> Path:
    from platformdirs import user_cache_dir

    return Path(user_cache_dir(APP_NAME, appauthor=False)) / "models"


def _check_name(name: str) -> tuple[str, int]:
    if name not in CATALOG:
        raise ModelError(f"unknown model {name!r}; choose one of: {', '.join(CATALOG)}")
    return CATALOG[name]


def model_path(name: str, root: Path | None = None) -> Path:
    _check_name(name)
    return (root or models_dir()) / name


def is_downloaded(name: str, root: Path | None = None) -> bool:
    path = model_path(name, root)
    return all((path / f).is_file() for f in REQUIRED_FILES)


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def list_models(root: Path | None = None) -> list[ModelInfo]:
    out = []
    for name, (repo, mb) in CATALOG.items():
        path = model_path(name, root)
        out.append(ModelInfo(name, repo, mb, path, is_downloaded(name, root), _dir_size_mb(path)))
    return out


def download(
    name: str,
    root: Path | None = None,
    *,
    snapshot_download: Callable[..., Any] | None = None,
) -> Path:
    """Download ``name`` into the local cache (network!). Returns the model dir."""
    repo, _ = _check_name(name)
    target = model_path(name, root)
    target.mkdir(parents=True, exist_ok=True)
    if snapshot_download is None:
        from huggingface_hub import snapshot_download as _sd

        snapshot_download = _sd
    snapshot_download(repo_id=repo, local_dir=str(target), allow_patterns=ALLOW_PATTERNS)
    if not is_downloaded(name, root):
        raise ModelError(f"download of {name!r} incomplete: missing files in {target}")
    return target


def remove(name: str, root: Path | None = None) -> bool:
    path = model_path(name, root)
    if path.exists():
        shutil.rmtree(path)
        return True
    return False
