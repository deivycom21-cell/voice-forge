"""Deterministic synthetic signals (16 kHz mono float32).

Used by the test-suite and by ``voice-forge doctor`` so that every audio code
path can be exercised without a microphone. Same seed -> same samples on any OS.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

SAMPLE_RATE = 16_000
Audio = npt.NDArray[np.float32]


def _n(ms: float, sr: int) -> int:
    return round(sr * ms / 1000.0)


def silence(ms: float, sr: int = SAMPLE_RATE) -> Audio:
    return np.zeros(_n(ms, sr), dtype=np.float32)


def tone(ms: float, freq: float = 220.0, amp: float = 0.3, sr: int = SAMPLE_RATE) -> Audio:
    t = np.arange(_n(ms, sr), dtype=np.float64) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def noise(ms: float, amp: float = 0.01, seed: int = 0, sr: int = SAMPLE_RATE) -> Audio:
    """Gaussian noise whose RMS is ``amp``."""
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(0.0, amp, _n(ms, sr)), -1.0, 1.0).astype(np.float32)


def speechlike(ms: float, amp: float = 0.3, seed: int = 0, sr: int = SAMPLE_RATE) -> Audio:
    """Harmonic signal with a 4 Hz syllabic envelope (roughly voice-shaped)."""
    rng = np.random.default_rng(seed)
    n = _n(ms, sr)
    t = np.arange(n, dtype=np.float64) / sr
    f0 = 140.0 + 20.0 * rng.random()
    sig = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t) ** 2
    out = sig * env
    peak = float(np.max(np.abs(out))) or 1.0
    return (amp * out / peak).astype(np.float32)


def concat(*parts: Audio) -> Audio:
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


def mix(a: Audio, b: Audio) -> Audio:
    """Sum two signals, padding the shorter one with zeros."""
    n = max(len(a), len(b))
    out = np.zeros(n, dtype=np.float32)
    out[: len(a)] += a
    out[: len(b)] += b
    return np.clip(out, -1.0, 1.0)


def db_to_amp(db: float) -> float:
    return float(10 ** (db / 20.0))
