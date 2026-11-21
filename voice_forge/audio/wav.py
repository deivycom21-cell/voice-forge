"""Minimal WAV I/O and resampling using only the stdlib + numpy."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from voice_forge.audio.synth import SAMPLE_RATE, Audio


def write_wav(path: Path, audio: Audio, sr: int = SAMPLE_RATE) -> Path:
    """Write mono float32 [-1, 1] audio as 16-bit PCM."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


def read_wav(path: Path) -> tuple[Audio, int]:
    """Read a PCM WAV (8/16/32-bit, any channels) as mono float32."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {width * 8} bits")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32), sr


def resample(audio: Audio, src_sr: int, dst_sr: int = SAMPLE_RATE) -> Audio:
    """Linear-interpolation resampling (good enough for speech recognition)."""
    if src_sr == dst_sr or len(audio) == 0:
        return audio.astype(np.float32)
    duration = len(audio) / src_sr
    n_out = max(1, round(duration * dst_sr))
    x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
    out: Audio = np.asarray(np.interp(x_new, x_old, audio), dtype=np.float32)
    return out


def load_for_stt(path: Path) -> Audio:
    """Read any PCM WAV and return 16 kHz mono float32."""
    audio, sr = read_wav(path)
    return resample(audio, sr, SAMPLE_RATE)
