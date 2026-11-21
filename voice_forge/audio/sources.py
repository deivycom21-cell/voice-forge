"""Audio sources: microphone (sounddevice), in-memory arrays and WAV files.

All sources yield 16 kHz mono float32 blocks of ``block_ms`` milliseconds.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from voice_forge.audio.synth import SAMPLE_RATE, Audio
from voice_forge.audio.wav import load_for_stt

log = logging.getLogger(__name__)


@runtime_checkable
class AudioSource(Protocol):
    sample_rate: int

    def frames(self) -> Iterator[Audio]: ...

    def close(self) -> None: ...


class ArraySource:
    """Feeds a pre-recorded array block by block (tests, ``--from-wav``)."""

    def __init__(
        self,
        audio: Audio,
        *,
        block_ms: int = 30,
        sample_rate: int = SAMPLE_RATE,
        realtime: bool = False,
        tail_silence_ms: float = 0.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.block = max(1, round(sample_rate * block_ms / 1000))
        tail = np.zeros(round(sample_rate * tail_silence_ms / 1000), dtype=np.float32)
        self.audio = np.concatenate([audio.astype(np.float32), tail])
        self.realtime = realtime
        self._closed = False

    def frames(self) -> Iterator[Audio]:
        for start in range(0, len(self.audio), self.block):
            if self._closed:
                return
            chunk = self.audio[start : start + self.block]
            if len(chunk) < self.block:
                chunk = np.pad(chunk, (0, self.block - len(chunk)))
            if self.realtime:
                time.sleep(self.block / self.sample_rate)
            yield chunk

    def close(self) -> None:
        self._closed = True


# Synthetic signals are just arrays.
SyntheticSource = ArraySource


class WavFileSource(ArraySource):
    """Reads any PCM WAV, converts it to 16 kHz mono and feeds it in blocks."""

    def __init__(self, path: Path, *, block_ms: int = 30, tail_silence_ms: float = 1000.0) -> None:
        super().__init__(load_for_stt(path), block_ms=block_ms, tail_silence_ms=tail_silence_ms)
        self.path = path


class MicSource:
    """Live microphone capture through ``sounddevice`` (PortAudio).

    The PortAudio callback pushes blocks into a bounded queue; when the queue
    is full the oldest block is dropped and :attr:`overflows` is incremented.
    """

    def __init__(
        self,
        *,
        device: str | int | None = None,
        block_ms: int = 30,
        sample_rate: int = SAMPLE_RATE,
        max_queue: int = 200,
        sd_module: Any = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.block = round(sample_rate * block_ms / 1000)
        self.device = device if device not in ("", None) else None
        self._queue: queue.Queue[Audio] = queue.Queue(maxsize=max_queue)
        self._closed = threading.Event()
        self.overflows = 0
        self._sd = sd_module

    def _module(self) -> Any:
        if self._sd is None:
            import sounddevice

            self._sd = sounddevice
        return self._sd

    def _callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        if status:
            log.debug("audio status: %s", status)
        block = np.asarray(indata, dtype=np.float32).reshape(-1).copy()
        try:
            self._queue.put_nowait(block)
        except queue.Full:
            with contextlib.suppress(queue.Empty):  # race with the consumer
                self._queue.get_nowait()
            self.overflows += 1
            self._queue.put_nowait(block)

    def frames(self) -> Iterator[Audio]:
        sd = self._module()
        self._closed.clear()
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block,
            device=self.device,
            callback=self._callback,
        )
        stream.start()
        try:
            while not self._closed.is_set():
                try:
                    yield self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue
        finally:
            stream.stop()
            stream.close()
            while not self._queue.empty():
                self._queue.get_nowait()

    def close(self) -> None:
        self._closed.set()


def list_input_devices(sd_module: Any = None) -> list[dict[str, Any]]:
    """Return available input devices (empty list when PortAudio is missing)."""
    try:
        sd: Any = sd_module
        if sd is None:
            import sounddevice

            sd = sounddevice
        devices = sd.query_devices()
    except Exception:  # PortAudio missing, no devices...
        return []
    return [
        {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
        for i, d in enumerate(devices)
        if d.get("max_input_channels", 0) > 0
    ]
