"""Recording state machine: wait for speech, capture it, stop on silence.

States: LISTENING (waiting for speech, filling the pre-roll) -> SPEECH -> DONE.

Stop reasons:
  * ``silence``      trailing silence >= ``silence_ms``
  * ``max_duration`` segment reached ``max_seconds``
  * ``stopped``      the caller set ``stop`` (hotkey released / pressed again)
  * ``cancelled``    the caller set ``cancel`` (audio discarded)
  * ``no_speech``    nothing heard before ``start_timeout_s``
  * ``source_end``   the audio source ran out
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from voice_forge.audio.sources import AudioSource
from voice_forge.audio.synth import SAMPLE_RATE, Audio
from voice_forge.audio.vad import EnergyVAD

StopReason = Literal["silence", "max_duration", "stopped", "cancelled", "no_speech", "source_end"]

TRAILING_KEEP_MS = 200.0


@dataclass(frozen=True)
class RecordingResult:
    audio: Audio | None
    reason: StopReason
    elapsed_ms: float = 0.0

    @property
    def has_audio(self) -> bool:
        return self.audio is not None and len(self.audio) > 0

    @property
    def duration_s(self) -> float:
        return 0.0 if self.audio is None else len(self.audio) / SAMPLE_RATE


class Recorder:
    def __init__(
        self,
        vad_factory: Callable[[], EnergyVAD] | None = None,
        *,
        sample_rate: int = SAMPLE_RATE,
        block_ms: int = 30,
        preroll_ms: float = 300.0,
        silence_ms: float = 700.0,
        min_speech_ms: float = 150.0,
        max_seconds: float = 30.0,
        start_timeout_s: float = 8.0,
        on_level: Callable[[float], None] | None = None,
    ) -> None:
        self.vad_factory = vad_factory or EnergyVAD
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.preroll_ms = preroll_ms
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_seconds = max_seconds
        self.start_timeout_s = start_timeout_s
        self.on_level = on_level

    def record(
        self,
        source: AudioSource,
        *,
        stop: threading.Event | None = None,
        cancel: threading.Event | None = None,
        start_timeout_s: float | None = None,
    ) -> RecordingResult:
        vad = self.vad_factory()
        timeout_ms = 1000.0 * (
            start_timeout_s if start_timeout_s is not None else self.start_timeout_s
        )
        preroll_frames = max(1, math.ceil((self.preroll_ms + self.min_speech_ms) / self.block_ms))
        preroll: deque[Audio] = deque(maxlen=preroll_frames)
        segment: list[Audio] = []
        frame_ms_list: list[float] = []
        started = False
        heard_any = False
        speech_run = 0.0
        silence_run = 0.0
        elapsed = 0.0
        seg_ms = 0.0

        def finish(reason: StopReason) -> RecordingResult:
            if not started:
                if reason == "stopped" and heard_any and preroll:
                    return RecordingResult(np.concatenate(list(preroll)), reason, elapsed)
                return RecordingResult(
                    None, "no_speech" if reason != "cancelled" else reason, elapsed
                )
            frames = segment
            if reason == "silence":
                # keep at most TRAILING_KEEP_MS of the trailing silence
                drop_ms = max(0.0, silence_run - TRAILING_KEEP_MS)
                while frames and drop_ms >= frame_ms_list[-1] - 1e-6:
                    drop_ms -= frame_ms_list.pop()
                    frames = frames[:-1]
            return RecordingResult(np.concatenate(frames).astype(np.float32), reason, elapsed)

        frames_iter = iter(source.frames())
        try:
            for frame in frames_iter:
                if cancel is not None and cancel.is_set():
                    return RecordingResult(None, "cancelled", elapsed)
                frame_ms = 1000.0 * len(frame) / self.sample_rate
                elapsed += frame_ms
                speech = vad.is_speech(frame)
                if self.on_level is not None:
                    self.on_level(vad.last_level_db)

                if not started:
                    preroll.append(frame)
                    speech_run = speech_run + frame_ms if speech else 0.0
                    heard_any = heard_any or speech
                    if speech_run >= self.min_speech_ms:
                        started = True
                        segment = list(preroll)
                        frame_ms_list = [1000.0 * len(f) / self.sample_rate for f in segment]
                        seg_ms = sum(frame_ms_list)
                    elif elapsed >= timeout_ms:
                        return RecordingResult(None, "no_speech", elapsed)
                else:
                    segment.append(frame)
                    frame_ms_list.append(frame_ms)
                    seg_ms += frame_ms
                    silence_run = 0.0 if speech else silence_run + frame_ms
                    if silence_run >= self.silence_ms:
                        return finish("silence")
                    if seg_ms >= self.max_seconds * 1000.0:
                        return finish("max_duration")

                if stop is not None and stop.is_set():
                    return finish("stopped")
            return finish("source_end")
        finally:
            close = getattr(frames_iter, "close", None)
            if callable(close):
                close()
