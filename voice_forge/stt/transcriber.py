"""Speech-to-text backends.

:class:`WhisperTranscriber` runs faster-whisper fully offline on the CPU.
:class:`FakeTranscriber` returns scripted text (tests, ``--dry-run`` demos).
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from voice_forge.audio.synth import SAMPLE_RATE, Audio
from voice_forge.core.types import Transcript
from voice_forge.stt import models


@runtime_checkable
class Transcriber(Protocol):
    def transcribe(self, audio: Audio, *, language: str | None = None) -> Transcript: ...


class WhisperTranscriber:
    """faster-whisper on CPU, int8, loaded lazily and thread-safely."""

    def __init__(
        self,
        model_name: str = "base",
        *,
        compute_type: str = "int8",
        cpu_threads: int = 4,
        beam_size: int = 1,
        vocabulary: str = "",
        language: str | None = "es",
        models_root: Path | None = None,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.beam_size = beam_size
        self.vocabulary = vocabulary
        self.language = None if language in (None, "", "auto") else language
        self.models_root = models_root
        self._factory = model_factory
        self._model: Any = None
        self._lock = threading.Lock()

    def load(self) -> Any:
        with self._lock:
            if self._model is None:
                # Never touch the network at runtime.
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                if not models.is_downloaded(self.model_name, self.models_root):
                    raise models.ModelError(
                        f"model {self.model_name!r} is not downloaded. "
                        f"Run: voice-forge models download {self.model_name}"
                    )
                factory = self._factory
                if factory is None:
                    from faster_whisper import WhisperModel

                    factory = WhisperModel
                self._model = factory(
                    str(models.model_path(self.model_name, self.models_root)),
                    device="cpu",
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads,
                    local_files_only=True,
                )
            return self._model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: Audio, *, language: str | None = None) -> Transcript:
        model = self.load()
        lang = language if language not in (None, "", "auto") else self.language
        t0 = time.perf_counter()
        segments, info = model.transcribe(
            audio,
            language=lang,
            beam_size=self.beam_size,
            vad_filter=True,
            initial_prompt=self.vocabulary or None,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return Transcript(
            text=text,
            language=getattr(info, "language", lang),
            duration_s=len(audio) / SAMPLE_RATE,
            elapsed_s=time.perf_counter() - t0,
        )


class FakeTranscriber:
    """Returns scripted transcripts in order (or via a callable)."""

    def __init__(self, script: Iterable[str] | Callable[[Audio], str] = ()) -> None:
        self._fn: Callable[[Audio], str] | None = script if callable(script) else None
        self._script = [] if callable(script) else list(script)
        self.calls: list[Audio] = []

    def load(self) -> None:
        """No-op (API parity with :class:`WhisperTranscriber`)."""

    def transcribe(self, audio: Audio, *, language: str | None = None) -> Transcript:
        self.calls.append(audio)
        if self._fn is not None:
            text = self._fn(audio)
        else:
            text = self._script.pop(0) if self._script else ""
        return Transcript(text=text, language=language, duration_s=len(audio) / SAMPLE_RATE)
