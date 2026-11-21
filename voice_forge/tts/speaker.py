"""Offline spoken feedback.

Engine priority with ``engine = "auto"``: Piper (if installed and a voice model
is configured) -> pyttsx3 (SAPI5 on Windows, espeak-ng on Linux) -> espeak-ng
CLI -> silent. Speech runs on a worker thread so callers never block, and an
engine error permanently degrades to silence instead of crashing.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)

PHRASES: dict[str, dict[str, str]] = {
    "es": {
        "ready": "Listo para escuchar",
        "done": "Listo",
        "failed": "Falló",
        "confirm": "¿Confirmas?",
        "cancelled": "Cancelado",
        "blocked": "Bloqueado por seguridad",
        "unknown": "No te entendí",
        "nothing": "No escuché nada",
        "copied": "Prompt copiado",
        "sent": "Enviado a Claude",
        "asking": "Preguntando a Claude",
        "no_answer": "Claude no respondió",
        "running": "Ejecutando",
        "tests.run": "Corriendo tests",
        "tests.failed": "Repitiendo tests fallidos",
        "tests.target": "Corriendo tests",
        "git.push": "Subiendo cambios",
        "git.pull": "Trayendo cambios",
        "git.commit": "Haciendo commit",
        "open.claude": "Abriendo Claude",
        "open.editor": "Abriendo editor",
        "muted": "Silencio",
        "unmuted": "Voz activada",
        "bye": "Hasta luego",
        "error": "Hubo un error",
    },
    "en": {
        "ready": "Ready",
        "done": "Done",
        "failed": "Failed",
        "confirm": "Confirm?",
        "cancelled": "Cancelled",
        "blocked": "Blocked for safety",
        "unknown": "I didn't get that",
        "nothing": "I didn't hear anything",
        "copied": "Prompt copied",
        "sent": "Sent to Claude",
        "asking": "Asking Claude",
        "no_answer": "Claude didn't answer",
        "running": "Running",
        "tests.run": "Running tests",
        "tests.failed": "Rerunning failed tests",
        "tests.target": "Running tests",
        "git.push": "Pushing",
        "git.pull": "Pulling",
        "git.commit": "Committing",
        "open.claude": "Opening Claude",
        "open.editor": "Opening editor",
        "muted": "Muted",
        "unmuted": "Voice on",
        "bye": "Goodbye",
        "error": "Something went wrong",
    },
}


def phrase(key: str, lang: str = "es", default_key: str | None = None) -> str:
    table = PHRASES.get(lang, PHRASES["es"])
    if key in table:
        return table[key]
    if default_key is not None:
        return table.get(default_key, "")
    return key


@runtime_checkable
class Speaker(Protocol):
    muted: bool
    name: str

    def say(self, text: str) -> None: ...

    def close(self) -> None: ...


class NullSpeaker:
    name = "none"

    def __init__(self) -> None:
        self.muted = False

    def say(self, text: str) -> None:
        pass

    def close(self) -> None:
        pass


class RecordingSpeaker:
    """Collects what would have been said (tests)."""

    name = "recording"

    def __init__(self) -> None:
        self.muted = False
        self.said: list[str] = []

    def say(self, text: str) -> None:
        if not self.muted and text:
            self.said.append(text)

    def close(self) -> None:
        pass


SpeakFn = Callable[[str], None]


class AsyncSpeaker:
    """Runs a blocking engine on a daemon worker thread with a small queue.

    ``engine_factory`` is called *inside* the worker thread (COM/SAPI5 needs
    to be initialised on the thread that uses it).
    """

    def __init__(
        self, name: str, engine_factory: Callable[[], SpeakFn], *, max_queue: int = 4
    ) -> None:
        self.name = name
        self.muted = False
        self.failed: BaseException | None = None
        self._factory = engine_factory
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=max_queue)
        self._busy = 0
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._worker, name=f"vf-tts-{name}", daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        try:
            speak = self._factory()
        except Exception as exc:
            log.warning("TTS engine %s unavailable: %s", self.name, exc)
            self.failed = exc
            self._done(clear=True)
            return
        while True:
            text = self._queue.get()
            if text is None:
                return
            try:
                speak(text)
            except Exception as exc:
                log.warning("TTS engine %s failed, going silent: %s", self.name, exc)
                self.failed = exc
                self._done(clear=True)
                return
            self._done()

    def _done(self, clear: bool = False) -> None:
        with self._cond:
            self._busy = 0 if clear else max(0, self._busy - 1)
            self._cond.notify_all()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until everything queued has been spoken (or ``timeout``)."""
        with self._cond:
            return self._cond.wait_for(lambda: self._busy == 0 or self.failed is not None, timeout)

    def say(self, text: str) -> None:
        if self.muted or self.failed is not None or not text:
            return
        with self._cond:
            try:
                self._queue.put_nowait(text)
                self._busy += 1
            except queue.Full:  # drop rather than lag behind
                pass

    def close(self) -> None:
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)


# --------------------------------------------------------------------------- engines


def _pyttsx3_factory(rate: int, voice: str, lang: str) -> Callable[[], SpeakFn]:
    def factory() -> SpeakFn:
        if sys.platform == "win32":
            try:
                import pythoncom

                pythoncom.CoInitialize()
            except Exception:  # pragma: no cover - pywin32 missing
                pass
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        chosen = pick_voice(engine.getProperty("voices") or [], voice, lang)
        if chosen:
            engine.setProperty("voice", chosen)

        def speak(text: str) -> None:
            engine.say(text)
            engine.runAndWait()

        return speak

    return factory


def pick_voice(voices: list[Any], preferred: str, lang: str) -> str | None:
    """Pick a voice id: explicit match on id/name first, then by language."""
    if preferred:
        for v in voices:
            if (
                preferred.lower()
                in (str(getattr(v, "id", "")) + str(getattr(v, "name", ""))).lower()
            ):
                return str(v.id)
    keys = {"es": ("spanish", "español", "espanol", "es-", "es_"),
            "en": ("english", "en-", "en_")}.get(lang, (lang,))  # fmt: skip
    for v in voices:
        langs = " ".join(
            x.decode(errors="ignore").strip(chr(5)) if isinstance(x, bytes) else str(x)
            for x in (getattr(v, "languages", None) or [])
        ).lower()
        hay = f"{getattr(v, 'name', '')} {getattr(v, 'id', '')} {langs}".lower()
        if langs.startswith(lang) or any(k in hay for k in keys):
            return str(v.id)
    return None


def _espeak_factory(rate: int, lang: str, exe: str) -> Callable[[], SpeakFn]:
    def factory() -> SpeakFn:
        def speak(text: str) -> None:
            subprocess.run([exe, "-v", lang, "-s", str(rate), text], check=False,  # noqa: S603
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)  # fmt: skip

        return speak

    return factory


def _piper_factory(model_path: str) -> Callable[[], SpeakFn]:
    def factory() -> SpeakFn:
        import io
        import wave

        import numpy as np
        import sounddevice as sd
        from piper import PiperVoice

        voice = PiperVoice.load(model_path)

        def speak(text: str) -> None:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                if hasattr(voice, "synthesize_wav"):
                    voice.synthesize_wav(text, wf)
                else:  # piper-tts < 1.3
                    voice.synthesize(text, wf)
            buf.seek(0)
            with wave.open(buf, "rb") as wf:
                sr = wf.getframerate()
                data = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
            sd.play(data.astype(np.float32) / 32768.0, sr)
            sd.wait()

        return speak

    return factory


def available_engines(which: Callable[[str], str | None] = shutil.which) -> list[str]:
    engines = []
    try:
        import piper  # noqa: F401

        engines.append("piper")
    except Exception:
        pass
    try:
        import pyttsx3  # noqa: F401

        engines.append("pyttsx3")
    except Exception:
        pass
    if which("espeak-ng") or which("espeak"):
        engines.append("espeak")
    return engines


def build_speaker(
    *,
    enabled: bool = True,
    engine: str = "auto",
    lang: str = "es",
    rate: int = 185,
    voice: str = "",
    piper_model: str = "",
    which: Callable[[str], str | None] = shutil.which,
    engines: list[str] | None = None,
) -> Speaker:
    if not enabled or engine == "none":
        return NullSpeaker()
    have = engines if engines is not None else available_engines(which)
    order = ["piper", "pyttsx3", "espeak"] if engine == "auto" else [engine]
    for name in order:
        if name not in have:
            continue
        if name == "piper":
            if piper_model and Path(piper_model).is_file():
                return AsyncSpeaker("piper", _piper_factory(piper_model))
            continue
        if name == "pyttsx3":
            return AsyncSpeaker("pyttsx3", _pyttsx3_factory(rate, voice, lang))
        if name == "espeak":
            exe = which("espeak-ng") or which("espeak")
            if exe:
                return AsyncSpeaker("espeak", _espeak_factory(rate, lang, exe))
    return NullSpeaker()
