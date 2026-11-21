"""Optional end-to-end test with the REAL Whisper model and a system TTS voice.

Skipped by default (marker ``model``). Run with::

    voice-forge models download base
    pytest -m model
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from voice_forge.audio.recorder import Recorder
from voice_forge.audio.sources import WavFileSource
from voice_forge.intent.rules import RuleParser
from voice_forge.stt import models
from voice_forge.stt.transcriber import WhisperTranscriber

pytestmark = pytest.mark.model

CASES = [
    ("Corre los tests", ("pytest", "-q")),
    ("Estado de git", ("git", "status", "--short", "--branch")),
    ("Haz push forzado", ("git", "push", "--force-with-lease")),
]


def synthesize(text: str, path: Path) -> bool:
    if sys.platform == "win32":
        try:
            import pyttsx3
        except ImportError:
            return False
        engine = pyttsx3.init()
        for voice in engine.getProperty("voices"):
            if "ES-" in voice.id.upper():
                engine.setProperty("voice", voice.id)
                break
        else:
            return False
        engine.setProperty("rate", 170)
        engine.save_to_file(text, str(path))
        engine.runAndWait()
        return path.exists()
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if exe is None:
        return False
    subprocess.run([exe, "-v", "es", "-w", str(path), text], check=True)
    return True


@pytest.fixture(scope="module")
def transcriber() -> WhisperTranscriber:
    from platformdirs import user_cache_dir

    root = Path(user_cache_dir("voice-forge", appauthor=False)) / "models"
    if not models.is_downloaded("base", root):
        pytest.skip("run `voice-forge models download base` first")
    tr = WhisperTranscriber("base", models_root=root, vocabulary="git, push, pytest, tests, commit")
    tr.load()
    return tr


@pytest.mark.parametrize(("text", "argv"), CASES)
def test_spoken_command_end_to_end(
    text: str, argv: tuple[str, ...], transcriber: WhisperTranscriber, tmp_path: Path
) -> None:
    wav = tmp_path / "cmd.wav"
    if not synthesize(text, wav):
        pytest.skip("no Spanish TTS voice available to synthesize speech")
    rec = Recorder().record(WavFileSource(wav))
    assert rec.has_audio and rec.audio is not None
    heard = transcriber.transcribe(rec.audio, language="es").text
    intent = RuleParser().parse(heard)
    assert intent.argv[-len(argv) :] == argv, f"heard {heard!r}"
