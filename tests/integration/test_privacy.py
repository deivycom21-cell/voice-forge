"""Guard for principle P1: the audio -> text path never opens a network socket."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_audio import fake_sd
from tests.unit.test_stt import FakeWhisperModel, ready_models
from voice_forge.audio import synth
from voice_forge.audio.recorder import Recorder
from voice_forge.audio.sources import MicSource
from voice_forge.intent.rules import RuleParser
from voice_forge.stt import models
from voice_forge.stt.transcriber import WhisperTranscriber


class NetworkUsed(AssertionError):
    pass


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def deny(*args: Any, **kwargs: Any) -> None:
        raise NetworkUsed("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)


def test_audio_to_intent_path_has_no_network(no_network: None, tmp_path: Path) -> None:
    sig = synth.concat(synth.silence(400), synth.speechlike(1000), synth.silence(1000))
    rec = Recorder().record(MicSource(sd_module=fake_sd(sig)))
    assert rec.has_audio and rec.audio is not None
    tr = WhisperTranscriber(
        "base", models_root=ready_models(tmp_path), model_factory=FakeWhisperModel
    )
    text = tr.transcribe(rec.audio).text
    intent = RuleParser().parse(text)
    assert intent.argv == ("pytest", "-q")


def test_guard_actually_blocks(no_network: None) -> None:
    with pytest.raises(NetworkUsed):
        socket.create_connection(("example.com", 80))


def test_runtime_never_downloads(tmp_path: Path) -> None:
    """Loading a model that is not on disk fails with a hint instead of downloading it."""
    tr = WhisperTranscriber("tiny", models_root=tmp_path / "empty")
    with pytest.raises(models.ModelError, match="models download"):
        tr.load()
