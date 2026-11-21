from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from voice_forge.audio import synth
from voice_forge.stt import models
from voice_forge.stt.transcriber import FakeTranscriber, Transcriber, WhisperTranscriber


def fake_snapshot_download(**kwargs: Any) -> str:
    target = Path(kwargs["local_dir"])
    for name in models.REQUIRED_FILES:
        (target / name).write_bytes(b"x" * 10)
    fake_snapshot_download.calls.append(kwargs)  # type: ignore[attr-defined]
    return str(target)


fake_snapshot_download.calls = []  # type: ignore[attr-defined]


def test_catalog_and_paths(tmp_path: Path) -> None:
    assert {"tiny", "base"} <= set(models.CATALOG)
    assert models.model_path("tiny", tmp_path) == tmp_path / "tiny"
    with pytest.raises(models.ModelError):
        models.model_path("gigantic", tmp_path)


def test_download_list_remove(tmp_path: Path) -> None:
    assert not models.is_downloaded("tiny", tmp_path)
    path = models.download("tiny", tmp_path, snapshot_download=fake_snapshot_download)
    call = fake_snapshot_download.calls[-1]  # type: ignore[attr-defined]
    assert call["repo_id"] == "Systran/faster-whisper-tiny"
    assert "model.bin" in call["allow_patterns"]
    assert path == tmp_path / "tiny" and models.is_downloaded("tiny", tmp_path)
    info = {m.name: m for m in models.list_models(tmp_path)}
    assert info["tiny"].downloaded and not info["base"].downloaded
    assert info["tiny"].size_mb > 0
    assert models.remove("tiny", tmp_path) is True
    assert models.remove("tiny", tmp_path) is False


def test_incomplete_download_raises(tmp_path: Path) -> None:
    with pytest.raises(models.ModelError, match="incomplete"):
        models.download("base", tmp_path, snapshot_download=lambda **kw: None)


@dataclass
class Seg:
    text: str


@dataclass
class Info:
    language: str


class FakeWhisperModel:
    instances: list[FakeWhisperModel] = []

    def __init__(self, path: str, **kwargs: Any) -> None:
        self.path = path
        self.kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        FakeWhisperModel.instances.append(self)

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Any, Info]:
        self.calls.append(kwargs)
        return iter([Seg(" Corre los "), Seg("tests. ")]), Info(kwargs.get("language") or "en")


def ready_models(tmp_path: Path, name: str = "base") -> Path:
    models.download(name, tmp_path, snapshot_download=fake_snapshot_download)
    return tmp_path


def test_whisper_transcriber_params(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    tr = WhisperTranscriber("base", vocabulary="git, pytest", models_root=ready_models(tmp_path),
                            model_factory=FakeWhisperModel, cpu_threads=2)  # fmt: skip
    assert not tr.loaded
    result = tr.transcribe(synth.tone(1000))
    model = FakeWhisperModel.instances[-1]
    assert model.path == str(tmp_path / "base")
    assert model.kwargs == {
        "device": "cpu",
        "compute_type": "int8",
        "cpu_threads": 2,
        "local_files_only": True,
    }
    call = model.calls[-1]
    assert call["beam_size"] == 1 and call["vad_filter"] is True
    assert call["initial_prompt"] == "git, pytest" and call["language"] == "es"
    assert result.text == "Corre los tests." and result.language == "es"
    assert result.duration_s == pytest.approx(1.0)
    import os

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    # lazy load happens once
    tr.transcribe(synth.tone(100))
    assert FakeWhisperModel.instances[-1] is model


def test_whisper_auto_language(tmp_path: Path) -> None:
    tr = WhisperTranscriber("base", language="auto", models_root=ready_models(tmp_path),
                            model_factory=FakeWhisperModel)  # fmt: skip
    tr.transcribe(synth.tone(100))
    assert FakeWhisperModel.instances[-1].calls[-1]["language"] is None
    tr.transcribe(synth.tone(100), language="en")
    assert FakeWhisperModel.instances[-1].calls[-1]["language"] == "en"


def test_whisper_missing_model_gives_hint(tmp_path: Path) -> None:
    tr = WhisperTranscriber("tiny", models_root=tmp_path, model_factory=FakeWhisperModel)
    with pytest.raises(models.ModelError, match="voice-forge models download tiny"):
        tr.transcribe(np.zeros(10, dtype=np.float32))


def test_fake_transcriber_script_and_callable() -> None:
    fake = FakeTranscriber(["uno", "dos"])
    assert isinstance(fake, Transcriber)
    assert [fake.transcribe(synth.tone(10)).text for _ in range(3)] == ["uno", "dos", ""]
    assert len(fake.calls) == 3
    fn = FakeTranscriber(lambda a: f"{len(a)} samples")
    assert fn.transcribe(synth.tone(10)).text == "160 samples"
    fn.load()
