from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from voice_forge import __version__, cli
from voice_forge.actions.executor import FakeRunner
from voice_forge.audio import synth
from voice_forge.audio.wav import write_wav
from voice_forge.stt.transcriber import FakeTranscriber

runner = CliRunner()


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "global_config_path", lambda: tmp_path / "cfg" / "config.toml")
    monkeypatch.setattr(
        "voice_forge.config.global_config_path", lambda: tmp_path / "cfg" / "config.toml"
    )
    return tmp_path


def invoke(*args: str, input: str | None = None) -> Any:
    return runner.invoke(cli.app, list(args), input=input)


def test_version_and_help() -> None:
    res = invoke("--version")
    assert res.exit_code == 0 and __version__ in res.output
    res = invoke("--help")
    assert res.exit_code == 0
    for cmd in ("listen", "models", "doctor", "config", "interpret"):
        assert cmd in res.output


def test_interpret_dry_run_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRunner()
    monkeypatch.setattr("voice_forge.app.SubprocessRunner", lambda: fake)
    res = invoke("interpret", "estado de git")
    assert res.exit_code == 0, res.output
    assert "git status --short --branch" in res.output and "dry-run" in res.output
    assert fake.calls == []


def test_interpret_execute(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRunner(stdout="## main")
    monkeypatch.setattr("voice_forge.app.SubprocessRunner", lambda: fake)
    res = invoke("interpret", "estado de git", "--execute")
    assert res.exit_code == 0 and "## main" in res.output
    assert fake.argvs == [("git", "status", "--short", "--branch")]


def test_interpret_execute_dangerous_asks(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRunner()
    monkeypatch.setattr("voice_forge.app.SubprocessRunner", lambda: fake)
    res = invoke("interpret", "haz push forzado", "-x", input="n\n")
    assert "requiere confirmación" in res.output and fake.calls == []
    res = invoke("interpret", "haz push forzado", "-x", input="y\n")
    assert fake.argvs == [("git", "push", "--force-with-lease")]


def test_interpret_delegate_print() -> None:
    res = invoke("interpret", "claude, explica el recorder", "--target", "print")
    assert res.exit_code == 0 and "Explica el recorder." in res.output


def test_config_init_show_set(tmp_path: Path) -> None:
    assert invoke("config", "path").output.strip().endswith("config.toml")
    assert invoke("config", "init").exit_code == 0
    assert (tmp_path / "cfg" / "config.toml").exists()
    assert invoke("config", "init").exit_code == 1  # exists
    assert invoke("config", "init", "--force").exit_code == 0
    res = invoke("config", "set", "stt.model", "tiny")
    assert res.exit_code == 0
    assert 'model = "tiny"' in invoke("config", "show").output
    bad = invoke("config", "set", "stt.language", "klingon")
    assert bad.exit_code == 2 and "stt.language" in bad.output


def test_invalid_config_exits_cleanly(tmp_path: Path) -> None:
    (tmp_path / "cfg").mkdir()
    (tmp_path / "cfg" / "config.toml").write_text("[stt]\nmodel = 5\n", encoding="utf-8")
    res = invoke("config", "show")
    assert res.exit_code == 2 and "inválida" in res.output


def test_models_list_download_remove(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from voice_forge.stt import models

    def fake_download(name: str) -> Path:
        p = tmp_path / "models" / name
        p.mkdir(parents=True)
        for f in models.REQUIRED_FILES:
            (p / f).write_text("x")
        return p

    monkeypatch.setattr(models, "models_dir", lambda: tmp_path / "models")
    monkeypatch.setattr(models, "download", fake_download)
    assert "tiny" in invoke("models", "list").output
    res = invoke("models", "download", "tiny")
    assert res.exit_code == 0 and "listo" in res.output
    assert "ya está descargado" in invoke("models", "download", "tiny").output
    assert invoke("models", "download", "huge").exit_code == 1
    assert "eliminado" in invoke("models", "remove", "tiny").output
    assert "no estaba" in invoke("models", "remove", "tiny").output
    assert invoke("models", "remove", "huge").exit_code == 1


def test_listen_without_model_gives_hint() -> None:
    res = invoke("listen")
    assert res.exit_code == 1 and "models download base" in res.output


def test_listen_rejects_bad_options() -> None:
    assert invoke("listen", "--lang", "fr").exit_code == 2
    assert invoke("listen", "--model", "gigantic").exit_code == 2


def test_listen_from_wav_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    if shutil.which("git"):
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    wav = write_wav(
        tmp_path / "orden.wav",
        synth.concat(synth.silence(400), synth.speechlike(900), synth.silence(900)),
    )
    fake = FakeRunner(stdout="3 passed")
    monkeypatch.setattr("voice_forge.stt.models.is_downloaded", lambda name, root=None: True)
    monkeypatch.setattr(
        "voice_forge.app.build_transcriber", lambda cfg: FakeTranscriber(["corre los tests"])
    )
    monkeypatch.setattr("voice_forge.app.SubprocessRunner", lambda: fake)
    res = invoke("listen", "--from-wav", str(wav), "--no-tts")
    assert res.exit_code == 0, res.output
    assert "corre los tests" in res.output and "3 passed" in res.output
    assert len(fake.argvs) == 1 and fake.argvs[0][-2:] == (
        "pytest",
        "-q",
    )  # "python -m pytest" if not on PATH
    missing = invoke("listen", "--from-wav", str(tmp_path / "nope.wav"), "--no-tts")
    assert missing.exit_code == 2


def test_doctor_runs_without_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    res = invoke("doctor", "--bench")
    assert res.exit_code in (0, 1)
    for needle in (
        "Python",
        "Micrófono",
        "Modelo",
        "Voz",
        "Atajo",
        "Herramientas",
        "Parser de reglas",
    ):
        assert needle in res.output, needle


def test_last_without_answers() -> None:
    res = invoke("last")
    assert res.exit_code == 1 and "no hay respuestas" in res.output


def test_question_is_answered_and_last_shows_it(monkeypatch: pytest.MonkeyPatch) -> None:
    answer = "Un **VAD** detecta cuándo hablas.\n\nSegunda parte de la respuesta."
    fake = FakeRunner(stdout=answer)
    monkeypatch.setattr("voice_forge.app.SubprocessRunner", lambda: fake)
    res = invoke("interpret", "Claude, ¿qué es un VAD?", "--target", "claude", "-x")
    assert res.exit_code == 0, res.output
    assert fake.calls[0]["input"] == "¿Qué es un VAD?"
    assert "Segunda parte de la respuesta." in res.output
    shown = invoke("last")
    assert shown.exit_code == 0
    assert "¿Qué es un VAD?" in shown.output and "Segunda parte de la respuesta." in shown.output
    raw = invoke("last", "--raw")
    assert raw.output.strip() == answer


def test_question_with_llm_claude_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRunner(stdout="Respuesta.")
    monkeypatch.setattr("voice_forge.app.SubprocessRunner", lambda: fake)
    monkeypatch.setattr("voice_forge.intent.llm.ClaudeCliBackend.available", lambda self: False)
    res = invoke("interpret", "explícame el recorder", "--llm", "claude", "-x")
    assert res.exit_code == 0, res.output
    assert fake.calls[0]["argv"] == ("claude", "-p")
    assert invoke("last", "--raw").output.strip() == "Respuesta."
