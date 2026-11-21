"""Shared fixtures. No test touches the real user config, microphone, model or network."""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_forge.core.types import ContextSnapshot


@pytest.fixture(autouse=True)
def _isolate_user_dirs(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path_factory.mktemp("userdirs")
    monkeypatch.setattr("voice_forge.config.global_config_path", lambda: home / "config.toml")
    monkeypatch.setattr("voice_forge.stt.models.models_dir", lambda: home / "models")
    monkeypatch.setattr(
        "voice_forge.actions.answers.last_answer_path", lambda: home / "last_answer.json"
    )


@pytest.fixture
def py_ctx(tmp_path: Path) -> ContextSnapshot:
    return ContextSnapshot(
        cwd=tmp_path,
        git_root=tmp_path,
        branch="main",
        upstream="origin/main",
        changed_files=("voice_forge/cli.py", "README.md"),
        markers=("pyproject.toml",),
        tools=frozenset({"git", "pytest", "ruff", "claude"}),
    )
