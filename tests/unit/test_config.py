from __future__ import annotations

import sys
from pathlib import Path

import pytest

from voice_forge.config import (
    DEFAULT_ALIASES,
    PROJECT_FILE,
    AliasSpec,
    Config,
    ConfigError,
    config_from_dict,
    deep_merge,
    find_project_file,
    load_config,
    parse_cli_value,
    set_value,
)

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


def test_defaults_are_safe() -> None:
    cfg = Config()
    assert cfg.stt.model == "base" and cfg.stt.compute_type == "int8"
    assert cfg.llm.backend == "none"  # offline by default
    assert cfg.delegate.target == "clipboard"
    assert cfg.llm.allow_remote_ollama is False
    assert cfg.aliases == DEFAULT_ALIASES


def test_merge_precedence_project_over_global(tmp_path: Path) -> None:
    gpath = tmp_path / "global.toml"
    gpath.write_text('[stt]\nmodel = "tiny"\nlanguage = "en"\n', encoding="utf-8")
    project = tmp_path / "proj"
    (project / "src").mkdir(parents=True)
    (project / ".git").mkdir()
    (project / PROJECT_FILE).write_text(
        '[stt]\nlanguage = "es"\n[aliases]\n"despliega" = { cmd = "make deploy", confirm = true }\n',
        encoding="utf-8",
    )
    cfg = load_config(project / "src", gpath)
    assert cfg.stt.model == "tiny"  # from global
    assert cfg.stt.language == "es"  # project wins
    assert cfg.aliases["despliega"] == AliasSpec("make deploy", confirm=True)
    assert "abre el dashboard" in cfg.aliases  # defaults kept
    assert cfg.project_file == project / PROJECT_FILE
    assert cfg.global_file == gpath


def test_project_search_stops_at_git_root(tmp_path: Path) -> None:
    (tmp_path / PROJECT_FILE).write_text("", encoding="utf-8")
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    assert find_project_file(repo) is None
    assert find_project_file(tmp_path) == tmp_path / PROJECT_FILE


@pytest.mark.parametrize(
    ("data", "path"),
    [
        ({"stt": {"model": 3}}, "stt.model"),
        ({"stt": {"language": "fr"}}, "stt.language"),
        ({"audio": {"silence_ms": -1}}, "audio.silence_ms"),
        ({"audio": {"block_ms": 500}}, "audio.block_ms"),
        ({"audio": {"nope": 1}}, "audio.nope"),
        ({"unknown_section": {}}, "unknown_section"),
        ({"tts": {"enabled": "yes"}}, "tts.enabled"),
        ({"intent": {"fuzzy_threshold": 1.5}}, "intent.fuzzy_threshold"),
        ({"safety": {"extra_confirm_patterns": ["("]}}, "safety.extra_confirm_patterns[0]"),
        ({"safety": {"extra_blocked_patterns": "x"}}, "safety.extra_blocked_patterns"),
        ({"aliases": {"x": 3}}, "aliases.'x'"),
        ({"aliases": {"x": {"confirm": True}}}, "aliases.'x'"),
        ({"aliases": {"x": "  "}}, "aliases.'x'"),
        ({"stt": "base"}, "stt"),
    ],
)
def test_invalid_values_report_path(data: dict, path: str) -> None:
    with pytest.raises(ConfigError) as exc:
        config_from_dict(data)
    assert exc.value.path == path


def test_int_accepted_for_float_and_bool_rejected_for_int() -> None:
    cfg = config_from_dict({"audio": {"max_seconds": 10}})
    assert cfg.audio.max_seconds == 10.0 and isinstance(cfg.audio.max_seconds, float)
    with pytest.raises(ConfigError):
        config_from_dict({"audio": {"block_ms": True}})


def test_toml_roundtrip() -> None:
    cfg = config_from_dict({"aliases": {"build": {"cmd": "make", "confirm": True}, "x": "echo hi"}})
    again = config_from_dict(tomllib.loads(cfg.to_toml()))
    assert again.to_dict() == cfg.to_dict()
    assert again.aliases["build"].confirm is True


def test_invalid_toml_file(tmp_path: Path) -> None:
    g = tmp_path / "g.toml"
    g.write_text("[stt\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(tmp_path, g, use_project=False)


def test_set_value_validates_before_writing(tmp_path: Path) -> None:
    g = tmp_path / "cfg" / "config.toml"
    cfg = set_value(g, "stt.model", "tiny")
    assert cfg.stt.model == "tiny"
    assert set_value(g, "audio.silence_ms", "900").audio.silence_ms == 900
    before = g.read_text(encoding="utf-8")
    with pytest.raises(ConfigError):
        set_value(g, "stt.language", "klingon")
    with pytest.raises(ConfigError):
        set_value(g, "model", "x")
    assert g.read_text(encoding="utf-8") == before


def test_parse_cli_value_and_merge() -> None:
    assert parse_cli_value("true") is True
    assert parse_cli_value("12") == 12
    assert parse_cli_value('["a"]') == ["a"]
    assert parse_cli_value("hola mundo") == "hola mundo"
    assert deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}, "b": 1}) == {
        "a": {"x": 1, "y": 3},
        "b": 1,
    }
