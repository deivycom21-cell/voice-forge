"""Configuration: dataclasses with safe defaults, loaded from TOML.

Two layers are merged (later wins):

1. Global file: ``platformdirs.user_config_dir("voice-forge")/config.toml``
2. Project file: ``.voice-forge.toml`` found walking up from the cwd.
"""

from __future__ import annotations

import sys
import types
import typing
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

import tomli_w

APP_NAME = "voice-forge"
PROJECT_FILE = ".voice-forge.toml"

DEFAULT_VOCABULARY = (
    "git, commit, push, pull, merge, rebase, stash, branch, rama, diff, log, pytest, tests, "
    "Claude, Claude Code, Cursor, swarm-pilot, aether, ollama, refactor, dashboard, deploy, "
    "feature, fix, hotfix, main, develop, release"
)


class ConfigError(ValueError):
    """Invalid configuration value (``path`` points at the offending key)."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path


@dataclass
class AudioConfig:
    device: str = ""  # empty = system default input
    block_ms: int = 30
    abs_threshold_db: float = -50.0
    margin_db: float = 12.0
    max_noise_floor_db: float = -40.0
    calibration_ms: int = 300
    preroll_ms: int = 300
    silence_ms: int = 700
    min_speech_ms: int = 150
    max_seconds: float = 30.0
    start_timeout_s: float = 8.0


@dataclass
class STTConfig:
    model: str = "base"
    language: Literal["es", "en", "auto"] = "es"
    compute_type: str = "int8"
    cpu_threads: int = 4
    beam_size: int = 1
    vocabulary: str = DEFAULT_VOCABULARY


@dataclass
class HotkeyConfig:
    combo: str = "ctrl+alt+v"
    mode: Literal["toggle", "hold"] = "toggle"
    backend: Literal["auto", "pynput", "terminal"] = "auto"


@dataclass
class IntentConfig:
    fuzzy_threshold: float = 0.82
    delegate_min_words: int = 12
    ui_language: Literal["es", "en"] = "es"


@dataclass
class LLMConfig:
    backend: Literal["none", "ollama", "claude"] = "none"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:1.5b"
    claude_model: str = ""
    timeout_s: float = 20.0
    allow_remote_ollama: bool = False


@dataclass
class DelegateConfig:
    target: Literal["clipboard", "claude", "print"] = "clipboard"
    claude_args: list[str] = field(default_factory=list)
    claude_timeout_s: float = 600.0


@dataclass
class TTSConfig:
    enabled: bool = True
    engine: Literal["auto", "piper", "pyttsx3", "espeak", "none"] = "auto"
    rate: int = 185
    voice: str = ""
    piper_model: str = ""


@dataclass
class SafetyConfig:
    confirm_timeout_s: float = 10.0
    command_timeout_s: float = 300.0
    extra_confirm_patterns: list[str] = field(default_factory=list)
    extra_blocked_patterns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AliasSpec:
    """A user defined voice phrase mapped to a command."""

    cmd: str
    confirm: bool = False
    detached: bool = False
    shell: bool = False


DEFAULT_ALIASES: dict[str, AliasSpec] = {
    "abre el dashboard": AliasSpec("swarm-pilot dashboard", detached=True),
    "open the dashboard": AliasSpec("swarm-pilot dashboard", detached=True),
}


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    intent: IntentConfig = field(default_factory=IntentConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    delegate: DelegateConfig = field(default_factory=DelegateConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    aliases: dict[str, AliasSpec] = field(default_factory=lambda: dict(DEFAULT_ALIASES))
    # provenance (not serialised)
    global_file: Path | None = None
    project_file: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for f in fields(self):
            if f.name in ("global_file", "project_file"):
                continue
            value = getattr(self, f.name)
            if f.name == "aliases":
                data["aliases"] = {k: _alias_to_toml(v) for k, v in value.items()}
            else:
                data[f.name] = asdict(value)
        return data

    def to_toml(self) -> str:
        return tomli_w.dumps(self.to_dict())


def _alias_to_toml(spec: AliasSpec) -> str | dict[str, Any]:
    if spec == AliasSpec(spec.cmd):
        return spec.cmd
    return {k: v for k, v in asdict(spec).items() if v or k == "cmd"}


# --------------------------------------------------------------------------- validation


def _check_type(value: Any, tp: Any, path: str) -> Any:
    origin = get_origin(tp)
    if origin is Literal:
        choices = get_args(tp)
        if value not in choices:
            raise ConfigError(
                path, f"must be one of {', '.join(map(repr, choices))}, got {value!r}"
            )
        return value
    if origin in (typing.Union, types.UnionType):
        for arg in get_args(tp):
            try:
                return _check_type(value, arg, path)
            except ConfigError:
                continue
        raise ConfigError(path, f"invalid value {value!r}")
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(path, f"must be a list, got {type(value).__name__}")
        (item_tp,) = get_args(tp)
        return [_check_type(v, item_tp, f"{path}[{i}]") for i, v in enumerate(value)]
    if tp is type(None):
        if value is not None:
            raise ConfigError(path, "must be null")
        return None
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(path, f"must be true/false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(path, f"must be an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(path, f"must be a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(path, f"must be a string, got {value!r}")
        return value
    return value  # pragma: no cover - unsupported annotations are accepted as-is


def _build_section(cls: type[Any], data: Any, path: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(path, "must be a table")
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{path}.{sorted(unknown)[0]}", "unknown key")
    kwargs = {k: _check_type(v, hints[k], f"{path}.{k}") for k, v in data.items()}
    obj = cls(**kwargs)
    _validate_ranges(obj, path)
    return obj


def _validate_ranges(obj: Any, path: str) -> None:
    positive = {
        "block_ms", "calibration_ms", "silence_ms", "max_seconds", "start_timeout_s",
        "cpu_threads", "beam_size", "delegate_min_words", "timeout_s", "confirm_timeout_s",
        "command_timeout_s", "rate", "claude_timeout_s",
    }  # fmt: skip
    for f in fields(obj):
        value = getattr(obj, f.name)
        if f.name in positive and isinstance(value, (int, float)) and value <= 0:
            raise ConfigError(f"{path}.{f.name}", "must be > 0")
    if isinstance(obj, IntentConfig) and not 0.0 < obj.fuzzy_threshold <= 1.0:
        raise ConfigError(f"{path}.fuzzy_threshold", "must be in (0, 1]")
    if isinstance(obj, AudioConfig) and obj.block_ms > 200:
        raise ConfigError(f"{path}.block_ms", "must be <= 200")
    if isinstance(obj, (SafetyConfig,)):
        import re

        for name in ("extra_confirm_patterns", "extra_blocked_patterns"):
            for i, pat in enumerate(getattr(obj, name)):
                try:
                    re.compile(pat)
                except re.error as exc:
                    raise ConfigError(f"{path}.{name}[{i}]", f"invalid regex: {exc}") from exc


def _build_aliases(data: Any, path: str = "aliases") -> dict[str, AliasSpec]:
    if not isinstance(data, dict):
        raise ConfigError(path, "must be a table")
    out: dict[str, AliasSpec] = {}
    for phrase, spec in data.items():
        p = f"{path}.{phrase!r}"
        if isinstance(spec, str):
            alias = AliasSpec(spec)
        elif isinstance(spec, dict):
            if "cmd" not in spec:
                raise ConfigError(p, "missing 'cmd'")
            alias = _build_section(AliasSpec, spec, p)
        else:
            raise ConfigError(p, "must be a string or a table with 'cmd'")
        if not alias.cmd.strip():
            raise ConfigError(p, "empty command")
        out[phrase] = alias
    return out


def config_from_dict(data: dict[str, Any]) -> Config:
    """Build a validated :class:`Config` from a (merged) raw dict."""
    cfg = Config()
    unknown = set(data) - {f.name for f in fields(Config)} - {"global_file", "project_file"}
    if unknown:
        raise ConfigError(sorted(unknown)[0], "unknown section")
    updates: dict[str, Any] = {}
    for f in fields(Config):
        if f.name not in data or f.name in ("global_file", "project_file"):
            continue
        if f.name == "aliases":
            merged = dict(DEFAULT_ALIASES)
            merged.update(_build_aliases(data["aliases"]))
            updates["aliases"] = merged
        else:
            section_cls = type(getattr(cfg, f.name))
            updates[f.name] = _build_section(section_cls, data[f.name], f.name)
    return replace(cfg, **updates)


# --------------------------------------------------------------------------- files


def global_config_path() -> Path:
    from platformdirs import user_config_dir

    return Path(user_config_dir(APP_NAME, appauthor=False)) / "config.toml"


def find_project_file(start: Path) -> Path | None:
    """Walk up from ``start`` until a project file or a ``.git`` root is found."""
    current = start.resolve()
    for directory in (current, *current.parents):
        candidate = directory / PROJECT_FILE
        if candidate.is_file():
            return candidate
        if (directory / ".git").exists():
            return None
    return None


def read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            data: dict[str, Any] = tomllib.load(fh)
            return data
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(str(path), f"invalid TOML: {exc}") from exc


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    cwd: Path | None = None,
    global_path: Path | None = None,
    *,
    use_project: bool = True,
) -> Config:
    """Load and merge the global and project configuration files."""
    gpath = global_path if global_path is not None else global_config_path()
    raw: dict[str, Any] = {}
    gfile = None
    if gpath.is_file():
        raw = read_toml(gpath)
        gfile = gpath
    pfile = find_project_file(cwd or Path.cwd()) if use_project else None
    if pfile is not None:
        raw = deep_merge(raw, read_toml(pfile))
    cfg = config_from_dict(raw)
    return replace(cfg, global_file=gfile, project_file=pfile)


def parse_cli_value(value: str) -> Any:
    """Parse ``value`` as a TOML literal, falling back to a plain string."""
    try:
        return tomllib.loads(f"v = {value}")["v"]
    except tomllib.TOMLDecodeError:
        return value


def set_value(path: Path, dotted_key: str, value: str) -> Config:
    """Set ``section.key`` in the TOML file at ``path`` (validated before writing)."""
    raw = read_toml(path) if path.is_file() else {}
    parts = dotted_key.split(".")
    if len(parts) < 2:
        raise ConfigError(dotted_key, "use section.key, e.g. stt.model")
    node = raw
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(dotted_key, "not a table")
    node[parts[-1]] = parse_cli_value(value)
    cfg = config_from_dict(raw)  # validate before touching disk
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(raw), encoding="utf-8")
    return cfg
