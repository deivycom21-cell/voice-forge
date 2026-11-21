"""Core immutable data types shared across voice-forge."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Literal

IntentSource = Literal["rules", "llm", "alias"]


class IntentKind(str, Enum):
    """What the user wants to happen."""

    EXECUTE = "execute"
    DELEGATE = "delegate"
    CONFIRM = "confirm"
    CANCEL = "cancel"
    META = "meta"
    UNKNOWN = "unknown"


class Risk(IntEnum):
    """Risk level assigned by the safety shield (ordered: higher is riskier)."""

    SAFE = 0
    CONFIRM = 1
    BLOCKED = 2


class Status(str, Enum):
    """High level state of the assistant, shown in the HUD."""

    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    CONFIRMING = "confirming"
    EXECUTING = "executing"


@dataclass(frozen=True)
class Intent:
    """A resolved user intention.

    ``argv`` is only meaningful for ``EXECUTE``; ``prompt`` for ``DELEGATE``;
    ``meta`` names the meta action (``repeat``, ``history``, ``mute``...).
    """

    kind: IntentKind
    utterance: str = ""
    argv: tuple[str, ...] = ()
    prompt: str | None = None
    meta: str | None = None
    source: IntentSource = "rules"
    confidence: float = 1.0
    label: str = ""
    name: str = ""
    detached: bool = False
    confirm: bool = False
    shell: bool = False

    @property
    def display(self) -> str:
        """Short human readable description."""
        if self.label:
            return self.label
        if self.argv:
            return " ".join(self.argv)
        if self.prompt:
            return self.prompt[:80]
        return self.kind.value


@dataclass(frozen=True)
class Transcript:
    """Result of speech-to-text."""

    text: str
    language: str | None = None
    duration_s: float = 0.0
    elapsed_s: float = 0.0


@dataclass(frozen=True)
class ContextSnapshot:
    """What voice-forge knows about the current working directory."""

    cwd: Path
    git_root: Path | None = None
    branch: str | None = None
    upstream: str | None = None
    ahead: int = 0
    behind: int = 0
    detached_head: bool = False
    changed_files: tuple[str, ...] = ()
    recent_files: tuple[str, ...] = ()
    markers: tuple[str, ...] = ()
    tools: frozenset[str] = field(default_factory=frozenset)

    @property
    def root(self) -> Path:
        """Project root: the git root if any, else the cwd."""
        return self.git_root or self.cwd

    @property
    def is_git(self) -> bool:
        return self.git_root is not None

    def has_tool(self, name: str) -> bool:
        return name in self.tools

    @property
    def project_kind(self) -> str:
        """Best guess of the project ecosystem."""
        m = set(self.markers)
        if m & {
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "requirements.txt",
            "pytest.ini",
            "tox.ini",
        }:
            return "python"
        if "package.json" in m:
            return "node"
        if "Cargo.toml" in m:
            return "rust"
        if "go.mod" in m:
            return "go"
        return "unknown"


@dataclass(frozen=True)
class ActionResult:
    """Outcome of running a command."""

    argv: tuple[str, ...]
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    error: str | None = None
    detached: bool = False

    @property
    def ok(self) -> bool:
        if self.error is not None:
            return False
        return self.detached or self.returncode == 0
