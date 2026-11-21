"""Silent, fast, never-failing snapshot of the current project.

Every git call has a short timeout; any failure degrades to empty fields.
Results are cached for ``ttl_s`` seconds per directory.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from voice_forge.core.types import ContextSnapshot

DEFAULT_TOOLS = (
    "git", "claude", "swarm-pilot", "aether", "pytest", "npm", "cargo", "go",
    "cursor", "code", "ollama", "ruff",
)  # fmt: skip
MARKERS = (
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "pytest.ini", "tox.ini",
    "package.json", "Cargo.toml", "go.mod", "Makefile", "pom.xml", "Gemfile", "composer.json",
    ".voice-forge.toml", "CLAUDE.md",
)  # fmt: skip
IGNORED_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "target", ".tox", ".idea", ".vscode", ".next",
}  # fmt: skip

Runner = Callable[[list[str], Path, float], "subprocess.CompletedProcess[str] | None"]


def _run(argv: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def parse_branch_header(header: str) -> tuple[str | None, str | None, int, int, bool]:
    """Parse ``## main...origin/main [ahead 1, behind 2]`` -> (branch, upstream, ahead, behind, detached)."""
    h = header[3:] if header.startswith("## ") else header
    ahead = behind = 0
    if "[" in h and h.endswith("]"):
        h, _, info = h.partition(" [")
        for part in info.rstrip("]").split(","):
            part = part.strip()
            if part.startswith("ahead "):
                ahead = int(part.split()[1])
            elif part.startswith("behind "):
                behind = int(part.split()[1])
    for prefix in ("No commits yet on ", "Initial commit on "):
        if h.startswith(prefix):
            return h[len(prefix) :], None, 0, 0, False
    if h.startswith("HEAD (no branch)"):
        return None, None, 0, 0, True
    branch, sep, upstream = h.partition("...")
    return branch or None, (upstream or None) if sep else None, ahead, behind, False


def parse_porcelain_z(output: str) -> tuple[str, list[str]]:
    """Parse ``git status --porcelain=v1 --branch -z``. Returns (header, paths)."""
    entries = output.split("\0")
    header = ""
    paths: list[str] = []
    i = 0
    while i < len(entries):
        entry = entries[i]
        i += 1
        if not entry:
            continue
        if entry.startswith("## "):
            header = entry
            continue
        status, path = entry[:2], entry[3:]
        paths.append(path)
        if status[0] in "RC":  # rename/copy: next entry is the original path
            i += 1
    return header, paths


def recent_files(
    root: Path, limit: int = 10, max_scan: int = 3000, max_depth: int = 5
) -> list[str]:
    """Most recently modified files under ``root`` (relative, forward slashes)."""
    found: list[tuple[float, str]] = []
    scanned = 0
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack and scanned < max_scan:
        directory, depth = stack.pop()
        try:
            with os.scandir(directory) as it:
                for entry in it:
                    if scanned >= max_scan:
                        break
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name not in IGNORED_DIRS and depth < max_depth:
                                stack.append((Path(entry.path), depth + 1))
                        elif entry.is_file(follow_symlinks=False):
                            scanned += 1
                            rel = Path(entry.path).relative_to(root).as_posix()
                            found.append((entry.stat().st_mtime, rel))
                    except OSError:
                        continue
        except OSError:
            continue
    found.sort(key=lambda t: (-t[0], t[1]))
    return [rel for _, rel in found[:limit]]


class ContextProbe:
    def __init__(
        self,
        *,
        tools: Iterable[str] = DEFAULT_TOOLS,
        which: Callable[[str], str | None] = shutil.which,
        runner: Runner = _run,
        clock: Callable[[], float] = time.monotonic,
        ttl_s: float = 2.0,
        git_timeout_s: float = 1.0,
        recent_limit: int = 10,
    ) -> None:
        self.tool_names = tuple(tools)
        self._which = which
        self._run = runner
        self._clock = clock
        self.ttl_s = ttl_s
        self.git_timeout_s = git_timeout_s
        self.recent_limit = recent_limit
        self._cache: dict[Path, tuple[float, ContextSnapshot]] = {}
        self._tools_cache: frozenset[str] | None = None

    def tools(self) -> frozenset[str]:
        if self._tools_cache is None:
            self._tools_cache = frozenset(t for t in self.tool_names if self._which(t))
        return self._tools_cache

    def invalidate(self) -> None:
        self._cache.clear()

    def snapshot(self, cwd: Path | None = None) -> ContextSnapshot:
        cwd = (cwd or Path.cwd()).resolve()
        now = self._clock()
        cached = self._cache.get(cwd)
        if cached is not None and now - cached[0] < self.ttl_s:
            return cached[1]
        snap = self._build(cwd)
        self._cache[cwd] = (now, snap)
        return snap

    def _build(self, cwd: Path) -> ContextSnapshot:
        tools = self.tools()
        git_root: Path | None = None
        branch = upstream = None
        ahead = behind = 0
        detached = False
        changed: list[str] = []
        if "git" in tools:
            top = self._run(["git", "rev-parse", "--show-toplevel"], cwd, self.git_timeout_s)
            if top is not None and top.returncode == 0 and top.stdout.strip():
                git_root = Path(top.stdout.strip()).resolve()
                st = self._run(
                    ["git", "status", "--porcelain=v1", "--branch", "-z"], cwd, self.git_timeout_s
                )
                if st is not None and st.returncode == 0:
                    header, changed = parse_porcelain_z(st.stdout)
                    branch, upstream, ahead, behind, detached = parse_branch_header(header)
        root = git_root or cwd
        markers = tuple(m for m in MARKERS if (root / m).exists())
        try:
            recent = tuple(recent_files(root, self.recent_limit))
        except OSError:
            recent = ()
        return ContextSnapshot(
            cwd=cwd,
            git_root=git_root,
            branch=branch,
            upstream=upstream,
            ahead=ahead,
            behind=behind,
            detached_head=detached,
            changed_files=tuple(changed),
            recent_files=recent,
            markers=markers,
            tools=tools,
        )
