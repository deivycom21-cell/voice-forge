from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.helpers import FakeClock
from voice_forge.context.probe import (
    ContextProbe,
    parse_branch_header,
    parse_porcelain_z,
    recent_files,
)

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def init_repo(path: Path) -> Path:
    git(path, "init", "-q")
    git(path, "symbolic-ref", "HEAD", "refs/heads/main")
    git(path, "config", "user.email", "test@example.invalid")
    git(path, "config", "user.name", "Test")
    git(path, "config", "commit.gpgsign", "false")
    return path


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("## main", ("main", None, 0, 0, False)),
        ("## main...origin/main", ("main", "origin/main", 0, 0, False)),
        ("## feat/x...origin/feat/x [ahead 2, behind 1]", ("feat/x", "origin/feat/x", 2, 1, False)),
        ("## main...origin/main [behind 3]", ("main", "origin/main", 0, 3, False)),
        ("## No commits yet on main", ("main", None, 0, 0, False)),
        ("## HEAD (no branch)", (None, None, 0, 0, True)),
    ],
)
def test_parse_branch_header(header: str, expected: tuple) -> None:
    assert parse_branch_header(header) == expected


def test_parse_porcelain_with_rename() -> None:
    out = "## main\0 M a.py\0R  new.py\0old.py\0?? b.txt\0"
    header, paths = parse_porcelain_z(out)
    assert header == "## main"
    assert paths == ["a.py", "new.py", "b.txt"]


def test_recent_files_ignores_heavy_dirs_and_orders_by_mtime(tmp_path: Path) -> None:
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("x")
    (tmp_path / "src").mkdir()
    for i, name in enumerate(["a.py", "src/b.py", "c.md"]):
        p = tmp_path / name
        p.write_text(name)
        os.utime(p, (1_000_000 + i, 1_000_000 + i))
    assert recent_files(tmp_path, limit=2) == ["c.md", "src/b.py"]
    assert "node_modules/x.js" not in recent_files(tmp_path)


@needs_git
def test_snapshot_real_repo(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n")
    (repo / "a.py").write_text("print(1)\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    (repo / "a.py").write_text("print(2)\n")
    (repo / "new.txt").write_text("hi")
    (repo / "sub").mkdir()
    snap = ContextProbe(tools=("git", "pytest")).snapshot(repo / "sub")
    assert snap.git_root == repo.resolve()
    assert snap.branch == "main" and snap.upstream is None
    assert set(snap.changed_files) == {"a.py", "new.txt"}
    assert "pyproject.toml" in snap.markers and snap.project_kind == "python"
    assert "git" in snap.tools
    assert snap.cwd == (repo / "sub").resolve()


@needs_git
def test_snapshot_repo_without_commits(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    snap = ContextProbe(tools=("git",)).snapshot(repo)
    assert snap.branch == "main" and snap.is_git


def test_snapshot_outside_git(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text("{}")
    probe = ContextProbe(
        tools=("git", "npm"),
        runner=lambda argv, cwd, t: subprocess.CompletedProcess(argv, 128, "", "fatal"),
    )
    snap = probe.snapshot(tmp_path)
    assert snap.git_root is None and snap.branch is None
    assert snap.project_kind == "node" and snap.root == tmp_path.resolve()


def test_git_timeout_degrades_gracefully(tmp_path: Path) -> None:
    probe = ContextProbe(which=lambda t: "/bin/" + t, runner=lambda argv, cwd, t: None)
    snap = probe.snapshot(tmp_path)
    assert snap.git_root is None and snap.changed_files == ()


def test_tools_detected_via_which(tmp_path: Path) -> None:
    probe = ContextProbe(
        tools=("git", "claude", "aether"), which=lambda t: "/x" if t == "claude" else None
    )
    assert probe.snapshot(tmp_path).tools == frozenset({"claude"})


def test_cache_respects_ttl(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(argv: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if argv[1] == "rev-parse":
            return subprocess.CompletedProcess(argv, 0, str(tmp_path) + "\n", "")
        return subprocess.CompletedProcess(argv, 0, "## dev\0 M x.py\0", "")

    clock = FakeClock()
    probe = ContextProbe(
        tools=("git",), which=lambda t: "/x", runner=runner, clock=clock, ttl_s=2.0
    )
    s1 = probe.snapshot(tmp_path)
    assert s1.branch == "dev" and s1.changed_files == ("x.py",)
    probe.snapshot(tmp_path)
    assert len(calls) == 2  # cached
    clock.advance(2.5)
    probe.snapshot(tmp_path)
    assert len(calls) == 4
    probe.invalidate()
    probe.snapshot(tmp_path)
    assert len(calls) == 6
