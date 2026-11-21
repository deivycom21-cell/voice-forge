from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from voice_forge.config import DEFAULT_ALIASES, AliasSpec
from voice_forge.core.types import ContextSnapshot, IntentKind
from voice_forge.intent.normalize import fold, normalize, recover_original, to_int
from voice_forge.intent.rules import (
    RuleParser,
    branch_slug,
    build_test_argv,
    help_lines,
    in_new_terminal,
)

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

CASES: list[dict[str, Any]] = tomllib.loads(
    (Path(__file__).parents[1] / "fixtures" / "utterances.toml").read_text(encoding="utf-8")
)["case"]


# --------------------------------------------------------------------------- normalize


def test_fold_is_length_preserving() -> None:
    for s in ["Árbol ÑANDÚ", "Ça va? ¿Qué tal?", "İstanbul", "ﬁx"]:
        assert len(fold(s)) == len(s)
    assert fold("Análisis ÑU") == "analisis nu"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("¡Eh, corre los tests, por favor!", "corre los tests"),
        ("Bueno, este, estado de git porfa", "estado de git"),
        ("  GUITAR   status. ", "git status"),
        ("ok", "ok"),
        ("py test", "pytest"),
        ("snake_case nombre", "snake case nombre"),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected
    assert normalize(normalize(raw)) == normalize(raw)  # idempotent


def test_recover_original_and_to_int() -> None:
    assert (
        recover_original("Haz commit: Arreglé el ANÁLISIS!", "arregle el analisis")
        == "Arreglé el ANÁLISIS"
    )
    assert recover_original("abc", "") == ""
    assert recover_original("abc", "zzz") == "zzz"
    assert (
        to_int("cinco", 10) == 5
        and to_int("7", 10) == 7
        and to_int("muchos", 10) == 10
        and to_int(None, 3) == 3
    )


# --------------------------------------------------------------------------- grammar table


@pytest.fixture(scope="module")
def parser() -> RuleParser:
    return RuleParser(DEFAULT_ALIASES)


@pytest.mark.parametrize("case", CASES, ids=[c["text"] or "<empty>" for c in CASES])
def test_utterance_table(case: dict[str, Any], parser: RuleParser, py_ctx: ContextSnapshot) -> None:
    intent = parser.parse(case["text"], py_ctx)
    assert intent.kind.value == case["kind"], intent
    if "argv" in case:
        assert list(intent.argv) == case["argv"]
    if "prompt" in case:
        assert intent.prompt == case["prompt"]
    if "meta" in case:
        assert intent.meta == case["meta"]


def test_table_is_big_enough() -> None:
    assert len(CASES) >= 75


@pytest.mark.parametrize(
    ("text", "argv"),
    [
        ("estao de git", ("git", "status", "--short", "--branch")),
        ("corre los tez", ("pytest", "-q")),
        ("abre el dashbord", ("swarm-pilot", "dashboard")),
    ],
)
def test_fuzzy_matching(
    text: str, argv: tuple[str, ...], parser: RuleParser, py_ctx: ContextSnapshot
) -> None:
    intent = parser.parse(text, py_ctx)
    assert intent.argv == argv
    assert 0.82 <= intent.confidence < 1.0


def test_fuzzy_never_produces_confirmation(parser: RuleParser) -> None:
    assert (
        parser.parse("confirmó").kind != IntentKind.CONFIRM or normalize("confirmó") == "confirmo"
    )
    assert parser.parse("confirmar todo").kind != IntentKind.CONFIRM


def test_context_changes_commands(tmp_path: Path) -> None:
    p = RuleParser()
    node = ContextSnapshot(cwd=tmp_path, markers=("package.json",))
    assert p.parse("corre los tests", node).argv == ("npm", "test")
    assert p.parse("corre los tests del login", node).argv == ("npm", "test", "--", "login")
    assert p.parse("corre el linter", node).argv == ("npm", "run", "lint")
    rust = ContextSnapshot(cwd=tmp_path, markers=("Cargo.toml",))
    assert p.parse("run the tests", rust).argv == ("cargo", "test")
    go = ContextSnapshot(cwd=tmp_path, markers=("go.mod",))
    assert p.parse("run the tests", go).argv == ("go", "test", "./...")
    no_pytest = ContextSnapshot(cwd=tmp_path, markers=("pyproject.toml",))
    assert p.parse("corre los tests", no_pytest).argv == ("python", "-m", "pytest", "-q")
    assert p.parse("corre el linter", no_pytest).argv == ("python", "-m", "ruff", "check", ".")
    assert p.parse("corre los tests").argv == ("pytest", "-q")  # no context


def test_push_sets_upstream_for_new_branch(py_ctx: ContextSnapshot) -> None:
    p = RuleParser()
    new_branch = replace(py_ctx, branch="feature/x", upstream=None)
    assert p.parse("sube los cambios", new_branch).argv == (
        "git",
        "push",
        "-u",
        "origin",
        "feature/x",
    )
    assert p.parse("sube los cambios", py_ctx).argv == ("git", "push")


def test_user_aliases(tmp_path: Path) -> None:
    p = RuleParser({
        "despliega a staging": AliasSpec("make deploy ENV=staging", confirm=True),
        "Limpia Todo": AliasSpec("docker compose down && docker system prune", shell=True),
    })  # fmt: skip
    a = p.parse("Despliega a staging, por favor")
    assert a.argv == ("make", "deploy", "ENV=staging") and a.confirm and a.source == "alias"
    b = p.parse("limpia todo")
    assert b.shell and b.argv[-1] == "docker compose down && docker system prune"
    assert b.argv[0] in ("cmd", "sh")


@pytest.mark.parametrize(
    ("spoken", "slug"),
    [
        ("feature barra login con google", "feature/login-con-google"),
        ("main", "main"),
        ("fix guion bajo parser", "fix_parser"),
        ("release punto 2", "release.2"),
        ("Ñandú raro!!", "nandu-raro"),
        ("", "nueva-rama"),
    ],
)
def test_branch_slug(spoken: str, slug: str) -> None:
    assert branch_slug(normalize(spoken) if spoken else spoken) == slug


def test_in_new_terminal_per_platform() -> None:
    assert in_new_terminal(("claude",), "win32", lambda x: None) == (
        "cmd",
        "/c",
        "start",
        "",
        "claude",
    )
    assert in_new_terminal(("claude",), "win32", lambda x: "wt.exe")[:3] == ("wt", "-d", ".")
    assert in_new_terminal(("claude",), "linux", lambda x: "/usr/bin/" + x if x == "konsole" else None) == (
        "konsole", "-e", "claude")  # fmt: skip
    assert in_new_terminal(("claude",), "linux", lambda x: None) is None


def test_open_claude_rule(parser: RuleParser) -> None:
    intent = parser.parse("abre claude")
    assert intent.kind in (IntentKind.EXECUTE, IntentKind.META)
    if intent.kind == IntentKind.EXECUTE:
        assert intent.argv[-1] == "claude" and intent.detached


def test_test_argv_target_python(py_ctx: ContextSnapshot) -> None:
    assert build_test_argv(py_ctx, failed=True) == ("pytest", "-q", "--lf")
    assert help_lines("en") and help_lines("es")


def test_latency_budget(parser: RuleParser, py_ctx: ContextSnapshot) -> None:
    phrases = [c["text"] for c in CASES] + ["frase totalmente desconocida x y"]
    parser.parse("warm up", py_ctx)
    worst = 0.0
    t_total = time.perf_counter()
    for _ in range(10):
        for text in phrases:
            t0 = time.perf_counter()
            parser.parse(text, py_ctx)
            worst = max(worst, time.perf_counter() - t0)
    n = 10 * len(phrases)
    avg_ms = (time.perf_counter() - t_total) * 1000 / n
    assert worst < 0.100, f"worst case {worst * 1000:.1f} ms"
    assert avg_ms < 5.0, f"average {avg_ms:.2f} ms"
