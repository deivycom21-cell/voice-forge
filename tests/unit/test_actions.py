from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from voice_forge.actions.delegate import Delegator
from voice_forge.actions.executor import CommandRunner, FakeRunner, SubprocessRunner, tail
from voice_forge.core.types import ActionResult

PY = sys.executable


def test_subprocess_runner_success(tmp_path: Path) -> None:
    r = SubprocessRunner().run([PY, "-c", "import os; print('ok', os.getcwd())"], cwd=tmp_path)
    assert r.ok and r.returncode == 0
    assert r.stdout.startswith("ok") and str(tmp_path.name) in r.stdout
    assert r.duration_s > 0


def test_subprocess_runner_nonzero_and_stderr(tmp_path: Path) -> None:
    r = SubprocessRunner().run(
        [PY, "-c", "import sys; sys.stderr.write('bad'); sys.exit(3)"], cwd=tmp_path
    )
    assert not r.ok and r.returncode == 3 and r.stderr == "bad"


def test_subprocess_runner_timeout(tmp_path: Path) -> None:
    r = SubprocessRunner().run([PY, "-c", "import time; time.sleep(5)"], cwd=tmp_path, timeout=0.5)
    assert not r.ok and "tiempo agotado" in (r.error or "")


def test_subprocess_runner_stdin(tmp_path: Path) -> None:
    r = SubprocessRunner().run([PY, "-c", "import sys; print(sys.stdin.read().upper())"], cwd=tmp_path,
                               input="hola ñ")  # fmt: skip
    assert r.stdout == "HOLA Ñ"


def test_subprocess_runner_not_found_and_empty(tmp_path: Path) -> None:
    r = SubprocessRunner().run(["definitely-not-a-real-binary-xyz"], cwd=tmp_path)
    assert not r.ok and "no está instalado" in (r.error or "")
    assert SubprocessRunner().run([], cwd=tmp_path).error == "comando vacío"


def test_subprocess_runner_never_uses_shell(tmp_path: Path, monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs, argv=argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    SubprocessRunner(which=lambda p: "/usr/bin/" + p).run(
        ["git", "commit", "-m", "a; rm -rf /"], cwd=tmp_path
    )
    assert "shell" not in seen or seen["shell"] is False
    assert seen["argv"] == ["/usr/bin/git", "commit", "-m", "a; rm -rf /"]


def test_detached_launch_uses_popen(tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []
    runner = SubprocessRunner(
        which=lambda p: "/x/" + p, popen=lambda argv, **kw: calls.append((argv, kw))
    )
    r = runner.run(["cursor", "."], cwd=tmp_path, detached=True)
    assert r.ok and r.detached and r.returncode is None
    argv, kw = calls[0]
    assert argv == ["/x/cursor", "."] and kw["cwd"] == tmp_path
    assert kw["stdin"] == subprocess.DEVNULL
    assert ("creationflags" in kw) if sys.platform == "win32" else kw["start_new_session"]


def test_detached_launch_error(tmp_path: Path) -> None:
    def boom(argv: list[str], **kw: Any) -> None:
        raise OSError("denied")

    r = SubprocessRunner(which=lambda p: p, popen=boom).run(["x"], cwd=tmp_path, detached=True)
    assert r.error == "denied"


def test_tail() -> None:
    text = "\n".join(str(i) for i in range(100))
    out = tail(text, 5)
    assert out.splitlines()[-1] == "99" and "95 líneas omitidas" in out
    assert tail("a\nb", 5) == "a\nb"


def test_fake_runner(tmp_path: Path) -> None:
    canned = ActionResult(("git", "status"), returncode=1, stderr="nope")
    fr = FakeRunner({("git", "status"): canned})
    assert isinstance(fr, CommandRunner)
    assert fr.run(["git", "status"], cwd=tmp_path) is canned
    assert fr.run(["ls"], cwd=tmp_path).ok
    assert fr.argvs == [("git", "status"), ("ls",)]


# --------------------------------------------------------------------------- delegation


def test_delegate_to_clipboard(tmp_path: Path) -> None:
    copied: list[str] = []
    res = Delegator("clipboard", FakeRunner(), copy=copied.append).deliver("PROMPT", tmp_path)
    assert res.ok and copied == ["PROMPT"] and res.target == "clipboard"


def test_clipboard_failure_is_reported_not_raised(tmp_path: Path) -> None:
    def broken(_: str) -> None:
        raise RuntimeError("no xclip")

    res = Delegator("clipboard", FakeRunner(), copy=broken).deliver("P", tmp_path)
    assert not res.ok and "no xclip" in res.detail and res.output == "P"


def test_delegate_to_claude_uses_stdin_not_argv(tmp_path: Path) -> None:
    runner = FakeRunner(stdout="hecho")
    res = Delegator("clipboard", runner, claude_args=["--model", "sonnet"]).deliver(
        'rm -rf / "; & | prompt', tmp_path, target="claude"
    )
    call = runner.calls[0]
    assert call["argv"] == ("claude", "-p", "--model", "sonnet")
    assert call["input"] == 'rm -rf / "; & | prompt'
    assert call["cwd"] == tmp_path
    assert res.ok and res.output == "hecho"


def test_delegate_to_claude_json_and_errors(tmp_path: Path) -> None:
    runner = FakeRunner(stdout=json.dumps({"result": "listo"}))
    res = Delegator("claude", runner, claude_args=["--output-format", "json"]).deliver(
        "x", tmp_path
    )
    assert res.output == "listo"
    missing = FakeRunner(
        {("claude", "-p"): ActionResult(("claude", "-p"), error="'claude' no está instalado")}
    )
    assert not Delegator("claude", missing).deliver("x", tmp_path).ok
    failing = FakeRunner(
        {("claude", "-p"): ActionResult(("claude", "-p"), returncode=1, stderr="auth")}
    )
    res2 = Delegator("claude", failing).deliver("x", tmp_path)
    assert not res2.ok and res2.output == "auth"


def test_delegate_print(tmp_path: Path) -> None:
    res = Delegator("print", FakeRunner()).deliver("hola", tmp_path)
    assert res.ok and res.output == "hola"


# --------------------------------------------------------------------------- questions (ask)


def test_can_answer() -> None:
    assert Delegator("claude", FakeRunner()).can_answer
    assert Delegator("clipboard", FakeRunner(), answer_with_claude=True).can_answer
    assert not Delegator("clipboard", FakeRunner()).can_answer
    assert not Delegator("print", FakeRunner()).can_answer


def test_ask_sends_clean_question_via_stdin_and_keeps_full_output(tmp_path: Path) -> None:
    long_answer = "\n".join(f"línea {i}" for i in range(200))
    runner = FakeRunner(stdout=long_answer + "\n")
    res = Delegator("claude", runner, claude_args=["--model", "haiku"]).ask(
        "¿Qué es un VAD?", tmp_path
    )
    call = runner.calls[0]
    assert call["argv"] == ("claude", "-p", "--model", "haiku")
    assert call["input"] == "¿Qué es un VAD?" and call["full_output"] is True
    assert res.ok and res.output == long_answer
    assert "voice-forge last" in res.detail


def test_ask_errors(tmp_path: Path) -> None:
    empty = Delegator("claude", FakeRunner(stdout="   \n")).ask("q", tmp_path)
    assert not empty.ok and "ninguna respuesta" in empty.detail
    missing = FakeRunner({("claude", "-p"): ActionResult(("claude", "-p"), error="no instalado")})
    assert Delegator("claude", missing).ask("q", tmp_path).detail == "no instalado"
    json_out = FakeRunner(stdout=json.dumps({"result": "Respuesta."}))
    res = Delegator("claude", json_out, claude_args=["--output-format", "json"]).ask("q", tmp_path)
    assert res.output == "Respuesta."


def test_subprocess_runner_full_output_is_not_truncated(tmp_path: Path) -> None:
    code = "print('\\n'.join(str(i) for i in range(100)))"
    short = SubprocessRunner(max_output_lines=5).run([PY, "-c", code], cwd=tmp_path)
    assert "omitidas" in short.stdout
    full = SubprocessRunner(max_output_lines=5).run(
        [PY, "-c", code], cwd=tmp_path, full_output=True
    )
    assert full.stdout.splitlines() == [str(i) for i in range(100)]
