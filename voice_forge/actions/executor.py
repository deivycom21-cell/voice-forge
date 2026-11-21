"""Command execution. argv lists only — ``shell=True`` is never used."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from voice_forge.core.types import ActionResult


@runtime_checkable
class CommandRunner(Protocol):
    def run(
        self,
        argv: list[str] | tuple[str, ...],
        *,
        cwd: Path,
        timeout: float = 300.0,
        detached: bool = False,
        input: str | None = None,
        full_output: bool = False,
    ) -> ActionResult: ...


def child_env() -> dict[str, str]:
    """Environment for children: force UTF-8 I/O (Windows defaults to the ANSI codepage)."""
    return {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def tail(text: str, max_lines: int) -> str:
    lines = text.rstrip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join([f"… ({len(lines) - max_lines} líneas omitidas)", *lines[-max_lines:]])


class SubprocessRunner:
    def __init__(
        self,
        *,
        max_output_lines: int = 40,
        which: Callable[[str], str | None] = shutil.which,
        popen: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self.max_output_lines = max_output_lines
        self._which = which
        self._popen = popen

    def resolve(self, program: str) -> str | None:
        if os.path.isabs(program) and os.path.exists(program):
            return program
        return self._which(program)

    def run(
        self,
        argv: list[str] | tuple[str, ...],
        *,
        cwd: Path,
        timeout: float = 300.0,
        detached: bool = False,
        input: str | None = None,
        full_output: bool = False,
    ) -> ActionResult:
        argv_t = tuple(argv)
        if not argv_t:
            return ActionResult(argv_t, error="comando vacío")
        exe = self.resolve(argv_t[0])
        if exe is None:
            return ActionResult(
                argv_t, error=f"'{argv_t[0]}' no está instalado o no está en el PATH"
            )
        full = [exe, *argv_t[1:]]
        limit = 10**9 if full_output else self.max_output_lines  # answers must not be truncated
        t0 = time.perf_counter()
        if detached:
            kwargs: dict[str, Any] = {
                "cwd": cwd,
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            }
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(
                    subprocess, "DETACHED_PROCESS", 0
                )
            else:
                kwargs["start_new_session"] = True
            try:
                self._popen(full, **kwargs)
            except OSError as exc:
                return ActionResult(argv_t, error=str(exc))
            return ActionResult(argv_t, detached=True, duration_s=time.perf_counter() - t0)
        try:
            proc = subprocess.run(  # noqa: S603 - argv list, no shell
                full,
                cwd=cwd,
                input=input,
                stdin=None if input is not None else subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=child_env(),
            )
        except subprocess.TimeoutExpired as exc:
            out = (
                exc.stdout.decode("utf-8", "replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            return ActionResult(
                argv_t, stdout=tail(out, limit), duration_s=time.perf_counter() - t0,
                error=f"tiempo agotado ({timeout:.0f}s)",
            )  # fmt: skip
        except OSError as exc:
            return ActionResult(argv_t, error=str(exc), duration_s=time.perf_counter() - t0)
        return ActionResult(
            argv_t,
            returncode=proc.returncode,
            stdout=tail(proc.stdout or "", limit),
            stderr=tail(proc.stderr or "", limit),
            duration_s=time.perf_counter() - t0,
        )


class FakeRunner:
    """Records calls and returns canned results. Never runs anything."""

    def __init__(self, results: dict[tuple[str, ...], ActionResult] | None = None, *, returncode: int = 0,
                 stdout: str = "") -> None:  # fmt: skip
        self.results = results or {}
        self.returncode = returncode
        self.stdout = stdout
        self.calls: list[dict[str, Any]] = []

    @property
    def argvs(self) -> list[tuple[str, ...]]:
        return [c["argv"] for c in self.calls]

    def run(
        self,
        argv: list[str] | tuple[str, ...],
        *,
        cwd: Path,
        timeout: float = 300.0,
        detached: bool = False,
        input: str | None = None,
        full_output: bool = False,
    ) -> ActionResult:
        argv_t = tuple(argv)
        self.calls.append(
            {
                "argv": argv_t,
                "cwd": cwd,
                "timeout": timeout,
                "detached": detached,
                "input": input,
                "full_output": full_output,
            }
        )
        if argv_t in self.results:
            return self.results[argv_t]
        return ActionResult(argv_t, returncode=None if detached else self.returncode, stdout=self.stdout,
                            detached=detached)  # fmt: skip
