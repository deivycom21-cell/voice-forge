"""Deliver a structured prompt to an agent: clipboard, ``claude -p`` or stdout.

The prompt is passed to ``claude`` through **stdin**, never as a command-line
argument, so arbitrary spoken text can not be interpreted by a shell or by
Windows' ``.cmd`` argument parsing.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from voice_forge.actions.executor import CommandRunner

Target = Literal["clipboard", "claude", "print"]


@dataclass(frozen=True)
class DelegationResult:
    target: str
    ok: bool
    detail: str = ""
    output: str = ""


def _default_copy(text: str) -> None:
    import pyperclip

    pyperclip.copy(text)


class Delegator:
    def __init__(
        self,
        target: Target,
        runner: CommandRunner,
        *,
        claude_args: Sequence[str] = (),
        claude_timeout_s: float = 600.0,
        copy: Callable[[str], None] | None = None,
    ) -> None:
        self.target = target
        self.runner = runner
        self.claude_args = tuple(claude_args)
        self.claude_timeout_s = claude_timeout_s
        self._copy = copy or _default_copy

    def deliver(self, prompt: str, cwd: Path, target: Target | None = None) -> DelegationResult:
        chosen = target or self.target
        if chosen == "clipboard":
            try:
                self._copy(prompt)
            except Exception as exc:  # pyperclip raises its own exception types
                return DelegationResult(
                    "clipboard", False, f"portapapeles no disponible: {exc}", prompt
                )
            return DelegationResult("clipboard", True, "prompt copiado al portapapeles", prompt)
        if chosen == "claude":
            argv = ["claude", "-p", *self.claude_args]
            result = self.runner.run(argv, cwd=cwd, timeout=self.claude_timeout_s, input=prompt)
            if result.error:
                return DelegationResult("claude", False, result.error, prompt)
            output = result.stdout
            if "--output-format" in self.claude_args:
                with contextlib.suppress(ValueError, AttributeError):
                    output = str(json.loads(result.stdout).get("result", result.stdout))
            if not result.ok:
                return DelegationResult("claude", False, f"claude terminó con código {result.returncode}",
                                        result.stderr or output)  # fmt: skip
            return DelegationResult("claude", True, "enviado a claude -p", output)
        return DelegationResult("print", True, "prompt mostrado", prompt)
