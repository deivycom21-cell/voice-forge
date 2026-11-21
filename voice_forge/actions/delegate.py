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
    """Routes prompts.

    * Code orders go to ``target`` (clipboard / claude / print) via :meth:`deliver`.
    * Questions are answered by ``claude -p`` via :meth:`ask` when :attr:`can_answer`
      (target is ``claude`` or ``answer_with_claude`` is set, e.g. ``--llm claude``).
    """

    def __init__(
        self,
        target: Target,
        runner: CommandRunner,
        *,
        claude_args: Sequence[str] = (),
        claude_timeout_s: float = 600.0,
        copy: Callable[[str], None] | None = None,
        answer_with_claude: bool = False,
    ) -> None:
        self.target = target
        self.runner = runner
        self.claude_args = tuple(claude_args)
        self.claude_timeout_s = claude_timeout_s
        self.answer_with_claude = answer_with_claude
        self._copy = copy or _default_copy

    @property
    def can_answer(self) -> bool:
        return self.target == "claude" or self.answer_with_claude

    def _run_claude(self, prompt: str, cwd: Path, *, full_output: bool) -> DelegationResult:
        argv = ["claude", "-p", *self.claude_args]
        result = self.runner.run(
            argv, cwd=cwd, timeout=self.claude_timeout_s, input=prompt, full_output=full_output
        )
        if result.error:
            return DelegationResult("claude", False, result.error, prompt)
        output = result.stdout
        if "--output-format" in self.claude_args:
            with contextlib.suppress(ValueError, AttributeError):
                output = str(json.loads(result.stdout).get("result", result.stdout))
        if not result.ok:
            return DelegationResult(
                "claude", False, f"claude terminó con código {result.returncode}",
                result.stderr or output,
            )  # fmt: skip
        return DelegationResult("claude", True, "enviado a claude -p", output)

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
            return self._run_claude(prompt, cwd, full_output=False)
        return DelegationResult("print", True, "prompt mostrado", prompt)

    def ask(self, question: str, cwd: Path) -> DelegationResult:
        """Send a clean question to ``claude -p`` and return its complete answer."""
        res = self._run_claude(question, cwd, full_output=True)
        if not res.ok:
            return res
        answer = res.output.strip()
        if not answer:
            return DelegationResult("claude", False, "claude no devolvió ninguna respuesta", "")
        return DelegationResult(
            "claude", True, f"respuesta de claude ({len(answer)} caracteres) · voice-forge last",
            answer,
        )  # fmt: skip
