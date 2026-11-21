"""Minimal dark terminal HUD built with ``rich``.

:class:`HudState` is a pure reducer over events (easy to test); :func:`render`
turns it into a renderable; :class:`Hud` wires both to ``rich.live.Live``.
"""

from __future__ import annotations

import io
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from voice_forge.core import events as ev
from voice_forge.core.types import IntentKind, Risk, Status

STATUS_STYLE = {
    Status.IDLE: ("○", "grey62", "en espera"),
    Status.LISTENING: ("●", "bold red", "escuchando"),
    Status.TRANSCRIBING: ("◌", "bold yellow", "transcribiendo"),
    Status.THINKING: ("◆", "bold magenta", "pensando"),
    Status.CONFIRMING: ("?", "bold dark_orange", "esperando confirmación"),
    Status.EXECUTING: ("▶", "bold cyan", "ejecutando"),
}
RISK_STYLE = {Risk.SAFE: "green", Risk.CONFIRM: "dark_orange", Risk.BLOCKED: "red"}


@dataclass
class HistoryItem:
    when: str
    label: str
    ok: bool | None
    detail: str = ""


@dataclass
class HudState:
    hint: str = "CTRL+ALT+V"
    model: str = ""
    llm: str = "none"
    target: str = "clipboard"
    cwd: str = ""
    branch: str = ""
    dry_run: bool = False
    status: Status = Status.IDLE
    level_db: float = -100.0
    transcript: str = ""
    command: str = ""
    risk: Risk = Risk.SAFE
    reasons: tuple[str, ...] = ()
    message: str = ""
    output: str = ""
    history: deque[HistoryItem] = field(default_factory=lambda: deque(maxlen=8))

    def apply(self, event: ev.Event) -> None:
        if isinstance(event, ev.StatusChanged):
            self.status = event.status
            if event.status == Status.IDLE:
                self.level_db = -100.0
        elif isinstance(event, ev.ListeningStarted):
            self.status = Status.LISTENING
            self.transcript = ""
            self.message = 'di "confirmo" o "cancela"' if event.auto else ""
        elif isinstance(event, ev.LevelUpdate):
            self.level_db = event.level_db
        elif isinstance(event, ev.NothingHeard):
            self.message = "no se oyó nada"
        elif isinstance(event, ev.Transcribed):
            self.transcript = event.transcript.text
        elif isinstance(event, ev.IntentResolved):
            intent = event.intent
            self.risk = event.risk
            self.reasons = event.reasons
            if intent.kind == IntentKind.EXECUTE:
                self.command = "$ " + " ".join(intent.argv)
            elif intent.kind == IntentKind.DELEGATE:
                self.command = f"→ {self.target}: prompt ({len(intent.prompt or '')} caracteres)"
            elif intent.kind == IntentKind.UNKNOWN:
                self.command = "¿? no reconocido"
            else:
                self.command = intent.display
            if intent.source == "llm":
                self.command += "  [IA]"
        elif isinstance(event, ev.ConfirmationRequested):
            self.status = Status.CONFIRMING
            self.message = f'⚠ {"; ".join(event.reasons)} — di "confirmo" ({event.timeout_s:.0f}s)'
        elif isinstance(event, ev.ConfirmationResolved):
            self.message = "confirmado" if event.accepted else f"cancelado ({event.reason})"
            if not event.accepted:
                self._push(self.command.lstrip("$ "), False, event.reason)
        elif isinstance(event, ev.ActionStarted):
            self.status = Status.EXECUTING
            self.output = ""
        elif isinstance(event, ev.ActionFinished):
            r = event.result
            self.output = (r.stdout + ("\n" + r.stderr if r.stderr else "")).strip()
            detail = r.error or (
                "en segundo plano" if r.detached else f"código {r.returncode} · {r.duration_s:.1f}s"
            )
            self.message = detail
            self._push(event.intent.display, r.ok, detail)
        elif isinstance(event, ev.Delegated):
            self.message = event.detail
            if event.target == "print" or not event.ok:
                self.output = event.prompt
            else:
                self.output = event.prompt[:600]
            self._push(f"→ {event.target}", event.ok, event.detail)
        elif isinstance(event, ev.Blocked):
            self.message = "⛔ bloqueado: " + "; ".join(event.reasons)
            self._push(event.intent.display, False, "bloqueado")
        elif isinstance(event, ev.Info):
            self.message = event.message
        elif isinstance(event, ev.ErrorEvent):
            self.message = f"error en {event.where}: {event.message}"

    def _push(self, label: str, ok: bool | None, detail: str) -> None:
        self.history.appendleft(HistoryItem(time.strftime("%H:%M:%S"), label, ok, detail))


def level_bar(level_db: float, width: int = 24) -> Text:
    frac = min(1.0, max(0.0, (level_db + 60.0) / 60.0))
    filled = round(frac * width)
    bar = Text("▮" * filled, style="green" if frac < 0.8 else "yellow")
    bar.append("▯" * (width - filled), style="grey30")
    return bar


def render(state: HudState) -> RenderableType:
    icon, style, label = STATUS_STYLE[state.status]
    header = Table.grid(expand=True)
    header.add_column(ratio=1)
    header.add_column(justify="right")
    left = Text.assemble((f" {icon} ", style), (label.upper(), style), "   ")
    left.append_text(level_bar(state.level_db))
    sends_text = state.llm == "claude" or state.target == "claude"
    right = Text.assemble(
        ("voice-forge ", "bold"),
        (f"· {state.hint} ", "cyan"),
        (f"· {state.model} ", "grey62"),
        (f"· llm:{state.llm} ", "grey62"),
        ("· DRY-RUN " if state.dry_run else "", "bold yellow"),
        ("· texto→claude" if sends_text else "· offline", "yellow" if sends_text else "green"),
    )
    header.add_row(left, right)
    if state.cwd:
        header.add_row(
            Text(f" {state.cwd}" + (f"  ⎇ {state.branch}" if state.branch else ""), style="grey50"),
            "",
        )

    body = Table.grid(padding=(0, 1))
    body.add_column(style="grey62", width=10)
    body.add_column()
    body.add_row(
        "dijiste", Text(f"“{state.transcript}”" if state.transcript else "—", style="white")
    )
    body.add_row("comando", Text(state.command or "—", style=RISK_STYLE[state.risk]))
    if state.message:
        body.add_row("estado", Text(state.message, style="grey74"))
    parts: list[RenderableType] = [header, Panel(body, border_style="grey35", padding=(0, 1))]
    if state.output:
        lines = state.output.splitlines()[-10:]
        parts.append(Panel(Text("\n".join(lines), style="grey70"), title="salida", title_align="left",
                           border_style="grey23", padding=(0, 1)))  # fmt: skip
    if state.history:
        hist = Table(box=None, show_header=False, padding=(0, 1), expand=True)
        hist.add_column(style="grey50", width=8)
        hist.add_column(width=2)
        hist.add_column(ratio=1, overflow="ellipsis", no_wrap=True)
        hist.add_column(style="grey50", justify="right")
        for item in state.history:
            mark = (
                Text("✓", style="green")
                if item.ok
                else Text("✗", style="red")
                if item.ok is False
                else Text("·")
            )
            hist.add_row(item.when, mark, item.label, item.detail)
        parts.append(
            Panel(
                hist, title="historial", title_align="left", border_style="grey23", padding=(0, 1)
            )
        )
    return Group(*parts)


def render_text(state: HudState, width: int = 100) -> str:
    """Render to plain text (tests / non-interactive output)."""
    console = Console(
        width=width, record=True, color_system=None, force_terminal=False, file=io.StringIO()
    )
    console.print(render(state))
    return console.export_text()


class Hud:
    """Subscribes to the bus and keeps a ``rich.live.Live`` display updated."""

    def __init__(
        self, state: HudState, console: Console | None = None, refresh_per_second: int = 12
    ) -> None:
        self.state = state
        self.console = console or Console()
        self._lock = threading.Lock()
        self._live: Any = None
        self.refresh_per_second = refresh_per_second

    def __call__(self, event: ev.Event) -> None:
        with self._lock:
            self.state.apply(event)
            if self._live is not None and not isinstance(event, ev.LevelUpdate):
                self._live.update(render(self.state))
            elif self._live is not None:
                self._live.update(render(self.state), refresh=False)

    def __enter__(self) -> Hud:
        from rich.live import Live

        self._live = Live(render(self.state), console=self.console, refresh_per_second=self.refresh_per_second,
                          transient=False, redirect_stdout=False)  # fmt: skip
        self._live.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._live is not None:
            self._live.__exit__(None, None, None)
            self._live = None
