"""Push-to-talk triggers.

* :class:`PynputTrigger` — global hotkey (works while another window has focus).
* :class:`TerminalTrigger` — fallback: press Enter in the voice-forge terminal
  (used on Wayland / headless Linux, where global hooks are unavailable).
* :class:`ManualTrigger` — programmatic, used by tests.

Modes:
  ``toggle``  first press starts recording, second press stops it (VAD also stops it).
  ``hold``    recording lasts while the combo is held down.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Mapping
from typing import Any, Literal

Mode = Literal["toggle", "hold"]

_SPECIAL = {
    "ctrl", "alt", "shift", "cmd", "space", "enter", "tab", "esc", "backspace",
    "home", "end", "insert", "delete", "page_up", "page_down", "up", "down", "left", "right",
    "caps_lock", "alt_gr", "ctrl_l", "ctrl_r", "alt_l", "alt_r", "shift_l", "shift_r",
    "cmd_l", "cmd_r", "pause", "menu", "scroll_lock", "print_screen",
}  # fmt: skip
_ALIASES = {"control": "ctrl", "win": "cmd", "super": "cmd", "meta": "cmd", "escape": "esc",
            "return": "enter", "del": "delete", "option": "alt"}  # fmt: skip


class HotkeyError(ValueError):
    pass


def parse_combo(combo: str) -> str:
    """Convert ``"ctrl+alt+v"`` into pynput's ``"<ctrl>+<alt>+v"`` format."""
    parts = [p.strip().lower() for p in combo.replace(" ", "").split("+")]
    if not parts or any(not p for p in parts):
        raise HotkeyError(f"invalid hotkey: {combo!r}")
    out = []
    for part in parts:
        part = _ALIASES.get(part, part)
        if part in _SPECIAL or (
            part.startswith("f") and part[1:].isdigit() and 1 <= int(part[1:]) <= 24
        ):
            out.append(f"<{part}>")
        elif len(part) == 1:
            out.append(part)
        else:
            raise HotkeyError(f"unknown key {part!r} in hotkey {combo!r}")
    if len(set(out)) != len(out):
        raise HotkeyError(f"duplicated key in hotkey {combo!r}")
    return "+".join(out)


def detect_backend(
    env: Mapping[str, str] | None = None, platform: str | None = None
) -> Literal["pynput", "terminal"]:
    """Pick the trigger backend usable on this system."""
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else platform
    if platform.startswith("linux"):
        if env.get("WAYLAND_DISPLAY") or env.get("XDG_SESSION_TYPE") == "wayland":
            return "terminal"
        if not env.get("DISPLAY"):
            return "terminal"
    try:
        import pynput  # noqa: F401
    except Exception:
        return "terminal"
    return "pynput"


class BaseTrigger:
    """Shared press/release logic; subclasses call :meth:`_on_press` / :meth:`_on_release`."""

    name = "base"

    def __init__(self, mode: Mode = "toggle") -> None:
        if mode not in ("toggle", "hold"):
            raise HotkeyError(f"invalid mode {mode!r}")
        self.mode: Mode = mode
        self._activation = threading.Event()
        self._lock = threading.Lock()
        self._stop_target: threading.Event | None = None
        self._held = False

    # ---- called by backends
    def _on_press(self) -> None:
        with self._lock:
            self._held = True
            if self._stop_target is not None and self.mode == "toggle":
                self._stop_target.set()
                self._stop_target = None
            elif self._stop_target is None:
                self._activation.set()

    def _on_release(self) -> None:
        with self._lock:
            self._held = False
            if self._stop_target is not None and self.mode == "hold":
                self._stop_target.set()
                self._stop_target = None

    # ---- used by the orchestrator
    def wait_activation(self, timeout: float | None = None) -> bool:
        if self._activation.wait(timeout):
            self._activation.clear()
            return True
        return False

    def arm_stop(self, event: threading.Event) -> None:
        """Route the next stop gesture (2nd press / release) to ``event``."""
        with self._lock:
            if self.mode == "hold" and not self._held:
                event.set()  # already released before we armed
                return
            self._stop_target = event

    def disarm(self) -> None:
        with self._lock:
            self._stop_target = None

    def start(self) -> None:  # pragma: no cover - overridden
        pass

    def stop(self) -> None:  # pragma: no cover - overridden
        pass

    @property
    def hint(self) -> str:
        return ""


class ManualTrigger(BaseTrigger):
    """Programmatic trigger for tests and scripted runs."""

    name = "manual"

    def press(self) -> None:
        self._on_press()

    def release(self) -> None:
        self._on_release()


class PynputTrigger(BaseTrigger):
    name = "pynput"

    def __init__(
        self, combo: str = "ctrl+alt+v", mode: Mode = "toggle", keyboard_module: Any = None
    ) -> None:
        super().__init__(mode)
        self.combo = combo
        self.pynput_combo = parse_combo(combo)
        self._kb = keyboard_module
        self._listener: Any = None

    def start(self) -> None:
        kb: Any = self._kb
        if kb is None:
            from pynput import keyboard

            kb = keyboard
        hotkey = kb.HotKey(kb.HotKey.parse(self.pynput_combo), self._on_press)
        listener: Any = None

        def on_press(key: Any) -> None:
            hotkey.press(listener.canonical(key))

        def on_release(key: Any) -> None:
            hotkey.release(listener.canonical(key))
            if self._held:
                self._on_release()

        listener = kb.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()
        self._listener = listener

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    @property
    def hint(self) -> str:
        return self.combo.upper()


class TerminalTrigger(BaseTrigger):
    """Press Enter in the terminal. Hold mode is not possible -> always toggle."""

    name = "terminal"

    def __init__(self, stream: Any = None) -> None:
        super().__init__("toggle")
        self._stream = stream if stream is not None else sys.stdin
        self._thread: threading.Thread | None = None
        self._running = threading.Event()

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="vf-terminal-trigger", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running.is_set():
            line = self._stream.readline()
            if line == "":  # EOF
                return
            self._on_press()
            self._on_release()

    def stop(self) -> None:
        self._running.clear()

    @property
    def hint(self) -> str:
        return "ENTER"


def build_trigger(combo: str, mode: Mode, backend: str = "auto") -> BaseTrigger:
    chosen = detect_backend() if backend == "auto" else backend
    if chosen == "pynput":
        return PynputTrigger(combo, mode)
    if chosen == "terminal":
        return TerminalTrigger()
    raise HotkeyError(f"unknown hotkey backend {backend!r}")
