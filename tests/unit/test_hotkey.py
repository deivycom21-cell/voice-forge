from __future__ import annotations

import io
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from voice_forge.audio.hotkey import (
    HotkeyError,
    ManualTrigger,
    PynputTrigger,
    TerminalTrigger,
    build_trigger,
    detect_backend,
    parse_combo,
)


@pytest.mark.parametrize(
    ("combo", "expected"),
    [
        ("ctrl+alt+v", "<ctrl>+<alt>+v"),
        ("Ctrl + Space", "<ctrl>+<space>"),
        ("alt+v", "<alt>+v"),
        ("control+shift+F9", "<ctrl>+<shift>+<f9>"),
        ("win+escape", "<cmd>+<esc>"),
    ],
)
def test_parse_combo(combo: str, expected: str) -> None:
    assert parse_combo(combo) == expected


@pytest.mark.parametrize("combo", ["", "ctrl+", "ctrl+banana", "ctrl+ctrl", "f99"])
def test_parse_combo_invalid(combo: str) -> None:
    with pytest.raises(HotkeyError):
        parse_combo(combo)


def test_detect_backend() -> None:
    assert detect_backend({"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}, "linux") == "terminal"
    assert detect_backend({"XDG_SESSION_TYPE": "wayland"}, "linux") == "terminal"
    assert detect_backend({}, "linux") == "terminal"  # headless


def test_detect_backend_without_pynput(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *a: Any, **k: Any) -> Any:
        if name == "pynput":
            raise ImportError("nope")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert detect_backend({}, "win32") == "terminal"


def test_toggle_mode_second_press_stops() -> None:
    t = ManualTrigger("toggle")
    assert not t.wait_activation(0.01)
    t.press()
    t.release()
    assert t.wait_activation(0.01)
    stop = threading.Event()
    t.arm_stop(stop)
    assert not stop.is_set()
    t.press()
    assert stop.is_set()
    assert not t.wait_activation(0.01)  # the stop press is not a new activation


def test_hold_mode_release_stops() -> None:
    t = ManualTrigger("hold")
    t.press()
    assert t.wait_activation(0.01)
    stop = threading.Event()
    t.arm_stop(stop)
    assert not stop.is_set()
    t.release()
    assert stop.is_set()


def test_hold_mode_released_before_arming_stops_immediately() -> None:
    t = ManualTrigger("hold")
    t.press()
    t.release()
    stop = threading.Event()
    t.arm_stop(stop)
    assert stop.is_set()


def test_disarm_and_invalid_mode() -> None:
    t = ManualTrigger("toggle")
    stop = threading.Event()
    t.arm_stop(stop)
    t.disarm()
    t.press()
    assert not stop.is_set() and t.wait_activation(0.01)
    with pytest.raises(HotkeyError):
        ManualTrigger("sometimes")  # type: ignore[arg-type]


def test_terminal_trigger_enter_activates() -> None:
    t = TerminalTrigger(stream=io.StringIO("\n"))
    t.start()
    assert t.wait_activation(1.0)
    t.stop()
    assert t.hint == "ENTER" and t.mode == "toggle"


class FakeHotKey:
    @staticmethod
    def parse(combo: str) -> list[str]:
        return combo.split("+")

    def __init__(self, keys: list[str], on_activate: Any) -> None:
        self.keys = set(keys)
        self.state: set[str] = set()
        self.on_activate = on_activate

    def press(self, key: str) -> None:
        if key in self.keys and key not in self.state:
            self.state.add(key)
            if self.state == self.keys:
                self.on_activate()

    def release(self, key: str) -> None:
        self.state.discard(key)


class FakeListener:
    def __init__(self, on_press: Any, on_release: Any) -> None:
        self.on_press = on_press
        self.on_release = on_release
        self.started = self.stopped = False
        self.daemon = False

    def canonical(self, key: str) -> str:
        return key

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


def test_pynput_trigger_with_fake_keyboard() -> None:
    kb = SimpleNamespace(HotKey=FakeHotKey, Listener=FakeListener)
    t = PynputTrigger("ctrl+alt+v", "hold", keyboard_module=kb)
    t.start()
    listener = t._listener
    assert listener.started and listener.daemon
    for k in ("<ctrl>", "<alt>", "v"):
        listener.on_press(k)
    assert t.wait_activation(0.01)
    stop = threading.Event()
    t.arm_stop(stop)
    listener.on_release("v")
    assert stop.is_set()
    t.stop()
    assert listener.stopped and t.hint == "CTRL+ALT+V"


def test_build_trigger() -> None:
    assert isinstance(build_trigger("ctrl+alt+v", "toggle", "terminal"), TerminalTrigger)
    assert isinstance(build_trigger("ctrl+alt+v", "hold", "pynput"), PynputTrigger)
    with pytest.raises(HotkeyError):
        build_trigger("ctrl+alt+v", "toggle", "carrier-pigeon")
