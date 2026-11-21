from __future__ import annotations

from pathlib import Path

from voice_forge.core import events as ev
from voice_forge.core.types import ActionResult, ContextSnapshot, Intent, IntentKind, Risk, Status


def test_bus_delivers_in_order_and_filters_by_type() -> None:
    bus = ev.EventBus()
    got: list[str] = []
    bus.subscribe(lambda e: got.append(f"all:{type(e).__name__}"))
    bus.subscribe(lambda e: got.append("info"), ev.Info)
    bus.publish(ev.StatusChanged(Status.IDLE))
    bus.publish(ev.Info("x"))
    assert got == ["all:StatusChanged", "all:Info", "info"]


def test_failing_handler_does_not_break_others() -> None:
    bus = ev.EventBus()
    got: list[ev.Event] = []

    def boom(_: ev.Event) -> None:
        raise RuntimeError("boom")

    bus.subscribe(boom)
    bus.subscribe(got.append)
    bus.publish(ev.Info("hi"))
    assert len(got) == 1
    assert isinstance(bus.handler_errors[0], RuntimeError)


def test_unsubscribe() -> None:
    bus = ev.EventBus()
    got: list[ev.Event] = []
    unsub = bus.subscribe(got.append)
    unsub()
    unsub()  # idempotent
    bus.publish(ev.Info("x"))
    assert got == []


def test_recording_bus() -> None:
    bus = ev.RecordingBus()
    bus.publish(ev.Info("a"))
    bus.publish(ev.NothingHeard())
    assert bus.types() == ["Info", "NothingHeard"]
    assert [e.message for e in bus.of_type(ev.Info)] == ["a"]


def test_intent_display_and_result_ok() -> None:
    assert Intent(IntentKind.EXECUTE, argv=("git", "status")).display == "git status"
    assert Intent(IntentKind.EXECUTE, argv=("x",), label="lbl").display == "lbl"
    assert Intent(IntentKind.UNKNOWN).display == "unknown"
    assert ActionResult(("x",), returncode=0).ok
    assert not ActionResult(("x",), returncode=1).ok
    assert ActionResult(("x",), detached=True).ok
    assert not ActionResult(("x",), returncode=0, error="e").ok
    assert Risk.BLOCKED > Risk.CONFIRM > Risk.SAFE


def test_context_project_kind(tmp_path: Path) -> None:
    def kind(*markers: str) -> str:
        return ContextSnapshot(cwd=tmp_path, markers=markers).project_kind

    assert kind("pyproject.toml") == "python"
    assert kind("package.json") == "node"
    assert kind("Cargo.toml") == "rust"
    assert kind("go.mod") == "go"
    assert kind() == "unknown"
    snap = ContextSnapshot(cwd=tmp_path / "sub", git_root=tmp_path)
    assert snap.root == tmp_path and snap.is_git
