from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from voice_forge.core import events as ev
from voice_forge.core.types import ActionResult, Intent, IntentKind, Risk, Status, Transcript
from voice_forge.tts.speaker import (
    PHRASES,
    AsyncSpeaker,
    NullSpeaker,
    RecordingSpeaker,
    Speaker,
    build_speaker,
    phrase,
    pick_voice,
)
from voice_forge.ui.hud import Hud, HudState, level_bar, render_text

# --------------------------------------------------------------------------- TTS


def test_phrases_have_same_keys_in_both_languages() -> None:
    assert set(PHRASES["es"]) == set(PHRASES["en"])
    assert phrase("done", "es") == "Listo" and phrase("done", "en") == "Done"
    assert phrase("nope", "es") == "nope" and phrase("nope", "es", "done") == "Listo"
    assert phrase("done", "fr") == "Listo"


def test_build_speaker_selection(tmp_path: Path) -> None:
    assert isinstance(build_speaker(enabled=False), NullSpeaker)
    assert isinstance(build_speaker(engine="none"), NullSpeaker)
    assert isinstance(build_speaker(engines=[]), NullSpeaker)
    # piper without a voice model falls back to the next engine
    s = build_speaker(
        engines=["piper", "espeak"],
        which=lambda x: "/usr/bin/espeak-ng" if x == "espeak-ng" else None,
    )
    assert s.name == "espeak"
    s.close()
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"x")
    p = build_speaker(engines=["piper"], piper_model=str(model))
    assert p.name == "piper"
    p.close()
    assert build_speaker(engine="pyttsx3", engines=["espeak"]).name == "none"


def test_async_speaker_is_non_blocking_and_ordered() -> None:
    said: list[str] = []
    gate = threading.Event()

    def factory():
        def speak(text: str) -> None:
            gate.wait(2)
            said.append(text)

        return speak

    sp = AsyncSpeaker("fake", factory)
    assert isinstance(sp, Speaker)
    t0 = time.perf_counter()
    sp.say("uno")
    sp.say("dos")
    assert time.perf_counter() - t0 < 0.1  # did not wait for the slow engine
    assert not sp.wait_idle(0.05)
    gate.set()
    assert sp.wait_idle(2)
    assert said == ["uno", "dos"]
    sp.muted = True
    sp.say("tres")
    sp.close()
    sp.join(2)
    assert said == ["uno", "dos"]


def test_async_speaker_queue_full_drops() -> None:
    gate = threading.Event()
    said: list[str] = []
    sp = AsyncSpeaker("fake", lambda: lambda t: (gate.wait(2), said.append(t)), max_queue=1)
    for i in range(5):
        sp.say(str(i))
    gate.set()
    sp.wait_idle(2)
    assert 1 <= len(said) <= 2


def test_async_speaker_degrades_on_engine_errors() -> None:
    def bad_factory():
        raise RuntimeError("no SAPI5")

    sp = AsyncSpeaker("bad", bad_factory)
    sp.join(2)
    assert isinstance(sp.failed, RuntimeError)
    sp.say("ignored")  # no exception
    assert sp.wait_idle(0.1)

    def factory():
        def speak(text: str) -> None:
            raise OSError("audio device lost")

        return speak

    sp2 = AsyncSpeaker("flaky", factory)
    sp2.say("x")
    sp2.join(2)
    assert isinstance(sp2.failed, OSError)
    sp2.say("y")


def test_pick_voice() -> None:
    voices = [
        SimpleNamespace(id="TTS_MS_EN-US_ZIRA", name="Microsoft Zira - English", languages=[]),
        SimpleNamespace(
            id="TTS_MS_ES-MX_SABINA", name="Microsoft Sabina - Spanish (Mexico)", languages=[]
        ),
        SimpleNamespace(id="spanish-latin", name="espeak", languages=[b"\x05es-419"]),
    ]
    assert pick_voice(voices, "", "es") == "TTS_MS_ES-MX_SABINA"
    assert pick_voice(voices, "", "en") == "TTS_MS_EN-US_ZIRA"
    assert pick_voice(voices, "latin", "en") == "spanish-latin"
    assert pick_voice(voices[2:], "", "es") == "spanish-latin"
    assert pick_voice([], "", "es") is None


def test_recording_and_null_speakers() -> None:
    r = RecordingSpeaker()
    r.say("a")
    r.muted = True
    r.say("b")
    r.say("")
    assert r.said == ["a"]
    NullSpeaker().say("x")


# --------------------------------------------------------------------------- HUD


def test_hud_reducer_and_render() -> None:
    s = HudState(hint="CTRL+ALT+V", model="base", cwd="/repo", branch="main")
    intent = Intent(IntentKind.EXECUTE, "corre los tests", argv=("pytest", "-q"), name="tests.run")
    for e in [
        ev.ListeningStarted(),
        ev.LevelUpdate(-20.0),
        ev.Transcribed(Transcript("corre los tests")),
        ev.IntentResolved(intent, Risk.SAFE),
        ev.ActionStarted(intent),
        ev.ActionFinished(
            intent, ActionResult(("pytest", "-q"), 0, stdout="5 passed", duration_s=1.2)
        ),
        ev.StatusChanged(Status.IDLE),
    ]:
        s.apply(e)
    text = render_text(s)
    for needle in ("EN ESPERA", "CTRL+ALT+V", "“corre los tests”", "$ pytest -q", "5 passed", "historial",
                   "código 0", "⎇ main", "offline"):  # fmt: skip
        assert needle in text, needle


def test_hud_confirmation_block_and_delegation() -> None:
    s = HudState()
    push = Intent(IntentKind.EXECUTE, argv=("git", "push", "--force-with-lease"))
    s.apply(ev.IntentResolved(push, Risk.CONFIRM, ("push forzado",)))
    s.apply(ev.ConfirmationRequested(push, ("push forzado",), 10))
    assert s.status == Status.CONFIRMING
    assert 'di "confirmo"' in render_text(s)
    s.apply(ev.ConfirmationResolved(False, "tiempo agotado"))
    assert "cancelado (tiempo agotado)" in render_text(s)
    s.apply(ev.Blocked(push, ("x",)))
    s.apply(ev.Delegated("clipboard", "## Tarea\nhaz algo", True, "prompt copiado"))
    s.apply(ev.IntentResolved(Intent(IntentKind.DELEGATE, prompt="p", source="llm"), Risk.SAFE))
    s.apply(ev.IntentResolved(Intent(IntentKind.UNKNOWN), Risk.SAFE))
    s.apply(ev.Info("hola"))
    s.apply(ev.ErrorEvent("stt", "boom"))
    s.apply(ev.NothingHeard())
    s.apply(ev.ListeningStarted(auto=True))
    out = render_text(s)
    assert "¿? no reconocido" in out


def test_hud_history_is_capped() -> None:
    s = HudState()
    i = Intent(IntentKind.EXECUTE, argv=("ls",))
    for _ in range(20):
        s.apply(ev.ActionFinished(i, ActionResult(("ls",), 0)))
    assert len(s.history) == 8


def test_level_bar() -> None:
    assert str(level_bar(-100, 10)) == "▯" * 10
    assert str(level_bar(0, 10)) == "▮" * 10


def test_hud_live_wrapper_updates() -> None:
    from rich.console import Console

    hud = Hud(
        HudState(),
        Console(file=open(__import__("os").devnull, "w", encoding="utf-8"), force_terminal=True),
    )
    with hud:
        hud(ev.StatusChanged(Status.LISTENING))
        hud(ev.LevelUpdate(-10))
    hud(ev.Info("after exit"))
    assert hud.state.message == "after exit"
