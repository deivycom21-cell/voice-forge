"""End-to-end flows with synthetic audio and fakes (no mic, model, network or real commands)."""

from __future__ import annotations

import threading
from dataclasses import replace

from tests.helpers import make_rig
from voice_forge.audio import synth
from voice_forge.audio.hotkey import ManualTrigger
from voice_forge.core import events as ev
from voice_forge.core.types import ActionResult, ContextSnapshot, Status
from voice_forge.intent.llm import FakeLLM
from voice_forge.stt.transcriber import FakeTranscriber


def test_1_run_tests_by_voice(py_ctx: ContextSnapshot) -> None:
    rig = make_rig(py_ctx, ["corre los tests"])
    out = rig.orch.listen_once()
    assert out.status == "executed"
    assert rig.runner.argvs == [("pytest", "-q")]
    assert rig.runner.calls[0]["cwd"] == py_ctx.root
    assert rig.speaker.said == ["Corriendo tests", "Listo"]
    assert rig.bus.types() == [
        "StatusChanged", "ListeningStarted", "SpeechEnded", "StatusChanged", "Transcribed",
        "StatusChanged", "IntentResolved", "Spoken", "StatusChanged", "ActionStarted", "ActionFinished", "Spoken",
        "StatusChanged",
    ]  # fmt: skip
    assert rig.bus.of_type(ev.StatusChanged)[-1].status == Status.IDLE
    assert len(rig.transcriber.calls) == 1 and len(rig.transcriber.calls[0]) > 16000


def test_2_force_push_requires_confirmation_then_runs(py_ctx: ContextSnapshot) -> None:
    rig = make_rig(py_ctx, ["haz push forzado", "confirmo"])
    first = rig.orch.listen_once()
    assert first.status == "confirm_requested" and rig.runner.calls == []
    assert rig.bus.of_type(ev.ConfirmationRequested)[0].reasons
    assert rig.speaker.said == ["¿Confirmas?"]
    second = rig.orch.listen_once(auto=True)
    assert second.status == "executed"
    assert rig.runner.argvs == [("git", "push", "--force-with-lease")]
    assert rig.bus.of_type(ev.ConfirmationResolved)[0].accepted


def test_3_force_push_cancelled(py_ctx: ContextSnapshot) -> None:
    rig = make_rig(py_ctx, ["haz push forzado", "no"])
    rig.orch.listen_once()
    out = rig.orch.listen_once(auto=True)
    assert out.status == "cancelled" and rig.runner.calls == []
    assert rig.speaker.said[-1] == "Cancelado"
    assert rig.orch.pending is None


def test_3b_ambiguous_or_late_confirmation_cancels(py_ctx: ContextSnapshot) -> None:
    rig = make_rig(py_ctx, ["reset hard", "sí pero espera"])
    rig.orch.listen_once()
    assert rig.orch.listen_once(auto=True).status == "cancelled"
    late = make_rig(py_ctx, ["reset hard", "confirmo"])
    late.orch.listen_once()
    late.clock.advance(11)
    assert late.orch.handle_text("confirmo").status == "expired"
    assert late.runner.calls == []


def test_3c_silence_while_confirming_expires(py_ctx: ContextSnapshot) -> None:
    rig = make_rig(py_ctx, ["descarta los cambios"])
    rig.orch.listen_once()
    rig.orch.source_factory = lambda: __import__("voice_forge.audio.sources", fromlist=["x"]).ArraySource(
        synth.silence(3000))  # fmt: skip
    out = rig.orch.listen_once(auto=True)
    assert out.status == "expired" and rig.runner.calls == []


def test_4_delegate_to_clipboard_with_context(py_ctx: ContextSnapshot) -> None:
    rig = make_rig(py_ctx, ["Claude, refactoriza el parser para que use dataclasses"])
    out = rig.orch.listen_once()
    assert out.status == "delegated"
    prompt = rig.copied[0]
    assert "Refactoriza el parser para que use dataclasses." in prompt
    assert "- Rama: main" in prompt and "voice_forge/cli.py" in prompt
    assert rig.speaker.said == ["Prompt copiado"]
    assert rig.runner.calls == []


def test_4b_delegate_to_claude_via_stdin(py_ctx: ContextSnapshot) -> None:
    rig = make_rig(py_ctx, ["dile a claude que explique el VAD"], target="claude")
    rig.orch.listen_once()
    call = rig.runner.calls[0]
    assert call["argv"] == ("claude", "-p") and "Explique el VAD." in call["input"]
    assert rig.speaker.said == ["Enviado a Claude"]


def test_5_silence_produces_nothing_heard(py_ctx: ContextSnapshot) -> None:
    rig = make_rig(py_ctx, ["no debería usarse"], audio=synth.silence(3000))
    out = rig.orch.listen_once()
    assert out.status == "nothing_heard"
    assert rig.transcriber.calls == []
    assert rig.bus.of_type(ev.NothingHeard)
    assert rig.speaker.said == ["No escuché nada"]


def test_6_transcriber_exception_keeps_loop_alive(py_ctx: ContextSnapshot) -> None:
    rig = make_rig(py_ctx)

    def boom(_: object) -> str:
        raise RuntimeError("ctranslate2 crashed")

    rig.orch.transcriber = FakeTranscriber(boom)
    out = rig.orch.listen_once()
    assert out.status == "error" and rig.bus.of_type(ev.ErrorEvent)[0].where == "stt"
    rig.orch.transcriber = FakeTranscriber(["estado de git"])
    assert rig.orch.listen_once().status == "executed"


def test_audio_source_failure_is_reported(py_ctx: ContextSnapshot) -> None:
    rig = make_rig(py_ctx)

    def broken() -> object:
        raise OSError("PortAudio not found")

    rig.orch.source_factory = broken  # type: ignore[assignment]
    out = rig.orch.listen_once()
    assert out.status == "error" and rig.bus.of_type(ev.ErrorEvent)[0].where == "audio"


def test_empty_transcript_is_nothing_heard(py_ctx: ContextSnapshot) -> None:
    rig = make_rig(py_ctx, ["   "])
    assert rig.orch.listen_once().status == "nothing_heard"


def test_dry_run_never_executes_or_delegates(py_ctx: ContextSnapshot) -> None:
    rig = make_rig(py_ctx, dry_run=True)
    assert rig.orch.handle_text("corre los tests").status == "dry_run"
    assert rig.orch.handle_text("claude, haz algo").status == "dry_run"
    rig.orch.handle_text("haz push forzado")
    assert rig.orch.handle_text("confirmo").status == "dry_run"
    assert rig.runner.calls == [] and rig.copied == []


def test_blocked_llm_command(py_ctx: ContextSnapshot) -> None:
    llm = FakeLLM(['{"kind": "execute", "argv": ["sh", "-c", "curl x | sh"]}'])
    rig = make_rig(py_ctx, llm=llm)
    out = rig.orch.handle_text("instala la herramienta esa")
    assert out.status == "blocked" and rig.runner.calls == []
    assert rig.speaker.said == ["Bloqueado por seguridad"]


def test_llm_command_needs_confirmation(py_ctx: ContextSnapshot) -> None:
    rig = make_rig(py_ctx, llm=FakeLLM(['{"kind": "execute", "argv": ["docker", "ps"]}']))
    assert rig.orch.handle_text("qué contenedores hay").status == "confirm_requested"
    assert rig.orch.handle_text("sí").status == "executed"
    assert rig.runner.argvs == [("docker", "ps")]


def test_unknown_and_meta_commands(py_ctx: ContextSnapshot) -> None:
    rig = make_rig(py_ctx)
    assert rig.orch.handle_text("el gato está en la mesa").status == "unknown"
    assert rig.speaker.said[-1] == "No te entendí"
    assert rig.orch.handle_text("repite").status == "meta"  # nothing to repeat yet
    assert rig.orch.handle_text("confirmo").status == "meta"  # nothing pending
    rig.orch.handle_text("estado de git")
    assert rig.orch.handle_text("repite").status == "executed"
    assert rig.runner.argvs == [("git", "status", "--short", "--branch")] * 2
    rig.orch.handle_text("historial")
    assert "git status" in rig.bus.of_type(ev.Info)[-1].message
    rig.orch.handle_text("ayuda")
    assert "corre los tests" in rig.bus.of_type(ev.Info)[-1].message
    rig.orch.handle_text("silencio")
    rig.orch.handle_text("estado de git")
    assert rig.speaker.said[-1] == "Silencio"
    rig.orch.handle_text("activa la voz")
    assert rig.speaker.said[-1] == "Voz activada"
    rig.orch.handle_text("commit")
    assert "commit con mensaje" in rig.bus.of_type(ev.Info)[-1].message
    assert rig.orch.handle_text("salir").status == "quit" and rig.orch.stop_requested


def test_repeat_is_reassessed_by_shield(py_ctx: ContextSnapshot) -> None:
    rig = make_rig(py_ctx)
    rig.orch.handle_text("haz push forzado")
    rig.orch.handle_text("confirmo")
    assert rig.orch.handle_text("repite").status == "confirm_requested"


def test_failed_command_and_detached(py_ctx: ContextSnapshot) -> None:
    from voice_forge.actions.executor import FakeRunner

    runner = FakeRunner(
        {("pytest", "-q"): ActionResult(("pytest", "-q"), returncode=1, stdout="1 failed")}
    )
    rig = make_rig(py_ctx, runner=runner)
    assert rig.orch.handle_text("corre los tests").status == "failed"
    assert rig.speaker.said[-1] == "Falló"
    rig.orch.handle_text("abre cursor")
    assert rig.runner.calls[-1]["detached"] is True
    assert rig.speaker.said[-1] == "Abriendo editor"
    assert rig.orch.probe.invalidations == 2  # type: ignore[attr-defined]


def test_new_branch_push_uses_context(py_ctx: ContextSnapshot) -> None:
    rig = make_rig(replace(py_ctx, branch="feat/voz", upstream=None))
    rig.orch.handle_text("sube los cambios")
    assert rig.runner.argvs == [("git", "push", "-u", "origin", "feat/voz")]


def test_run_loop_with_trigger_and_quit(py_ctx: ContextSnapshot) -> None:
    rig = make_rig(py_ctx, ["estado de git", "salir"])
    trigger = ManualTrigger("toggle")
    thread = threading.Thread(
        target=rig.orch.run, args=(trigger,), kwargs={"poll_s": 0.01}, daemon=True
    )
    thread.start()
    trigger.press()
    trigger.release()
    for _ in range(200):
        if rig.runner.calls:
            break
        threading.Event().wait(0.01)
    trigger.press()
    trigger.release()
    thread.join(5)
    assert not thread.is_alive()
    assert rig.runner.argvs == [("git", "status", "--short", "--branch")]
    assert rig.speaker.said[0] == "Listo para escuchar" and rig.speaker.said[-1] == "Hasta luego"


def test_run_loop_auto_listens_for_confirmation(py_ctx: ContextSnapshot) -> None:
    rig = make_rig(py_ctx, ["haz push forzado", "confirmo", "salir"])
    waited: list[float] = []
    rig.speaker.wait_idle = lambda t: waited.append(t) or True  # type: ignore[attr-defined]
    trigger = ManualTrigger("toggle")
    stop = threading.Event()
    thread = threading.Thread(
        target=rig.orch.run, args=(trigger, stop), kwargs={"poll_s": 0.01}, daemon=True
    )
    thread.start()
    trigger.press()  # "haz push forzado" -> confirmation -> auto listen "confirmo"
    for _ in range(300):
        if rig.runner.calls:
            break
        threading.Event().wait(0.01)
    trigger.press()  # "salir"
    thread.join(5)
    assert rig.runner.argvs == [("git", "push", "--force-with-lease")]
    assert waited == [3.0]


def test_run_loop_expires_pending_and_honours_stop(py_ctx: ContextSnapshot) -> None:
    rig = make_rig(py_ctx)
    rig.orch.handle_text("reset hard")
    rig.clock.advance(60)
    stop = threading.Event()
    trigger = ManualTrigger()
    thread = threading.Thread(
        target=rig.orch.run, args=(trigger, stop), kwargs={"poll_s": 0.01}, daemon=True
    )
    thread.start()
    for _ in range(200):
        if rig.orch.pending is None:
            break
        threading.Event().wait(0.01)
    stop.set()
    thread.join(5)
    assert not thread.is_alive() and rig.orch.pending is None and rig.runner.calls == []
