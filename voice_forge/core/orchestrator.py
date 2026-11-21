"""Main loop: trigger -> record -> transcribe -> context -> intent -> shield -> act -> speak.

Every collaborator is injected, so the whole flow is testable with fakes.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voice_forge.actions.confirm import ConfirmationFlow, Decision
from voice_forge.actions.delegate import DelegationResult, Delegator
from voice_forge.actions.executor import CommandRunner
from voice_forge.actions.shield import Shield
from voice_forge.audio.recorder import Recorder
from voice_forge.audio.sources import AudioSource
from voice_forge.context.probe import ContextProbe
from voice_forge.core import events as ev
from voice_forge.core.types import ActionResult, ContextSnapshot, Intent, IntentKind, Risk, Status
from voice_forge.intent.engine import IntentEngine
from voice_forge.intent.rules import help_lines
from voice_forge.stt.transcriber import Transcriber
from voice_forge.tts.speaker import PHRASES, Speaker, phrase

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnOutcome:
    status: str
    text: str = ""
    intent: Intent | None = None
    result: ActionResult | None = None
    delegation: DelegationResult | None = None


class Orchestrator:
    def __init__(
        self,
        *,
        bus: ev.EventBus,
        recorder: Recorder,
        transcriber: Transcriber,
        probe: ContextProbe,
        engine: IntentEngine,
        shield: Shield,
        runner: CommandRunner,
        delegator: Delegator,
        speaker: Speaker,
        source_factory: Callable[[], AudioSource],
        cwd: Path | None = None,
        lang: str = "es",
        stt_language: str | None = "es",
        confirm_timeout_s: float = 10.0,
        command_timeout_s: float = 300.0,
        dry_run: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.bus = bus
        self.recorder = recorder
        self.transcriber = transcriber
        self.probe = probe
        self.engine = engine
        self.shield = shield
        self.runner = runner
        self.delegator = delegator
        self.speaker = speaker
        self.source_factory = source_factory
        self.cwd = cwd
        self.lang = lang
        self.stt_language = stt_language
        self.confirm_timeout_s = confirm_timeout_s
        self.command_timeout_s = command_timeout_s
        self.dry_run = dry_run
        self._clock = clock
        self.pending: ConfirmationFlow | None = None
        self._pending_ctx: ContextSnapshot | None = None
        self.last_executed: Intent | None = None
        self.history: list[TurnOutcome] = []
        self.stop_requested = False

    # ------------------------------------------------------------------ helpers
    def _say(self, key_or_text: str, *, raw: bool = False) -> None:
        text = key_or_text if raw else phrase(key_or_text, self.lang)
        if text:
            self.speaker.say(text)
            self.bus.publish(ev.Spoken(text))

    def _status(self, status: Status) -> None:
        self.bus.publish(ev.StatusChanged(status))

    def _finish(self, outcome: TurnOutcome) -> TurnOutcome:
        self.history.append(outcome)
        if self.pending is None:
            self._status(Status.IDLE)
        return outcome

    # ------------------------------------------------------------------ audio turn
    def listen_once(self, *, trigger: Any = None, auto: bool = False) -> TurnOutcome:
        stop_ev = threading.Event()
        if trigger is not None:
            trigger.arm_stop(stop_ev)
        self._status(Status.LISTENING)
        self.bus.publish(ev.ListeningStarted(auto=auto))
        timeout = self.pending.remaining_s if self.pending is not None else None
        source = None
        try:
            source = self.source_factory()
            rec = self.recorder.record(source, stop=stop_ev, start_timeout_s=timeout)
        except Exception as exc:
            log.exception("recording failed")
            self.bus.publish(ev.ErrorEvent("audio", str(exc)))
            self._say("error")
            return self._finish(TurnOutcome("error"))
        finally:
            if trigger is not None:
                trigger.disarm()
            if source is not None:
                source.close()

        if not rec.has_audio or rec.audio is None:
            self.bus.publish(ev.NothingHeard())
            if self.pending is not None:
                return self.expire_pending()
            self._say("nothing")
            return self._finish(TurnOutcome("nothing_heard"))

        self.bus.publish(ev.SpeechEnded(rec.duration_s, rec.reason))
        self._status(Status.TRANSCRIBING)
        try:
            transcript = self.transcriber.transcribe(rec.audio, language=self.stt_language)
        except Exception as exc:
            log.exception("transcription failed")
            self.bus.publish(ev.ErrorEvent("stt", str(exc)))
            self._say("error")
            if self.pending is not None:
                return self.expire_pending()
            return self._finish(TurnOutcome("error"))
        self.bus.publish(ev.Transcribed(transcript))
        if not transcript.text.strip():
            self.bus.publish(ev.NothingHeard())
            if self.pending is not None:
                return self.expire_pending()
            self._say("nothing")
            return self._finish(TurnOutcome("nothing_heard"))
        return self.handle_text(transcript.text)

    # ------------------------------------------------------------------ text turn
    def handle_text(self, text: str) -> TurnOutcome:
        if self.pending is not None:
            return self._answer_pending(text)
        self._status(Status.THINKING)
        try:
            ctx = self.probe.snapshot(self.cwd)
            intent = self.engine.resolve(text, ctx)
        except Exception as exc:
            log.exception("intent resolution failed")
            self.bus.publish(ev.ErrorEvent("intent", str(exc)))
            return self._finish(TurnOutcome("error", text))
        return self._dispatch(intent, ctx)

    def _dispatch(self, intent: Intent, ctx: ContextSnapshot) -> TurnOutcome:
        assessment = self.shield.assess(intent)
        self.bus.publish(ev.IntentResolved(intent, assessment.risk, assessment.reasons))
        kind = intent.kind
        if kind == IntentKind.EXECUTE:
            if assessment.risk == Risk.BLOCKED:
                self.bus.publish(ev.Blocked(intent, assessment.reasons))
                self._say("blocked")
                return self._finish(TurnOutcome("blocked", intent.utterance, intent))
            if assessment.risk == Risk.CONFIRM:
                self.pending = ConfirmationFlow(intent, self.confirm_timeout_s, self._clock)
                self._pending_ctx = ctx
                self._status(Status.CONFIRMING)
                self.bus.publish(
                    ev.ConfirmationRequested(intent, assessment.reasons, self.confirm_timeout_s)
                )
                self._say("confirm")
                return self._finish(TurnOutcome("confirm_requested", intent.utterance, intent))
            return self._execute(intent, ctx)
        if kind == IntentKind.DELEGATE:
            return self._delegate(intent, ctx)
        if kind in (IntentKind.CONFIRM, IntentKind.CANCEL):
            self.bus.publish(
                ev.Info(
                    "nada pendiente de confirmar" if self.lang == "es" else "nothing to confirm"
                )
            )
            return self._finish(TurnOutcome("meta", intent.utterance, intent))
        if kind == IntentKind.META:
            return self._meta(intent, ctx)
        self.bus.publish(ev.Info(f"no reconocido: “{intent.utterance}”"))
        self._say("unknown")
        return self._finish(TurnOutcome("unknown", intent.utterance, intent))

    # ------------------------------------------------------------------ confirmation
    def _answer_pending(self, text: str) -> TurnOutcome:
        flow = self.pending
        assert flow is not None
        ctx = self._pending_ctx or self.probe.snapshot(self.cwd)
        answer = self.engine.rules.parse(text)  # rules only: never ask an LLM to confirm
        decision = flow.feed_intent(answer)
        self.pending = None
        self._pending_ctx = None
        if decision == Decision.ACCEPTED:
            self.bus.publish(ev.ConfirmationResolved(True, "confirmado"))
            return self._execute(flow.intent, ctx)
        reason = "tiempo agotado" if decision == Decision.EXPIRED else f"respuesta: “{text}”"
        self.bus.publish(ev.ConfirmationResolved(False, reason))
        self._say("cancelled")
        return self._finish(
            TurnOutcome(
                "expired" if decision == Decision.EXPIRED else "cancelled", text, flow.intent
            )
        )

    def expire_pending(self) -> TurnOutcome:
        flow = self.pending
        self.pending = None
        self._pending_ctx = None
        if flow is None:
            return self._finish(TurnOutcome("nothing_heard"))
        flow.expire()
        self.bus.publish(ev.ConfirmationResolved(False, "tiempo agotado"))
        self._say("cancelled")
        return self._finish(TurnOutcome("expired", "", flow.intent))

    # ------------------------------------------------------------------ actions
    def _execute(self, intent: Intent, ctx: ContextSnapshot) -> TurnOutcome:
        if self.dry_run:
            self.bus.publish(ev.Info(f"dry-run: {' '.join(intent.argv)}"))
            self.last_executed = intent
            return self._finish(TurnOutcome("dry_run", intent.utterance, intent))
        if intent.name in PHRASES.get(self.lang, {}):
            self._say(intent.name)
        self._status(Status.EXECUTING)
        self.bus.publish(ev.ActionStarted(intent))
        result = self.runner.run(
            intent.argv, cwd=ctx.root, timeout=self.command_timeout_s, detached=intent.detached
        )
        self.bus.publish(ev.ActionFinished(intent, result))
        self.last_executed = intent
        self.probe.invalidate()
        if not result.detached:
            self._say("done" if result.ok else "failed")
        elif not result.ok:
            self._say("failed")
        return self._finish(
            TurnOutcome("executed" if result.ok else "failed", intent.utterance, intent, result)
        )

    def _delegate(self, intent: Intent, ctx: ContextSnapshot) -> TurnOutcome:
        prompt = intent.prompt or intent.utterance
        if self.dry_run:
            self.bus.publish(ev.Delegated("print", prompt, True, "dry-run: no se envió"))
            return self._finish(TurnOutcome("dry_run", intent.utterance, intent))
        self._status(Status.EXECUTING)
        res = self.delegator.deliver(prompt, ctx.root)
        self.bus.publish(ev.Delegated(res.target, prompt, res.ok, res.detail))
        if res.ok:
            self._say(
                "copied"
                if res.target == "clipboard"
                else "sent"
                if res.target == "claude"
                else "done"
            )
        else:
            self._say("failed")
        return self._finish(
            TurnOutcome(
                "delegated" if res.ok else "failed", intent.utterance, intent, delegation=res
            )
        )

    def _meta(self, intent: Intent, ctx: ContextSnapshot) -> TurnOutcome:
        action = intent.meta
        if action == "repeat":
            if self.last_executed is None:
                self.bus.publish(ev.Info("nada que repetir"))
                return self._finish(TurnOutcome("meta", intent.utterance, intent))
            return self._dispatch(self.last_executed, ctx)  # re-assessed by the shield
        if action == "history":
            done = [
                o.intent.display
                for o in self.history
                if o.intent is not None and o.status == "executed"
            ]
            self.bus.publish(ev.Info("historial: " + (" | ".join(done[-5:]) or "vacío")))
        elif action == "mute":
            self._say("muted")
            self.speaker.muted = True
        elif action == "unmute":
            self.speaker.muted = False
            self._say("unmuted")
        elif action == "help":
            self.bus.publish(ev.Info("\n".join(help_lines(self.lang))))
        elif action == "quit":
            self._say("bye")
            self.stop_requested = True
            return self._finish(TurnOutcome("quit", intent.utterance, intent))
        elif action == "say":
            self.bus.publish(ev.Info(intent.label))
        return self._finish(TurnOutcome("meta", intent.utterance, intent))

    # ------------------------------------------------------------------ main loop
    def run(self, trigger: Any, stop: threading.Event | None = None, poll_s: float = 0.2) -> None:
        stop = stop or threading.Event()
        trigger.start()
        self._say("ready")
        self._status(Status.IDLE)
        try:
            while not stop.is_set() and not self.stop_requested:
                if self.pending is not None:
                    if self.pending.check_timeout() == Decision.EXPIRED:
                        self.expire_pending()
                        continue
                    wait_idle = getattr(self.speaker, "wait_idle", None)
                    if callable(wait_idle):
                        wait_idle(3.0)  # don't record our own "¿Confirmas?"
                    self.listen_once(auto=True)
                    continue
                if not trigger.wait_activation(poll_s):
                    continue
                self.listen_once(trigger=trigger)
        finally:
            trigger.stop()
