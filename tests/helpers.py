"""Test helpers: build a fully faked orchestrator."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from voice_forge.actions.answers import AnswerRecord
from voice_forge.actions.delegate import Delegator
from voice_forge.actions.executor import FakeRunner
from voice_forge.actions.shield import Shield
from voice_forge.audio import synth
from voice_forge.audio.recorder import Recorder
from voice_forge.audio.sources import ArraySource
from voice_forge.config import DEFAULT_ALIASES
from voice_forge.core.events import RecordingBus
from voice_forge.core.orchestrator import Orchestrator
from voice_forge.core.types import ContextSnapshot
from voice_forge.intent.engine import IntentEngine
from voice_forge.intent.llm import LLMBackend
from voice_forge.intent.rules import RuleParser
from voice_forge.stt.transcriber import FakeTranscriber
from voice_forge.tts.speaker import RecordingSpeaker


class FakeProbe:
    def __init__(self, ctx: ContextSnapshot) -> None:
        self.ctx = ctx
        self.invalidations = 0

    def snapshot(self, cwd: Path | None = None) -> ContextSnapshot:
        return self.ctx

    def invalidate(self) -> None:
        self.invalidations += 1


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


def utterance_audio() -> synth.Audio:
    """~1 s of 'speech' surrounded by silence."""
    return synth.concat(synth.silence(400), synth.speechlike(1000, seed=1), synth.silence(1000))


@dataclass
class Rig:
    orch: Orchestrator
    bus: RecordingBus
    runner: FakeRunner
    speaker: RecordingSpeaker
    transcriber: FakeTranscriber
    copied: list[str] = field(default_factory=list)
    clock: FakeClock = field(default_factory=FakeClock)
    answers: list[AnswerRecord] = field(default_factory=list)


def make_rig(
    ctx: ContextSnapshot,
    script: Iterable[str] = (),
    *,
    audio: synth.Audio | None = None,
    llm: LLMBackend | None = None,
    dry_run: bool = False,
    target: str = "clipboard",
    runner: FakeRunner | None = None,
    confirm_timeout_s: float = 10.0,
    answer_with_claude: bool = False,
) -> Rig:
    bus = RecordingBus()
    runner = runner or FakeRunner()
    speaker = RecordingSpeaker()
    transcriber = FakeTranscriber(script)
    copied: list[str] = []
    clock = FakeClock()
    answers: list[AnswerRecord] = []
    signal = utterance_audio() if audio is None else audio
    orch = Orchestrator(
        bus=bus,
        recorder=Recorder(start_timeout_s=2.0),
        transcriber=transcriber,
        probe=FakeProbe(ctx),  # type: ignore[arg-type]
        engine=IntentEngine(RuleParser(DEFAULT_ALIASES), llm),
        shield=Shield(),
        runner=runner,
        delegator=Delegator(
            target,  # type: ignore[arg-type]
            runner,
            copy=copied.append,
            answer_with_claude=answer_with_claude,
        ),
        speaker=speaker,
        source_factory=lambda: ArraySource(signal),
        cwd=ctx.cwd,
        confirm_timeout_s=confirm_timeout_s,
        dry_run=dry_run,
        clock=clock,
        answer_store=answers.append,
    )
    return Rig(orch, bus, runner, speaker, transcriber, copied, clock, answers)
