"""Wire real components from a :class:`Config` (used by the CLI)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voice_forge.actions.answers import AnswerRecord, save_last_answer
from voice_forge.actions.delegate import Delegator, Target
from voice_forge.actions.executor import CommandRunner, SubprocessRunner
from voice_forge.actions.shield import Shield
from voice_forge.audio.recorder import Recorder
from voice_forge.audio.sources import AudioSource, MicSource, WavFileSource
from voice_forge.audio.vad import EnergyVAD
from voice_forge.config import Config
from voice_forge.context.probe import ContextProbe
from voice_forge.core import events as ev
from voice_forge.core.orchestrator import Orchestrator
from voice_forge.intent.engine import IntentEngine
from voice_forge.intent.llm import ClaudeCliBackend, LLMBackend, OllamaBackend
from voice_forge.intent.rules import RuleParser
from voice_forge.stt.transcriber import Transcriber, WhisperTranscriber
from voice_forge.tts.speaker import NullSpeaker, Speaker, build_speaker


@dataclass
class Overrides:
    model: str | None = None
    language: str | None = None
    target: Target | None = None
    llm: str | None = None
    no_tts: bool = False
    dry_run: bool = False
    from_wav: Path | None = None


def apply_overrides(cfg: Config, o: Overrides) -> Config:
    if o.model:
        cfg.stt.model = o.model
    if o.language:
        cfg.stt.language = o.language  # type: ignore[assignment]
    if o.target:
        cfg.delegate.target = o.target
    if o.llm:
        cfg.llm.backend = o.llm  # type: ignore[assignment]
    if o.no_tts:
        cfg.tts.enabled = False
    return cfg


def build_llm(cfg: Config) -> LLMBackend | None:
    if cfg.llm.backend == "ollama":
        return OllamaBackend(
            cfg.llm.ollama_url, cfg.llm.ollama_model, allow_remote=cfg.llm.allow_remote_ollama
        )
    if cfg.llm.backend == "claude":
        return ClaudeCliBackend(cfg.llm.claude_model)
    return None


def build_recorder(cfg: Config, bus: ev.EventBus) -> Recorder:
    a = cfg.audio
    calibration_frames = max(1, round(a.calibration_ms / a.block_ms))

    def vad() -> EnergyVAD:
        return EnergyVAD(
            a.abs_threshold_db,
            a.margin_db,
            calibration_frames,
            max_noise_floor_db=a.max_noise_floor_db,
        )

    return Recorder(
        vad,
        block_ms=a.block_ms,
        preroll_ms=a.preroll_ms,
        silence_ms=a.silence_ms,
        min_speech_ms=a.min_speech_ms,
        max_seconds=a.max_seconds,
        start_timeout_s=a.start_timeout_s,
        on_level=lambda db: bus.publish(ev.LevelUpdate(db)),
    )


def build_source_factory(cfg: Config, from_wav: Path | None = None) -> Callable[[], AudioSource]:
    if from_wav is not None:
        return lambda: WavFileSource(from_wav, block_ms=cfg.audio.block_ms)
    device: str | int | None = cfg.audio.device or None
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    return lambda: MicSource(device=device, block_ms=cfg.audio.block_ms)


def build_transcriber(cfg: Config) -> WhisperTranscriber:
    s = cfg.stt
    return WhisperTranscriber(
        s.model, compute_type=s.compute_type, cpu_threads=s.cpu_threads, beam_size=s.beam_size,
        vocabulary=s.vocabulary, language=s.language,
    )  # fmt: skip


def build_orchestrator(
    cfg: Config,
    *,
    bus: ev.EventBus,
    cwd: Path | None = None,
    dry_run: bool = False,
    from_wav: Path | None = None,
    transcriber: Transcriber | None = None,
    speaker: Speaker | None = None,
    runner: CommandRunner | None = None,
    probe: ContextProbe | None = None,
    answer_store: Callable[[AnswerRecord], Any] | None = save_last_answer,
) -> Orchestrator:
    runner = runner or SubprocessRunner()
    lang = cfg.intent.ui_language
    rules = RuleParser(cfg.aliases, fuzzy_threshold=cfg.intent.fuzzy_threshold,
                       delegate_min_words=cfg.intent.delegate_min_words)  # fmt: skip
    engine = IntentEngine(rules, build_llm(cfg), llm_timeout_s=cfg.llm.timeout_s, lang=lang)
    if speaker is None:
        t = cfg.tts
        speaker = build_speaker(enabled=t.enabled, engine=t.engine, lang=lang, rate=t.rate, voice=t.voice,
                                piper_model=t.piper_model) if t.enabled else NullSpeaker()  # fmt: skip
    return Orchestrator(
        bus=bus,
        recorder=build_recorder(cfg, bus),
        transcriber=transcriber or build_transcriber(cfg),
        probe=probe or ContextProbe(),
        engine=engine,
        shield=Shield(cfg.safety.extra_confirm_patterns, cfg.safety.extra_blocked_patterns),
        runner=runner,
        delegator=Delegator(
            cfg.delegate.target,
            runner,
            claude_args=cfg.delegate.claude_args,
            claude_timeout_s=cfg.delegate.claude_timeout_s,
            # questions are answered by claude -p with --target claude OR --llm claude
            answer_with_claude=cfg.llm.backend == "claude",
        ),
        speaker=speaker,
        source_factory=build_source_factory(cfg, from_wav),
        cwd=cwd,
        lang=lang,
        stt_language=None if cfg.stt.language == "auto" else cfg.stt.language,
        confirm_timeout_s=cfg.safety.confirm_timeout_s,
        command_timeout_s=cfg.safety.command_timeout_s,
        dry_run=dry_run,
        answer_store=answer_store,
    )
