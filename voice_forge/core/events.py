"""Typed events and a tiny synchronous in-process event bus.

The core publishes events; the HUD, the TTS layer and tests subscribe. The
core never knows who is listening.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from voice_forge.core.types import ActionResult, Intent, Risk, Status, Transcript

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Event:
    """Base class for every event."""


@dataclass(frozen=True)
class StatusChanged(Event):
    status: Status


@dataclass(frozen=True)
class ListeningStarted(Event):
    auto: bool = False  # True when listening without hotkey (e.g. confirmation)


@dataclass(frozen=True)
class LevelUpdate(Event):
    level_db: float


@dataclass(frozen=True)
class SpeechEnded(Event):
    duration_s: float
    reason: str


@dataclass(frozen=True)
class NothingHeard(Event):
    pass


@dataclass(frozen=True)
class Transcribed(Event):
    transcript: Transcript


@dataclass(frozen=True)
class IntentResolved(Event):
    intent: Intent
    risk: Risk = Risk.SAFE
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConfirmationRequested(Event):
    intent: Intent
    reasons: tuple[str, ...]
    timeout_s: float


@dataclass(frozen=True)
class ConfirmationResolved(Event):
    accepted: bool
    reason: str


@dataclass(frozen=True)
class ActionStarted(Event):
    intent: Intent


@dataclass(frozen=True)
class ActionFinished(Event):
    intent: Intent
    result: ActionResult


@dataclass(frozen=True)
class Delegated(Event):
    target: str
    prompt: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class Answered(Event):
    """A question was sent to ``claude -p`` and (maybe) answered."""

    question: str
    answer: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class Blocked(Event):
    intent: Intent
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Info(Event):
    message: str


@dataclass(frozen=True)
class Spoken(Event):
    text: str


@dataclass(frozen=True)
class ErrorEvent(Event):
    where: str
    message: str


E = TypeVar("E", bound=Event)
Handler = Callable[[Any], None]


class EventBus:
    """Synchronous publish/subscribe bus.

    A handler that raises never prevents other handlers from receiving the
    event; the error is logged and kept in :attr:`handler_errors`.
    """

    def __init__(self) -> None:
        self._subs: list[tuple[type[Event], Handler]] = []
        self._lock = threading.Lock()
        self.handler_errors: list[BaseException] = []

    def subscribe(
        self, handler: Callable[[E], None], event_type: type[E] | type[Event] = Event
    ) -> Callable[[], None]:
        """Register ``handler`` for ``event_type`` (and subclasses). Returns an unsubscribe fn."""
        entry: tuple[type[Event], Handler] = (event_type, handler)
        with self._lock:
            self._subs.append(entry)

        def unsubscribe() -> None:
            with self._lock:
                if entry in self._subs:
                    self._subs.remove(entry)

        return unsubscribe

    def publish(self, event: Event) -> None:
        with self._lock:
            subs = list(self._subs)
        for event_type, handler in subs:
            if isinstance(event, event_type):
                try:
                    handler(event)
                except Exception as exc:  # isolation: one bad subscriber must not break others
                    log.exception("event handler failed for %s", type(event).__name__)
                    self.handler_errors.append(exc)


class RecordingBus(EventBus):
    """EventBus that records every published event (handy for tests)."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[Event] = []

    def publish(self, event: Event) -> None:
        self.events.append(event)
        super().publish(event)

    def of_type(self, event_type: type[E]) -> list[E]:
        return [e for e in self.events if isinstance(e, event_type)]

    def types(self) -> list[str]:
        return [type(e).__name__ for e in self.events]
