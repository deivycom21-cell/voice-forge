"""Confirmation flow for risky actions.

Only an *isolated* confirmation word accepts (the rule parser's ``confirm``
rule is a full match, so "sí pero espera" is not a confirmation). Anything
else, or the timeout, cancels.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import Enum

from voice_forge.core.types import Intent, IntentKind

ACCEPT_KEYS = {"y", "s", "yes", "si", "sí"}


class Decision(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class ConfirmationFlow:
    def __init__(
        self, intent: Intent, timeout_s: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.intent = intent
        self.timeout_s = timeout_s
        self._clock = clock
        self.deadline = clock() + timeout_s
        self.decision = Decision.PENDING

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.deadline - self._clock())

    def check_timeout(self) -> Decision:
        if self.decision == Decision.PENDING and self._clock() >= self.deadline:
            self.decision = Decision.EXPIRED
        return self.decision

    def _resolve(self, accepted: bool) -> Decision:
        if self.check_timeout() != Decision.PENDING:
            return self.decision
        self.decision = Decision.ACCEPTED if accepted else Decision.CANCELLED
        return self.decision

    def feed_intent(self, answer: Intent) -> Decision:
        return self._resolve(answer.kind == IntentKind.CONFIRM)

    def feed_key(self, key: str) -> Decision:
        return self._resolve(key.strip().lower() in ACCEPT_KEYS)

    def expire(self) -> Decision:
        if self.decision == Decision.PENDING:
            self.decision = Decision.EXPIRED
        return self.decision
