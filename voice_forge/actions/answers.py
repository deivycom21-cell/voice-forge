"""Answers from ``claude -p`` to spoken questions.

* :func:`save_last_answer` / :func:`load_last_answer` keep the most recent
  answer on disk so ``voice-forge last`` can show it again.
* :func:`spoken_summary` turns a (markdown) answer into one short sentence for TTS.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from voice_forge.config import APP_NAME


@dataclass(frozen=True)
class AnswerRecord:
    question: str
    answer: str
    cwd: str = ""
    created_at: str = ""  # ISO 8601, UTC

    @classmethod
    def now(cls, question: str, answer: str, cwd: str = "") -> AnswerRecord:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return cls(question=question, answer=answer, cwd=cwd, created_at=stamp)


def last_answer_path() -> Path:
    from platformdirs import user_data_dir

    return Path(user_data_dir(APP_NAME, appauthor=False)) / "last_answer.json"


def save_last_answer(record: AnswerRecord, path: Path | None = None) -> Path:
    """Write atomically (temp file + replace) so a crash never leaves a broken file."""
    target = path or last_answer_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".last_answer.", suffix=".json", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(asdict(record), fh, ensure_ascii=False, indent=2)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


def load_last_answer(path: Path | None = None) -> AnswerRecord | None:
    """Return the last saved answer, or ``None`` if missing or unreadable."""
    target = path or last_answer_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        return AnswerRecord(
            question=str(data["question"]),
            answer=str(data["answer"]),
            cwd=str(data.get("cwd", "")),
            created_at=str(data.get("created_at", "")),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


# --------------------------------------------------------------------------- spoken summary

_CODE_BLOCK = re.compile(r"```.*?(?:```|$)", re.S)
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_LINE_PREFIX = re.compile(r"^\s*(?:#{1,6}\s+|>\s*|[-*+]\s+|\d+[.)]\s+)")
_EMPHASIS = re.compile(r"(\*\*|__|\*|_|`)")
_SENTENCE = re.compile(r"^(.+?[.!?…])(?=\s|$)")


def plain_text(markdown: str) -> str:
    """Very small markdown -> plain text conversion, good enough for speech."""
    text = _CODE_BLOCK.sub(" ", markdown)
    text = _LINK.sub(r"\1", text)
    lines = [_LINE_PREFIX.sub("", line) for line in text.splitlines()]
    text = " ".join(line.strip() for line in lines if line.strip())
    text = _EMPHASIS.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def spoken_summary(answer: str, max_chars: int = 160) -> str:
    """First sentence of ``answer`` as plain text, cut at a word boundary."""
    text = plain_text(answer)
    if not text:
        return ""
    match = _SENTENCE.match(text)
    sentence = match.group(1) if match else text
    if len(sentence) <= max_chars:
        return sentence
    cut = sentence[:max_chars].rsplit(" ", 1)[0].rstrip(",;:")
    return cut + "…"
