"""Text normalization for matching spoken commands.

``normalize`` is lossy (lower case, no accents, no punctuation, no filler
words). ``recover_original`` maps a normalized fragment back to the original
text so free text (commit messages, prompts) keeps its accents and casing.
"""

from __future__ import annotations

import re
import unicodedata

# Common Whisper mishearings of technical words (applied word by word).
MISHEARINGS: dict[str, str] = {
    "guit": "git",
    "guitar": "git",
    "gith": "git",
    "jit": "git",
    "gitt": "git",
    "comit": "commit",
    "commits": "commits",
    "comitea": "commit",
    "pul": "pull",
    "puch": "push",
    "pytes": "pytest",
    "pitest": "pytest",
    "testes": "tests",
    "texts": "tests",
}
# Regex word fixes for words Whisper mangles in many ways ("acamit", "camid", "komit"...).
WORD_FIXES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(?:a|ha)?[ck][aoue]m+[eiy][td]+e?$"), "commit"),
)
MULTI_MISHEARINGS: tuple[tuple[str, str], ...] = (
    (r"\bpy test\b", "pytest"),
    (r"\bpi test\b", "pytest"),
    (r"\bpai test\b", "pytest"),
    (r"\bswarm pilot\b", "swarm pilot"),
)

_LEADING_FILLERS = (
    "eh", "ehh", "em", "emm", "mm", "este", "pues", "bueno", "o sea", "oye", "ok", "okay", "vale",
    "a ver", "um", "uh", "hey", "so", "well", "voice forge", "vf",
)  # fmt: skip
_TRAILING_FILLERS = ("por favor", "porfa", "please", "gracias", "thanks", "thank you", "ya", "ok")
_ANYWHERE = re.compile(r"\b(?:por favor|porfavor|porfa|please)\b")

_LEAD_RE = re.compile(
    r"^(?:" + "|".join(sorted(map(re.escape, _LEADING_FILLERS), key=len, reverse=True)) + r")\s+"
)
_TRAIL_RE = re.compile(
    r"\s+(?:" + "|".join(sorted(map(re.escape, _TRAILING_FILLERS), key=len, reverse=True)) + r")$"
)

NUMBER_WORDS = {
    "un": 1, "uno": 1, "una": 1, "one": 1, "dos": 2, "two": 2, "tres": 3, "three": 3,
    "cuatro": 4, "four": 4, "cinco": 5, "five": 5, "seis": 6, "six": 6, "siete": 7, "seven": 7,
    "ocho": 8, "eight": 8, "nueve": 9, "nine": 9, "diez": 10, "ten": 10, "quince": 15,
    "fifteen": 15, "veinte": 20, "twenty": 20, "treinta": 30, "thirty": 30,
}  # fmt: skip


def _fix_word(word: str) -> str:
    if word in MISHEARINGS:
        return MISHEARINGS[word]
    for pattern, repl in WORD_FIXES:
        if pattern.match(word):
            return repl
    return word


def fold(text: str) -> str:
    """Lower-case and strip accents WITHOUT changing the string length."""
    out = []
    for ch in text:
        low = ch.lower()
        base = unicodedata.normalize("NFKD", low)[:1] if low else ch
        out.append(base or ch)
    return "".join(out)


def _strip_fillers(s: str) -> str:
    s = _ANYWHERE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    changed = True
    while changed:
        changed = False
        for rx in (_LEAD_RE, _TRAIL_RE):
            new = rx.sub("", s).strip()
            if new and new != s:
                s, changed = new, True
    return s


def normalize(text: str) -> str:
    s = fold(text)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"_", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = " ".join(_fix_word(w) for w in s.split())
    for pattern, repl in MULTI_MISHEARINGS:
        s = re.sub(pattern, repl, s)
    return _strip_fillers(s)


def to_int(token: str | None, default: int) -> int:
    if not token:
        return default
    if token.isdigit():
        return int(token)
    return NUMBER_WORDS.get(token, default)


def recover_original(original: str, fragment: str) -> str:
    """Find ``fragment`` (normalized words) inside ``original`` and return that span."""
    words = fragment.split()
    if not words:
        return ""
    folded = fold(original)
    pattern = r"[\W_]*".join(re.escape(w) for w in words)
    match = re.search(pattern, folded)
    if match is None:
        return fragment
    return original[match.start() : match.end()].strip()
