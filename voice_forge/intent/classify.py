"""Deterministic classification of agent requests: code order vs. question.

* ``code``      "crea…", "arregla…", "refactoriza…", "add…", "fix…" -> edit template, delegated.
* ``question``  "explícame…", "qué es…", "resume…", "explain…", "what is…" -> answered by ``claude -p``.

Works on normalized text (lower case, no accents), so it is cheap and offline.
"""

from __future__ import annotations

import re
from typing import Literal

from voice_forge.intent.normalize import normalize

RequestKind = Literal["code", "question"]

# Openers that make an utterance a question/explanation request.
QUESTION_OPENERS = (
    r"explica(?:me|nos|r|s)?|explique(?:s|n)?|expliques"
    r"|que (?:es|son|significa|significan|hace|hacen|pasa|opinas|diferencia|ventajas|desventajas"
    r"|problema|tal|falta|deberia|puedo|sabes|tipo|version)"
    r"|como (?:funciona|funcionan|se|puedo|hago|hacer|esta|estan|deberia|seria)"
    r"|por que|porque|para que (?:sirve|sirven)|cual(?:es)? (?:es|son|seria|serian|fue)"
    r"|cuanto|cuantos|cuantas|donde|quien|quienes"
    r"|resume(?:me|nos)?|resumen|resuma|describe(?:me)?|describa|dime|cuentame|compara(?:me)?"
    r"|compare|analiza|analice|sabes"
    r"|explain|what|how|why|which|who|where|is there|are there|does|do you|can you explain"
    r"|could you explain|summari[sz]e|describe|tell me"
)
QUESTION_RE = re.compile(rf"^(?:{QUESTION_OPENERS})\b")

CODE_VERBS = (  # noqa: SIM905 - a word list is far more readable than a literal
    # imperative · infinitive · subjunctive ("dile a claude que agregue…")
    "crea crear cree creen arregla arreglar arregle corrige corregir corrija refactoriza refactorizar "
    "refactorice agrega agregar agregue anade anadir anada implementa implementar implemente "
    "escribe escribir escriba elimina eliminar elimine borra borrar borre quita quitar quite "
    "cambia cambiar cambie modifica modificar modifique actualiza actualizar actualice renombra "
    "renombrar renombre mueve mover mueva optimiza optimizar optimice migra migrar migre genera "
    "generar genere haz hazme hacer haga pon poner ponga mejora mejorar mejore limpia limpiar "
    "limpie documenta documentar documente sube anota completa termina "
    "create fix refactor add implement write remove delete change modify update rename move "
    "optimize optimise migrate generate make build improve clean document"
).split()
CODE_VERB_SET = frozenset(CODE_VERBS)
# "… y arréglalo": allow enclitic pronouns glued to the verb (arreglalo, cambialos, hazlo)
_THEN_CODE = re.compile(
    r"\b(?:y|and|luego|then|despues|tambien|also) (?:"
    + "|".join(CODE_VERBS)
    + r")(?:lo|la|los|las|le|les|me|nos)?\b"
)
_ASK_VERB = re.compile(r"^(?:oye |hey )?(?:preguntale|pregunta|preguntar|consulta|ask)\b")


def classify_request(request: str, utterance: str = "") -> RequestKind:
    """Decide whether an agent request is a code order or a question."""
    norm = normalize(request)
    words = norm.split()
    if not words:
        return "code"
    if words[0] in CODE_VERB_SET:
        return "code"  # "ask Claude to add type hints" is an order
    if QUESTION_RE.match(norm) or _ASK_VERB.match(normalize(utterance)):
        # "explícame el bug y arréglalo" is still an order
        return "code" if _THEN_CODE.search(norm) else "question"
    if any(w in CODE_VERB_SET for w in words[1:3]):
        return "code"  # "¿puedes arreglar el login?" is a polite order
    if "?" in request or "¿" in request or "?" in utterance:
        return "question"
    return "code"


_IMPERATIVE = re.compile(
    r"^(?:explica\w*|explique\w*|resum\w*|describ\w*|dime|cuentame|compar\w*|analiz\w*"
    r"|explain|summari[sz]e|describe|tell me|compare|list)\b"
)
_SPANISH_WORDS = frozenset(
    [
        "el",
        "la",
        "los",
        "las",
        "de",
        "del",
        "que",
        "es",
        "y",
        "en",
        "un",
        "una",
        "por",
        "para",
        "con",
        "se",
        "lo",
        "como",
    ]
)
_ENGLISH_WORDS = frozenset(
    [
        "the",
        "of",
        "is",
        "and",
        "to",
        "in",
        "a",
        "an",
        "what",
        "how",
        "for",
        "with",
        "it",
        "this",
        "that",
        "does",
    ]
)


def looks_spanish(norm: str) -> bool:
    words = norm.split()
    return sum(w in _SPANISH_WORDS for w in words) >= sum(w in _ENGLISH_WORDS for w in words)


def clean_question(text: str) -> str:
    """Tidy a spoken question: trim, capitalize and punctuate it.

    "qué es un VAD" -> "¿Qué es un VAD?" · "explícame el recorder" -> "Explícame el recorder." ·
    "el VAD funciona bien" (asked via "pregúntale si…") -> "¿El VAD funciona bien?"
    """
    body = " ".join(text.split()).strip(" ,;:").lstrip("¿¡").strip()
    if not body:
        return ""
    body = body[:1].upper() + body[1:]
    norm = normalize(body)
    imperative = bool(_IMPERATIVE.match(norm))
    if not body.endswith(("?", ".", "!", "…")):
        # "¿Qué es un VAD? Responde en dos frases" -> the question is already marked: end with "."
        body += "." if imperative or "?" in body else "?"
    if "?" in body and not imperative and looks_spanish(norm):
        body = "¿" + body
    return body
