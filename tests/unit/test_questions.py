"""Code orders vs. questions: classifier, clean questions, answer storage and spoken summary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voice_forge.actions.answers import (
    AnswerRecord,
    load_last_answer,
    plain_text,
    save_last_answer,
    spoken_summary,
)
from voice_forge.core.types import ContextSnapshot, IntentKind
from voice_forge.intent.classify import classify_request, clean_question
from voice_forge.intent.engine import IntentEngine
from voice_forge.intent.llm import FakeLLM
from voice_forge.intent.rules import RuleParser

# --------------------------------------------------------------------------- classifier


@pytest.mark.parametrize(
    ("request_text", "utterance", "kind"),
    [
        # questions / explanations
        ("explícame el módulo de audio", "", "question"),
        ("qué es un VAD", "", "question"),
        ("cómo funciona el recorder", "", "question"),
        ("por qué falla el test de contexto", "", "question"),
        ("resume los cambios de hoy", "", "question"),
        ("describe la arquitectura", "", "question"),
        ("explique el VAD", "dile a claude que explique el VAD", "question"),
        ("el VAD funciona bien", "pregúntale a Claude si el VAD funciona bien", "question"),
        ("¿hay tests para el escudo?", "", "question"),
        ("what is a VAD", "", "question"),
        ("how does the shield work", "", "question"),
        ("explain the recorder", "", "question"),
        ("summarize the README", "", "question"),
        ("what the recorder does", "ask claude what the recorder does", "question"),
        # code orders
        ("crea un módulo de métricas", "", "code"),
        ("arregla el parser", "", "code"),
        ("refactoriza el parser para usar dataclasses", "", "code"),
        ("agregue validación al config", "dile a claude que agregue validación al config", "code"),
        ("add type hints to the CLI", "ask Claude to add type hints to the CLI", "code"),
        ("fix the failing test", "", "code"),
        ("explícame el bug y arréglalo", "", "code"),
        ("explain the bug and fix it", "", "code"),
        ("puedes arreglar el login?", "", "code"),
        ("necesito que el recorder corte antes", "", "code"),  # default: order
        ("", "", "code"),
    ],
)
def test_classify_request(request_text: str, utterance: str, kind: str) -> None:
    assert classify_request(request_text, utterance) == kind


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("qué es un VAD", "¿Qué es un VAD?"),
        ("¿qué es un VAD?", "¿Qué es un VAD?"),
        ("cómo funciona el recorder", "¿Cómo funciona el recorder?"),
        ("explícame el módulo de audio", "Explícame el módulo de audio."),
        ("resume los cambios", "Resume los cambios."),
        ("el VAD funciona bien", "¿El VAD funciona bien?"),
        ("what is a VAD", "What is a VAD?"),
        ("explain the recorder", "Explain the recorder."),
        ("  qué   pasa,  ", "¿Qué pasa?"),
        ("¿Qué es un VAD? Responde en dos frases", "¿Qué es un VAD? Responde en dos frases."),
        ("", ""),
    ],
)
def test_clean_question(raw: str, clean: str) -> None:
    assert clean_question(raw) == clean


# --------------------------------------------------------------------------- rules + engine


@pytest.mark.parametrize(
    ("text", "prompt"),
    [
        ("Claude, ¿qué es un VAD?", "¿Qué es un VAD?"),
        ("Pregúntale a Claude qué es un VAD", "¿Qué es un VAD?"),
        ("pregúntale a Claude si el VAD funciona bien", "¿El VAD funciona bien?"),
        ("Explícame el módulo de audio", "Explícame el módulo de audio."),
        ("What is a VAD in this project", "What is a VAD in this project?"),
        ("Claude, resume los cambios de hoy", "Resume los cambios de hoy."),
    ],
)
def test_questions_are_sent_clean(text: str, prompt: str, py_ctx: ContextSnapshot) -> None:
    llm = FakeLLM(['{"prompt": "should not be used"}'])
    intent = IntentEngine(RuleParser(), llm).resolve(text, py_ctx)
    assert intent.kind == IntentKind.DELEGATE and intent.delegate_kind == "question"
    assert intent.prompt == prompt
    assert "## Tarea" not in prompt and "## Instrucciones" not in prompt
    assert llm.calls == []  # questions are never rewritten by the LLM


def test_code_orders_keep_the_template(py_ctx: ContextSnapshot) -> None:
    intent = IntentEngine(RuleParser()).resolve("Claude, refactoriza el parser", py_ctx)
    assert intent.delegate_kind == "code"
    prompt = intent.prompt or ""
    assert prompt.startswith("## Tarea\nRefactoriza el parser.")
    assert "- Rama: main" in prompt and "## Instrucciones" in prompt


def test_question_rule_does_not_steal_existing_commands() -> None:
    p = RuleParser()
    assert p.parse("¿qué ha cambiado?").name == "git.status"
    assert p.parse("what can you do").name == "meta.help"
    assert p.parse("¿en qué rama estoy?").name == "git.branch.current"
    assert p.parse("explícame el recorder").name == "delegate.question"
    assert p.parse("dime").kind == IntentKind.UNKNOWN  # opener alone is not a question


def test_engine_english_label(py_ctx: ContextSnapshot) -> None:
    intent = IntentEngine(RuleParser(), lang="en").resolve("what is a VAD", py_ctx)
    assert intent.label == "→ question"


# --------------------------------------------------------------------------- answers store


def test_save_and_load_last_answer(tmp_path: Path) -> None:
    path = tmp_path / "data" / "last.json"
    record = AnswerRecord.now("¿Qué es un VAD?", "Un **VAD** detecta voz.", "/repo")
    assert save_last_answer(record, path) == path
    loaded = load_last_answer(path)
    assert loaded == record
    assert record.created_at.endswith("+00:00")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["question"] == "¿Qué es un VAD?"  # stored as real UTF-8, not \u escapes
    assert "¿" in path.read_text(encoding="utf-8")
    assert list(path.parent.glob(".last_answer.*")) == []  # no temp files left behind


def test_load_missing_or_corrupt(tmp_path: Path) -> None:
    assert load_last_answer(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_last_answer(bad) is None
    bad.write_text('{"answer": "x"}', encoding="utf-8")
    assert load_last_answer(bad) is None


def test_default_path_is_isolated_in_tests() -> None:
    from voice_forge.actions import answers

    save_last_answer(AnswerRecord("q", "a"))
    assert load_last_answer() == AnswerRecord("q", "a")
    assert "userdirs" in str(answers.last_answer_path())  # conftest keeps the real dir untouched


# --------------------------------------------------------------------------- spoken summary


@pytest.mark.parametrize(
    ("answer", "summary"),
    [
        ("Un VAD detecta voz. Se usa para cortar la grabación.", "Un VAD detecta voz."),
        (
            "# VAD\n\nEl **VAD** detecta voz en el audio.\n\n- punto",
            "VAD El VAD detecta voz en el audio.",
        ),
        ("```python\nx = 1\n```\nEsto asigna `x`.", "Esto asigna x."),
        ("Mira [la doc](https://x.y) primero. Luego…", "Mira la doc primero."),
        ("- primero\n- segundo", "primero segundo"),
        ("¿Sabes qué? Funciona.", "¿Sabes qué?"),
        ("", ""),
    ],
)
def test_spoken_summary(answer: str, summary: str) -> None:
    assert spoken_summary(answer) == summary


def test_spoken_summary_truncates_long_sentences_at_word_boundary() -> None:
    long = "palabra " * 60 + "fin."
    out = spoken_summary(long, max_chars=50)
    assert out.endswith("…") and len(out) <= 51
    assert not out[:-1].endswith(" ")


def test_plain_text_strips_markdown() -> None:
    assert plain_text("## Título\n> cita\n1. uno\n__dos__") == "Título cita uno dos"
