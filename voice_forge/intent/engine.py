"""Hybrid intent engine: deterministic rules first, optional LLM as fallback.

Guarantees:
  * a rule match never calls the LLM;
  * LLM failures/timeouts degrade to deterministic behaviour (never block);
  * LLM-produced commands are tagged ``source="llm"`` (shield => >= CONFIRM).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from voice_forge.core.types import ContextSnapshot, Intent, IntentKind
from voice_forge.intent.classify import classify_request, clean_question
from voice_forge.intent.llm import LLMBackend, LLMError, extract_json
from voice_forge.intent.rules import RuleParser

log = logging.getLogger(__name__)

CLASSIFY_SYSTEM = """You turn a developer's spoken request (Spanish or English) into ONE JSON object.
Reply with JSON only, no prose. Schema:
{"kind": "execute" | "delegate" | "unknown",
 "argv": ["program", "arg", ...],   // only for execute: a single command, NO shell syntax (no pipes, &&, ;, >)
 "label": "short description",
 "prompt": "clean technical prompt"} // only for delegate
Use "execute" only for a single short terminal command. Use "delegate" when the request needs
an AI coding agent (code changes, explanations, refactors). Use "unknown" if unsure."""

PROMPT_SYSTEM = """You rewrite a developer's spoken, messy request into a clear, technical prompt for an
AI coding agent. Keep the user's intent and language. Be concise and specific; do not invent requirements.
Reply with JSON only: {"prompt": "..."}"""

TEMPLATES = {
    "es": {
        "task": "## Tarea",
        "context": "## Contexto",
        "dir": "Directorio",
        "branch": "Rama",
        "changed": "Archivos modificados",
        "recent": "Archivos recientes",
        "project": "Proyecto",
        "rules": "## Instrucciones",
        "rules_body": (
            "- Haz cambios mínimos y enfocados en la tarea.\n"
            "- Ejecuta los tests relevantes antes de terminar.\n"
            "- Al final, resume qué cambiaste y por qué."
        ),
        "none": "ninguno",
    },
    "en": {
        "task": "## Task",
        "context": "## Context",
        "dir": "Directory",
        "branch": "Branch",
        "changed": "Modified files",
        "recent": "Recent files",
        "project": "Project",
        "rules": "## Instructions",
        "rules_body": (
            "- Keep changes minimal and focused on the task.\n"
            "- Run the relevant tests before finishing.\n"
            "- At the end, summarize what you changed and why."
        ),
        "none": "none",
    },
}


def _fmt_list(items: tuple[str, ...], none: str, limit: int = 15) -> str:
    if not items:
        return none
    shown = ", ".join(items[:limit])
    return shown + (f" (+{len(items) - limit})" if len(items) > limit else "")


def clean_request(text: str) -> str:
    text = " ".join(text.split()).strip(" ,.;:")
    if not text:
        return text
    text = text[:1].upper() + text[1:]
    return text if text.endswith((".", "?", "!")) else text + "."


def build_prompt(request: str, ctx: ContextSnapshot | None, lang: str = "es") -> str:
    """Deterministic structured prompt (no LLM needed)."""
    t = TEMPLATES.get(lang, TEMPLATES["es"])
    lines = [t["task"], clean_request(request), ""]
    if ctx is not None:
        lines += [t["context"], f"- {t['dir']}: {ctx.root}"]
        if ctx.branch:
            lines.append(f"- {t['branch']}: {ctx.branch}")
        if ctx.is_git:
            lines.append(f"- {t['changed']}: {_fmt_list(ctx.changed_files, t['none'])}")
        elif ctx.recent_files:
            lines.append(f"- {t['recent']}: {_fmt_list(ctx.recent_files, t['none'], 8)}")
        if ctx.project_kind != "unknown" or ctx.markers:
            lines.append(f"- {t['project']}: {ctx.project_kind} ({', '.join(ctx.markers) or '-'})")
        lines.append("")
    lines += [t["rules"], t["rules_body"]]
    return "\n".join(lines)


def describe_context(ctx: ContextSnapshot | None) -> str:
    if ctx is None:
        return "(no context)"
    return (
        f"cwd={ctx.root}; branch={ctx.branch or '-'}; project={ctx.project_kind}; "
        f"changed={', '.join(ctx.changed_files[:10]) or '-'}; tools={', '.join(sorted(ctx.tools)) or '-'}"
    )


class IntentEngine:
    def __init__(
        self,
        rules: RuleParser,
        llm: LLMBackend | None = None,
        *,
        llm_timeout_s: float = 20.0,
        lang: str = "es",
    ) -> None:
        self.rules = rules
        self.llm = llm
        self.llm_timeout_s = llm_timeout_s
        self.lang = lang

    def _llm_ready(self) -> bool:
        if self.llm is None:
            return False
        try:
            return bool(self.llm.available())
        except Exception:
            return False

    def resolve(self, text: str, ctx: ContextSnapshot | None = None) -> Intent:
        intent = self.rules.parse(text, ctx)
        if intent.kind == IntentKind.UNKNOWN and text.strip() and self._llm_ready():
            intent = self._classify_with_llm(text, ctx) or intent
        if intent.kind == IntentKind.DELEGATE:
            request = intent.prompt or text
            if classify_request(request, text) == "question":
                # Questions go out clean: no file-editing template, no LLM rewrite.
                return replace(
                    intent, prompt=clean_question(request), delegate_kind="question",
                    label="→ pregunta" if self.lang == "es" else "→ question",
                )  # fmt: skip
            intent = replace(intent, delegate_kind="code")
            if intent.source == "llm":  # already rewritten by the LLM: don't call it twice
                prompt = build_prompt(request, ctx, self.lang)
            else:
                prompt = self.structure_prompt(request, ctx)
            intent = replace(intent, prompt=prompt)
        return intent

    def _classify_with_llm(self, text: str, ctx: ContextSnapshot | None) -> Intent | None:
        assert self.llm is not None
        user = f"Context: {describe_context(ctx)}\nRequest: {text}"
        try:
            data = extract_json(
                self.llm.complete(CLASSIFY_SYSTEM, user, timeout=self.llm_timeout_s)
            )
        except (LLMError, Exception) as exc:  # never let the LLM break the loop
            log.info("llm classification failed: %s", exc)
            return None
        return intent_from_llm(data, text)

    def structure_prompt(self, request: str, ctx: ContextSnapshot | None) -> str:
        improved = request
        if self._llm_ready():
            assert self.llm is not None
            user = f"Context: {describe_context(ctx)}\nSpoken request: {request}"
            try:
                data = extract_json(
                    self.llm.complete(PROMPT_SYSTEM, user, timeout=self.llm_timeout_s)
                )
                candidate = data.get("prompt")
                if isinstance(candidate, str) and candidate.strip():
                    improved = candidate.strip()
            except Exception as exc:
                log.info("llm prompt structuring failed: %s", exc)
        return build_prompt(improved, ctx, self.lang)


def intent_from_llm(data: dict[str, Any], text: str) -> Intent | None:
    """Validate the LLM JSON; return None when it does not match the schema."""
    kind = data.get("kind")
    label = data.get("label") if isinstance(data.get("label"), str) else ""
    if kind == "execute":
        argv = data.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(a, str) and a for a in argv)
            or len(argv) > 32
        ):
            return None
        return Intent(IntentKind.EXECUTE, text, argv=tuple(argv), source="llm", name="llm.execute",
                      label=label or " ".join(argv), confidence=0.7)  # fmt: skip
    if kind == "delegate":
        prompt = data.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            prompt = text
        return Intent(IntentKind.DELEGATE, text, prompt=prompt.strip(), source="llm", name="llm.delegate",
                      label=label or "→ agente", confidence=0.7)  # fmt: skip
    return None
