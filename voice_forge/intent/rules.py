"""Deterministic, bilingual (ES/EN) grammar: normalized text -> :class:`Intent`.

No network, no LLM, no disk: designed to resolve in well under 1 ms.

Resolution order:
  1. user aliases (exact)
  2. regex rules (full match on the normalized text)
  3. fuzzy match against rule/alias example phrases (short utterances only)
  4. long utterance -> implicit delegation
  5. UNKNOWN
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from voice_forge.config import AliasSpec
from voice_forge.core.types import ContextSnapshot, Intent, IntentKind
from voice_forge.intent.normalize import normalize, recover_original, to_int

Builder = Callable[["Match"], Intent]

CLAUDE = r"(?:claude|cloud|clod|claud|clode|klaud)"
RUN = r"(?:corre|correr|corra|ejecuta|ejecutar|lanza|lanzar|pasa|pasar|haz|hace|run|launch|execute)"
ARTS = r"(?:(?:los|las|el|la|todos los|todas las|the|all the|all) )?"
TESTS = r"(?:tests?|pruebas?|test suite|suite de tests|unit tests|testeos?)"


@dataclass(frozen=True)
class Match:
    """What a builder receives."""

    groups: Mapping[str, str | None]
    text: str  # original utterance
    norm: str
    ctx: ContextSnapshot | None

    def get(self, *names: str) -> str | None:
        for n in names:
            v = self.groups.get(n)
            if v:
                return v
        return None

    def original(self, fragment: str) -> str:
        return recover_original(self.text, fragment)


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    build: Builder
    examples: tuple[str, ...] = ()


# --------------------------------------------------------------------------- helpers


def cmd(name: str, *argv: str, label: str | None = None, detached: bool = False) -> Builder:
    def build(m: Match) -> Intent:
        return Intent(
            IntentKind.EXECUTE, m.text, argv=tuple(argv), name=name,
            label=label or " ".join(argv), detached=detached,
        )  # fmt: skip

    return build


def meta(name: str, action: str, label: str = "") -> Builder:
    def build(m: Match) -> Intent:
        return Intent(IntentKind.META, m.text, meta=action, name=name, label=label or action)

    return build


def say(m: Match, name: str, text: str) -> Intent:
    return Intent(IntentKind.META, m.text, meta="say", name=name, label=text)


def branch_slug(spoken: str) -> str:
    """``"feature barra login con google"`` -> ``"feature/login-con-google"``."""
    s = f" {spoken.strip()} "
    for words, sep in (
        (("guion bajo", "underscore"), "_"),
        (("barra", "slash", "diagonal"), "/"),
        (("guion", "dash", "hyphen"), "-"),
        (("punto", "dot"), "."),
    ):
        for w in words:
            s = s.replace(f" {w} ", f" {sep} ")
    tokens = s.split()
    out = ""
    for tok in tokens:
        if tok in "_/-.":
            out = out.rstrip("-") + tok
        else:
            out += ("" if not out or out[-1] in "_/-." else "-") + tok
    return re.sub(r"[^a-z0-9/_.\-]", "", out).strip("-/.") or "nueva-rama"


def build_test_argv(
    ctx: ContextSnapshot | None, *, failed: bool = False, target: str | None = None
) -> tuple[str, ...]:
    kind = ctx.project_kind if ctx else "python"
    if kind == "node":
        return ("npm", "test", *(("--", target) if target else ()))
    if kind == "rust":
        return ("cargo", "test", *((target,) if target else ()))
    if kind == "go":
        return ("go", "test", "./...", *(("-run", target) if target else ()))
    base: tuple[str, ...] = (
        ("pytest",) if ctx is None or ctx.has_tool("pytest") else ("python", "-m", "pytest")
    )
    extra: tuple[str, ...] = ("-q",)
    if failed:
        extra += ("--lf",)
    if target:
        extra += ("-k", " and ".join(target.split()))
    return base + extra


def in_new_terminal(
    argv: tuple[str, ...],
    platform: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> tuple[str, ...] | None:
    """Wrap ``argv`` so it opens in a new terminal window (argv only, no shell string)."""
    platform = platform or sys.platform
    if platform == "win32":
        if which("wt"):
            return ("wt", "-d", ".", *argv)
        return ("cmd", "/c", "start", "", *argv)
    if platform == "darwin":
        return ("open", "-a", "Terminal", ".") if argv == ("claude",) else None
    for term, prefix in (
        ("x-terminal-emulator", ("-e",)),
        ("gnome-terminal", ("--",)),
        ("konsole", ("-e",)),
        ("kitty", ()),
        ("alacritty", ("-e",)),
        ("xterm", ("-e",)),
    ):
        if which(term):
            return (term, *prefix, *argv)
    return None


# --------------------------------------------------------------------------- builders


def _git_log(m: Match) -> Intent:
    n = to_int(m.get("n", "n2"), 10)
    return Intent(IntentKind.EXECUTE, m.text, argv=("git", "log", "--oneline", "-n", str(n)),
                  name="git.log", label=f"git log ({n})")  # fmt: skip


def _git_switch(m: Match) -> Intent:
    name = branch_slug(m.get("name", "name2", "name3", "name4") or "")
    return Intent(IntentKind.EXECUTE, m.text, argv=("git", "switch", name), name="git.switch",
                  label=f"git switch {name}")  # fmt: skip


def _git_branch_create(m: Match) -> Intent:
    name = branch_slug(m.get("name", "name2") or "")
    return Intent(IntentKind.EXECUTE, m.text, argv=("git", "switch", "-c", name),
                  name="git.branch.create", label=f"git switch -c {name}")  # fmt: skip


def _git_commit(m: Match) -> Intent:
    raw = m.get("msg", "msg2") or ""
    if raw in ("mensaje", "con mensaje", "message", "with message"):
        return _git_commit_nomsg(m)
    msg = m.original(raw).strip(" \"'«»“”.")
    msg = msg[:1].upper() + msg[1:]
    return Intent(IntentKind.EXECUTE, m.text, argv=("git", "commit", "-m", msg), name="git.commit",
                  label=f'git commit -m "{msg}"')  # fmt: skip


def _git_commit_nomsg(m: Match) -> Intent:
    return say(
        m, "git.commit.nomsg", "Di: commit con mensaje, y el mensaje / Say: commit with message ..."
    )


def _git_push(m: Match) -> Intent:
    ctx = m.ctx
    if ctx and ctx.branch and ctx.is_git and not ctx.upstream and not ctx.detached_head:
        argv: tuple[str, ...] = ("git", "push", "-u", "origin", ctx.branch)
    else:
        argv = ("git", "push")
    return Intent(IntentKind.EXECUTE, m.text, argv=argv, name="git.push", label=" ".join(argv))


def _tests(m: Match) -> Intent:
    argv = build_test_argv(m.ctx)
    return Intent(IntentKind.EXECUTE, m.text, argv=argv, name="tests.run", label=" ".join(argv))


def _tests_failed(m: Match) -> Intent:
    argv = build_test_argv(m.ctx, failed=True)
    return Intent(IntentKind.EXECUTE, m.text, argv=argv, name="tests.failed", label=" ".join(argv))


def _tests_target(m: Match) -> Intent:
    target = (m.get("target") or "").strip()
    argv = build_test_argv(m.ctx, target=target)
    return Intent(IntentKind.EXECUTE, m.text, argv=argv, name="tests.target", label=" ".join(argv))


def _lint(m: Match) -> Intent:
    kind = m.ctx.project_kind if m.ctx else "python"
    if kind == "node":
        argv: tuple[str, ...] = ("npm", "run", "lint")
    elif kind == "rust":
        argv = ("cargo", "clippy")
    elif m.ctx is None or m.ctx.has_tool("ruff"):
        argv = ("ruff", "check", ".")
    else:
        argv = ("python", "-m", "ruff", "check", ".")
    return Intent(IntentKind.EXECUTE, m.text, argv=argv, name="lint", label=" ".join(argv))


def _open_claude(m: Match) -> Intent:
    wrapped = in_new_terminal(("claude",))
    if wrapped is None:
        return say(
            m, "open.claude", "No encontré un emulador de terminal / No terminal emulator found"
        )
    return Intent(IntentKind.EXECUTE, m.text, argv=wrapped, name="open.claude",
                  label="claude (nueva terminal)", detached=True)  # fmt: skip


def _open_editor(m: Match) -> Intent:
    editor = m.get("editor") or "code"
    exe = "cursor" if editor == "cursor" else "code"
    return Intent(IntentKind.EXECUTE, m.text, argv=(exe, "."), name="open.editor",
                  label=f"{exe} .", detached=True)  # fmt: skip


def _delegate(m: Match) -> Intent:
    req = m.original(m.get("req", "req2", "req3", "req4") or m.norm)
    return Intent(IntentKind.DELEGATE, m.text, prompt=req, name="delegate", label="→ agente")


# --------------------------------------------------------------------------- grammar


def R(*alternatives: str) -> re.Pattern[str]:
    return re.compile("(?:" + "|".join(alternatives) + ")")


RULES: tuple[Rule, ...] = (
    # ---- control
    Rule("confirm", R(r"(?:si )?(?:si|sii|confirmo|confirmado|confirma|dale|hazlo|adelante|procede)",
                      r"yes|yeah|yep|confirm|confirmed|do it|go ahead|proceed"),
         lambda m: Intent(IntentKind.CONFIRM, m.text, name="confirm"), ("confirmo", "yes confirm")),
    Rule("cancel", R(r"no|nop|nope|no no|cancela|cancelar|cancelalo|cancelado|para|parar|detente|alto",
                     r"olvidalo|dejalo|nada|no gracias|stop|cancel|abort|never ?mind|forget it"),
         lambda m: Intent(IntentKind.CANCEL, m.text, name="cancel"), ("cancela", "cancel")),
    Rule("meta.repeat", R(r"repite|repitelo|repite eso|otra vez|de nuevo|again|repeat|repeat that",
                          r"do it again|run it again"),
         meta("meta.repeat", "repeat"), ("repite", "repeat that")),
    Rule("meta.history", R(r"(?:muestra |ver |abre )?(?:el )?historial|(?:show )?(?:the )?history"),
         meta("meta.history", "history"), ("historial", "show history")),
    Rule("meta.mute", R(r"silencio|callate|modo silencio|sin voz|mute|be quiet|quiet|silence"),
         meta("meta.mute", "mute"), ("modo silencio", "mute")),
    Rule("meta.unmute", R(r"habla|activa la voz|con voz|unmute|voice on|speak"),
         meta("meta.unmute", "unmute"), ("activa la voz", "voice on")),
    Rule("meta.help", R(r"ayuda|help|que puedes hacer|que sabes hacer|comandos|what can you do|commands"),
         meta("meta.help", "help"), ("ayuda", "what can you do")),
    Rule("meta.quit", R(r"salir|terminar|apagate|adios|chao|cierra voice forge|quit|exit|goodbye|bye",
                        r"stop listening|deja de escuchar"),
         meta("meta.quit", "quit"), ("salir", "quit")),
    # ---- git
    Rule("git.status", R(r"(?:ver |mira |muestra |muestrame |dame |cual es )?(?:el )?estado(?: de| del)?(?: git| repo| repositorio)?",
                         r"git status|status(?: de git)?|que ha cambiado|que cambio|que cambie",
                         r"(?:show |what is |whats )?(?:me )?(?:the )?(?:git |repo )?status"),
         cmd("git.status", "git", "status", "--short", "--branch", label="git status"),
         ("estado de git", "git status", "estado del repo", "show git status")),
    Rule("git.diff", R(r"(?:ver |mira |muestra |muestrame )?(?:el |los |las )?(?:diff|cambios|diferencias)(?: de git| del repo| sin commitear)?",
                       r"git diff|(?:show )?(?:me )?(?:the )?(?:diff|changes)"),
         cmd("git.diff", "git", "diff", "--stat", label="git diff --stat"),
         ("muestra el diff", "git diff", "show the diff")),
    Rule("git.log", R(r"(?:ver |mira |muestra |muestrame )?(?:el )?(?:log|historial de (?:git|commits))",
                      r"git log|(?:los )?ultimos (?P<n>\w+) commits|(?:show )?(?:the )?(?:git )?log",
                      r"(?:show )?(?:me )?(?:the )?last (?P<n2>\w+) commits"),
         _git_log, ("muestra el log", "git log", "show the log")),
    Rule("git.branch.current", R(r"(?:en )?(?:que|cual) rama(?: estoy| es esta| es)?|rama actual",
                                 r"(?:what|which) branch(?: am i on| is this)?|current branch"),
         cmd("git.branch.current", "git", "branch", "--show-current", label="rama actual"),
         ("en que rama estoy", "what branch am i on")),
    Rule("git.branch.list", R(r"(?:lista|listar|muestra|muestrame|ver)(?: las| todas las)? ramas|ramas|git branch",
                              r"(?:list|show)(?: all)?(?: the)? branches|branches"),
         cmd("git.branch.list", "git", "branch", "--list", label="git branch"),
         ("lista las ramas", "list branches")),
    Rule("git.branch.create", R(r"(?:crea|crear|haz|nueva)(?: una)?(?: nueva)? rama(?: llamada| que se llame| con nombre)? (?P<name>.+)",
                                r"(?:create|make|new)(?: a)?(?: new)? branch(?: called| named)? (?P<name2>.+)"),
         _git_branch_create),
    Rule("git.switch", R(r"(?:cambia(?:te)?|cambiar|pasa(?:te)?|muevete|ve|vete|vamos) a (?:la )?rama (?P<name>.+)",
                         r"(?:switch|change|checkout|go)(?: to)?(?: the)? branch (?P<name2>.+)",
                         r"(?:switch|checkout) to (?P<name3>.+)|git (?:checkout|switch) (?P<name4>.+)"),
         _git_switch),
    Rule("git.add.all", R(r"(?:agrega|agregar|anade|anadir|mete) (?:todo|todos los cambios|todos los archivos|los cambios)(?: a git| al stage| al index)?",
                          r"git add(?: todo| all| a)?|(?:add|stage) (?:all|everything|all changes|all files)"),
         cmd("git.add.all", "git", "add", "-A", label="git add -A"),
         ("agrega todo", "add all changes")),
    Rule("git.commit", R(r"(?:(?:make|do) (?:a )?)?commit(?: with(?: the)? message| saying) (?P<msg2>.+)",
                         r"(?:(?:haz|hace|hacer|crea|crear) )?(?:un )?commit(?: con(?: el)? mensaje| diciendo| que diga| con| de)? (?P<msg>.+)"),
         _git_commit),
    Rule("git.commit.nomsg", R(r"(?:(?:haz|hace|hacer|crea|crear) )?(?:un )?commit|commit"),
         _git_commit_nomsg),
    Rule("git.pull", R(r"(?:(?:haz|hace|hacer) )?(?:un )?(?:git )?pull|trae(?:me)? (?:los )?(?:ultimos )?cambios",
                       r"baja(?:me)? (?:los )?cambios|actualiza(?: la rama| el repo| el repositorio)",
                       r"pull(?: the)?(?: latest)?(?: changes)?|get the latest(?: changes)?"),
         cmd("git.pull", "git", "pull", "--ff-only", label="git pull --ff-only"),
         ("haz pull", "trae los cambios", "pull the latest changes")),
    Rule("git.push.force", R(r"(?:(?:haz|hace) )?(?:un )?(?:git )?push (?:forzado|force|con force|a la fuerza|forzoso|force with lease)",
                             r"(?:fuerza|forza) (?:el )?push|force push|sube (?:los cambios )?(?:a la fuerza|forzado)"),
         cmd("git.push.force", "git", "push", "--force-with-lease", label="git push --force-with-lease"),
         ("push forzado", "force push")),
    Rule("git.push", R(r"(?:(?:haz|hace|hacer) )?(?:un )?(?:git )?push|sube(?:me)? (?:los )?cambios",
                       r"empuja (?:los )?cambios|sube (?:la )?rama|push(?: the)? (?:changes|branch|it)"),
         _git_push, ("haz push", "sube los cambios", "push the changes")),
    Rule("git.stash.pop", R(r"(?:recupera|saca|aplica|restaura)(?: el)? stash|(?:git )?stash pop|pop(?: the)? stash"),
         cmd("git.stash.pop", "git", "stash", "pop"), ("recupera el stash", "stash pop")),
    Rule("git.stash", R(r"(?:guarda|guardar)(?: los)? cambios(?: en el stash)?|(?:(?:haz|hace) )?(?:un )?(?:git )?stash",
                        r"stash(?: changes| my changes)?"),
         cmd("git.stash", "git", "stash", "push"), ("haz stash", "stash my changes")),
    Rule("git.reset.hard", R(r"(?:(?:haz|hace) )?(?:un )?(?:git )?reset(?: --)? ?(?:hard|duro)",
                             r"(?:descarta|borra)(?: todos)?(?: los)? cambios|discard(?: all)?(?: my)?(?: the)? changes"),
         cmd("git.reset.hard", "git", "reset", "--hard", label="git reset --hard"),
         ("reset hard", "descarta los cambios", "discard all changes")),
    Rule("git.fetch", R(r"(?:(?:haz|hace) )?(?:un )?(?:git )?fetch|trae(?: las)? referencias"),
         cmd("git.fetch", "git", "fetch", "--all", "--prune"), ("haz fetch", "git fetch")),
    # ---- tests / quality
    Rule("tests.failed", R(rf"{RUN} {ARTS}{TESTS} (?:que )?(?:fallidos|fallidas|fallaron|fallan|failed|failing)",
                           r"(?:rerun|re run|repite)(?: the| los)? (?:failed|failing|fallidos)(?: tests| pruebas)?",
                           rf"{RUN} (?:the |los )?(?:failed|failing) {TESTS}"),
         _tests_failed, ("corre los tests fallidos", "rerun failed tests")),
    Rule("tests.target", R(rf"{RUN} {ARTS}{TESTS} (?:de|del|para|sobre|of|for|in) (?:el |la |los |the )?(?P<target>.+)"),
         _tests_target),
    Rule("tests.run", R(rf"{RUN} {ARTS}{TESTS}", rf"{ARTS}{TESTS}", r"pytest|npm test|cargo test|go test"),
         _tests, ("corre los tests", "ejecuta las pruebas", "run the tests", "run tests")),
    Rule("lint", R(rf"{RUN} {ARTS}(?:linter|lint|ruff|eslint)|lint|linter|linting"),
         _lint, ("corre el linter", "run the linter")),
    # ---- tools
    Rule("open.claude", R(rf"(?:abre|abrir|abreme|lanza|lanzar|inicia|iniciar|arranca|open|launch|start)(?: a)? {CLAUDE}(?: code)?(?: aqui| here)?"),
         _open_claude, ("abre claude", "open claude")),
    Rule("open.editor", R(r"(?:abre|abrir|abreme|lanza|open|launch)(?: el proyecto| esto| this| the project| el repo)?(?: en| in)? (?P<editor>cursor|vs code|vscode|code|visual studio code)(?: aqui| here)?"),
         _open_editor),
    # ---- delegation
    Rule("delegate", R(rf"(?:oye |hey )?{CLAUDE}(?: code)? (?P<req>.+)",
                       rf"(?:dile|pidele|preguntale|encargale|manda|envia|pregunta)(?: a| al)? (?:{CLAUDE}|agente|el agente)(?: code)? (?:que |a |si )?(?P<req2>.+)",
                       r"(?:agente|prompt|tarea para el agente)(?: code)? (?P<req3>.+)",
                       rf"(?:ask|tell) (?:{CLAUDE}|the agent)(?: code)? (?:to )?(?P<req4>.+)"),
         _delegate),
)  # fmt: skip


class RuleParser:
    def __init__(
        self,
        aliases: Mapping[str, AliasSpec] | None = None,
        *,
        fuzzy_threshold: float = 0.82,
        delegate_min_words: int = 12,
        fuzzy_max_words: int = 7,
    ) -> None:
        self.fuzzy_threshold = fuzzy_threshold
        self.delegate_min_words = delegate_min_words
        self.fuzzy_max_words = fuzzy_max_words
        self.aliases: dict[str, AliasSpec] = {normalize(k): v for k, v in (aliases or {}).items()}
        self._by_name = {r.name: r for r in RULES}
        examples = [(normalize(ex), r.name) for r in RULES for ex in r.examples]
        examples += [(k, f"alias:{k}") for k in self.aliases]
        self._matchers: list[tuple[str, str, SequenceMatcher[str]]] = []
        for example, target in examples:
            sm: SequenceMatcher[str] = SequenceMatcher(None, autojunk=False)
            sm.set_seq2(example)
            self._matchers.append((example, target, sm))

    # ---- alias
    def _alias_intent(
        self, phrase: str, spec: AliasSpec, text: str, confidence: float = 1.0
    ) -> Intent:
        if spec.shell:
            argv: tuple[str, ...] = (
                ("cmd", "/c", spec.cmd) if os.name == "nt" else ("sh", "-c", spec.cmd)
            )
        else:
            argv = tuple(shlex.split(spec.cmd, posix=True))
        return Intent(
            IntentKind.EXECUTE, text, argv=argv, source="alias", name=f"alias:{phrase}",
            label=spec.cmd, detached=spec.detached, confirm=spec.confirm, shell=spec.shell,
            confidence=confidence,
        )  # fmt: skip

    # ---- main entry point
    def parse(self, text: str, ctx: ContextSnapshot | None = None) -> Intent:
        norm = normalize(text)
        if not norm:
            return Intent(IntentKind.UNKNOWN, text)
        if norm in self.aliases:
            return self._alias_intent(norm, self.aliases[norm], text)
        for rule in RULES:
            mt = rule.pattern.fullmatch(norm)
            if mt is not None:
                return rule.build(Match(mt.groupdict(), text, norm, ctx))
        n_words = len(norm.split())
        if n_words <= self.fuzzy_max_words:
            fuzzy = self._fuzzy(norm, text, ctx)
            if fuzzy is not None:
                return fuzzy
        if n_words >= self.delegate_min_words:
            intent = _delegate(Match({}, text, norm, ctx))
            return replace(intent, confidence=0.6, name="delegate.implicit")
        return Intent(IntentKind.UNKNOWN, text)

    def _fuzzy(self, norm: str, text: str, ctx: ContextSnapshot | None) -> Intent | None:
        best_score, best = 0.0, None
        n = len(norm)
        for example, target, sm in self._matchers:
            # ratio <= 2*min(len)/(len_a+len_b): cheap length prefilter
            m = len(example)
            if 2.0 * min(n, m) / (n + m) < self.fuzzy_threshold:
                continue
            sm.set_seq1(norm)
            if sm.quick_ratio() < self.fuzzy_threshold:
                continue
            score = sm.ratio()
            if score > best_score:
                best_score, best = score, (example, target)
        if best is None or best_score < self.fuzzy_threshold:
            return None
        example, target = best
        if target.startswith("alias:"):
            phrase = target[len("alias:") :]
            return self._alias_intent(phrase, self.aliases[phrase], text, best_score)
        rule = self._by_name[target]
        intent = rule.build(Match({}, text, example, ctx))
        if intent.kind in (IntentKind.CONFIRM,):
            return None  # never fuzzy-confirm a destructive action
        return replace(intent, confidence=best_score)


def help_lines(lang: str = "es") -> list[str]:
    if lang == "en":
        return [
            '"git status" · "show the diff" · "last 5 commits" · "create branch feature login"',
            '"commit with message fix parser" · "push the changes" · "pull the latest"',
            '"run the tests" · "rerun failed tests" · "run the linter" · "open claude" · "open cursor"',
            '"claude, <request>" → structured prompt · "repeat" · "history" · "mute" · "quit"',
        ]
    return [
        '"estado de git" · "muestra el diff" · "últimos cinco commits" · "crea rama feature login"',
        '"commit con mensaje arregla el parser" · "sube los cambios" · "trae los cambios"',
        '"corre los tests" · "corre los tests fallidos" · "corre el linter" · "abre claude" · "abre cursor"',
        '"claude, <petición>" → prompt estructurado · "repite" · "historial" · "silencio" · "salir"',
    ]
