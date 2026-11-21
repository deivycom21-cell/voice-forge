"""Safety shield: classifies the FINAL argv that would run.

* ``SAFE``     run directly (only for rule/alias commands)
* ``CONFIRM``  explicit voice confirmation required
* ``BLOCKED``  never executed

Invariants (tested):
  - every command coming from an LLM is at least CONFIRM;
  - shell metacharacters in LLM/alias argv are BLOCKED (we never use a shell);
  - the base destructive list cannot be disabled by configuration.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePath

from voice_forge.core.types import Intent, IntentKind, Risk

SHELL_META = re.compile(r"(;|&&|\|\||\||>|<|\$\(|`|\n|\r)")
INTERPRETERS = {
    "sh": {"-c"}, "bash": {"-c"}, "zsh": {"-c"}, "fish": {"-c"}, "dash": {"-c"},
    "cmd": {"/c", "/k"}, "powershell": {"-command", "-c", "-encodedcommand", "-enc", "-e"},
    "pwsh": {"-command", "-c", "-encodedcommand", "-enc", "-e"},
    "python": {"-c"}, "python3": {"-c"}, "py": {"-c"}, "node": {"-e", "--eval"},
    "perl": {"-e"}, "ruby": {"-e"},
}  # fmt: skip
ALWAYS_BLOCKED = {"mkfs", "format", "diskpart", "fdisk", "parted", "wipefs", "shred", "cipher"}
PRIVILEGE = {"sudo", "doas", "su", "runas", "gsudo"}
POWER = {"shutdown", "reboot", "halt", "poweroff", "init"}
JOINED_CONFIRM = (
    (re.compile(r"\bdrop\s+(table|database|schema)\b", re.I), "SQL DROP"),
    (re.compile(r"\btruncate\s+table\b", re.I), "SQL TRUNCATE"),
    (re.compile(r"\bremove-item\b.*-r(ecurse)?\b", re.I), "Remove-Item -Recurse"),
    (re.compile(r"\brm\s+-[a-z]*[rf]", re.I), "rm -r/-f"),
    (re.compile(r"\bgit\s+push\b.*(\s-f\b|--force|\s\+)", re.I), "git push --force"),
    (re.compile(r"\bgit\s+reset\b.*--hard", re.I), "git reset --hard"),
)


@dataclass(frozen=True)
class Assessment:
    risk: Risk
    reasons: tuple[str, ...] = ()

    @property
    def safe(self) -> bool:
        return self.risk == Risk.SAFE


def program_name(arg0: str) -> str:
    name = PurePath(arg0.replace("\\", "/")).name.lower()
    for ext in (".exe", ".cmd", ".bat", ".ps1", ".com"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def _short_flags(args: Iterable[str]) -> set[str]:
    """``-rf`` -> {'r','f'} (long options ignored)."""
    out: set[str] = set()
    for a in args:
        if a.startswith("-") and not a.startswith("--") and len(a) > 1:
            out.update(a[1:])
    return out


def _git_subcommand(args: list[str]) -> tuple[str | None, list[str]]:
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-C", "-c", "--git-dir", "--work-tree", "--namespace"):
            i += 2
            continue
        if a.startswith("-"):
            i += 1
            continue
        return a.lower(), args[i + 1 :]
    return None, []


def _assess_git(rest: list[str]) -> list[str]:
    sub, args = _git_subcommand(rest)
    low = [a.lower() for a in args]
    flags = _short_flags(args)
    reasons: list[str] = []
    if sub == "push":
        if (
            {
                "--force",
                "--force-with-lease",
                "--force-if-includes",
                "--mirror",
                "--delete",
                "--prune",
            }
            & set(low)
            or any(a.startswith(("--force-with-lease=", "--force=")) for a in low)
            or "f" in flags
            or "d" in flags
            or any(a.startswith(("+", ":")) for a in args)
        ):
            reasons.append("git push forzado/destructivo rescribe o borra historial remoto")
    elif sub == "reset" and {"--hard", "--merge", "--keep"} & set(low):
        reasons.append("git reset --hard descarta cambios locales")
    elif sub == "clean" and ("f" in flags or "--force" in low):
        reasons.append("git clean borra archivos no versionados")
    elif sub == "checkout" and ("--" in args or "." in args or "f" in flags or "--force" in low):
        reasons.append("git checkout sobre rutas descarta cambios")
    elif sub == "restore" and "--staged" not in low:
        reasons.append("git restore descarta cambios del working tree")
    elif sub == "branch" and ({"d", "D"} & flags or "--delete" in low):
        reasons.append("borrar ramas")
    elif sub == "stash" and low[:1] in (["drop"], ["clear"]):
        reasons.append("git stash drop/clear pierde cambios guardados")
    elif sub in ("rebase", "filter-branch", "filter-repo"):
        reasons.append(f"git {sub} rescribe historial")
    elif sub == "commit" and "--amend" in low:
        reasons.append("git commit --amend rescribe el último commit")
    elif sub == "update-ref" and "d" in flags:
        reasons.append("git update-ref -d borra referencias")
    elif sub == "reflog" and low[:1] == ["expire"]:
        reasons.append("git reflog expire elimina red de seguridad")
    elif sub == "gc" and any(a.startswith("--prune") for a in low):
        reasons.append("git gc --prune elimina objetos")
    elif sub == "switch" and ("--discard-changes" in low or "f" in flags or "--force" in low):
        reasons.append("git switch --discard-changes descarta cambios")
    return reasons


class Shield:
    def __init__(
        self,
        extra_confirm_patterns: Iterable[str] = (),
        extra_blocked_patterns: Iterable[str] = (),
    ) -> None:
        self.extra_confirm = [re.compile(p, re.I) for p in extra_confirm_patterns]
        self.extra_blocked = [re.compile(p, re.I) for p in extra_blocked_patterns]

    def assess(self, intent: Intent) -> Assessment:
        if intent.kind != IntentKind.EXECUTE:
            return Assessment(Risk.SAFE)
        return self.assess_argv(
            list(intent.argv),
            source=intent.source,
            force_confirm=intent.confirm,
            shell=intent.shell,
        )

    def assess_argv(
        self,
        argv: list[str],
        *,
        source: str = "rules",
        force_confirm: bool = False,
        shell: bool = False,
    ) -> Assessment:
        if not argv or not argv[0].strip():
            return Assessment(Risk.BLOCKED, ("comando vacío",))
        risk = Risk.SAFE
        reasons: list[str] = []

        def bump(level: Risk, reason: str) -> None:
            nonlocal risk
            risk = max(risk, level)
            if reason not in reasons:
                reasons.append(reason)

        joined = " ".join(argv)
        prog = program_name(argv[0])
        rest = argv[1:]
        low_rest = [a.lower() for a in rest]
        flags = _short_flags(rest)

        # ---- origin based
        if source == "llm":
            bump(Risk.CONFIRM, "comando generado por IA")
        if force_confirm:
            bump(Risk.CONFIRM, "alias marcado con confirm = true")
        if shell:
            bump(Risk.CONFIRM, "alias ejecutado a través de una shell")
        elif source != "rules" and any(SHELL_META.search(a) for a in argv):
            bump(Risk.BLOCKED, "metacaracteres de shell (no se usa shell)")

        # ---- interpreters with inline code
        inline = INTERPRETERS.get(prog)
        if inline and source != "rules" and not shell and inline & set(low_rest):
            bump(Risk.CONFIRM, f"{prog} con código en línea")

        # ---- program based
        if prog in ALWAYS_BLOCKED or prog.startswith("mkfs."):
            bump(Risk.BLOCKED, f"{prog}: operación de disco irreversible")
        if prog in PRIVILEGE:
            bump(Risk.CONFIRM, f"{prog}: privilegios elevados")
            if rest:  # also assess the wrapped command
                inner = self.assess_argv(rest, source=source)
                for r in inner.reasons:
                    bump(inner.risk, r)
        if prog in POWER:
            bump(Risk.CONFIRM, f"{prog}: apaga/reinicia el equipo")
        if prog in ("rm", "rmdir", "rd", "del", "erase", "unlink", "remove-item", "ri", "trash"):
            bump(Risk.CONFIRM, f"{prog}: borra archivos")
        if prog == "dd":
            bump(Risk.CONFIRM, "dd escribe bloques en crudo")
        if prog in ("chmod", "chown", "chgrp", "icacls", "takeown") and (
            {"R"} & flags or "/t" in low_rest or "--recursive" in low_rest
        ):
            bump(Risk.CONFIRM, f"{prog} recursivo")
        if prog == "git":
            for r in _assess_git(rest):
                bump(Risk.CONFIRM, r)
        if prog == "docker" and (
            low_rest[:2]
            in (["system", "prune"], ["volume", "rm"], ["volume", "prune"], ["image", "prune"])
            or low_rest[:1] in (["rm"], ["rmi"])
        ):
            bump(Risk.CONFIRM, "docker elimina recursos")
        if prog == "kubectl" and low_rest[:1] == ["delete"]:
            bump(Risk.CONFIRM, "kubectl delete")
        if (
            prog in ("npm", "pnpm", "yarn", "cargo", "twine", "poetry", "gh")
            and "publish" in low_rest[:2]
        ):
            bump(Risk.CONFIRM, "publica un paquete")
        if prog in ("twine",) and "upload" in low_rest[:1]:
            bump(Risk.CONFIRM, "publica un paquete")

        # ---- whole-command patterns (catch wrapped commands, e.g. through shells)
        for pattern, reason in JOINED_CONFIRM:
            if pattern.search(joined):
                bump(Risk.CONFIRM, reason)
        for pattern in self.extra_confirm:
            if pattern.search(joined):
                bump(Risk.CONFIRM, f"patrón configurado: {pattern.pattern}")
        for pattern in self.extra_blocked:
            if pattern.search(joined):
                bump(Risk.BLOCKED, f"patrón bloqueado: {pattern.pattern}")
        return Assessment(risk, tuple(reasons))
