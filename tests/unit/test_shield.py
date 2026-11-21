from __future__ import annotations

import pytest

from tests.helpers import FakeClock
from voice_forge.actions.confirm import ConfirmationFlow, Decision
from voice_forge.actions.shield import Shield, program_name
from voice_forge.core.types import Intent, IntentKind, Risk
from voice_forge.intent.rules import RuleParser

S, C, B = Risk.SAFE, Risk.CONFIRM, Risk.BLOCKED


@pytest.mark.parametrize(
    ("argv", "source", "risk"),
    [
        # safe everyday commands
        ("git status --short --branch", "rules", S),
        ("git push", "rules", S),
        ("git push -u origin feature/x", "rules", S),
        ("pytest -q", "rules", S),
        ("git commit -m fix", "rules", S),
        ("git restore --staged a.py", "rules", S),
        ("swarm-pilot dashboard", "alias", S),
        # git destructive
        ("git push -f", "rules", C),
        ("git push --force", "rules", C),
        ("git push --force-with-lease", "rules", C),
        ("git push origin +main", "rules", C),
        ("git push origin :old-branch", "rules", C),
        ("git push --delete origin x", "rules", C),
        ("git -C /tmp push -f", "rules", C),
        ("git reset --hard HEAD~1", "rules", C),
        ("git clean -fdx", "rules", C),
        ("git checkout -- .", "rules", C),
        ("git restore .", "rules", C),
        ("git branch -D feature", "rules", C),
        ("git stash drop", "rules", C),
        ("git stash clear", "rules", C),
        ("git rebase -i HEAD~3", "rules", C),
        ("git commit --amend", "rules", C),
        ("git switch --discard-changes main", "rules", C),
        # filesystem
        ("rm -rf build", "rules", C),
        ("rm -r -f build", "rules", C),
        ("rm -fr build", "rules", C),
        ("rm file.txt", "alias", C),
        ("/bin/rm -R x", "alias", C),
        ("Remove-Item build -Recurse -Force", "alias", C),
        ("del /s /q build", "alias", C),
        ("rmdir /s build", "alias", C),
        ("chmod -R 777 .", "alias", C),
        ("dd if=/dev/zero of=/dev/sda", "alias", C),
        # privilege / power / packages / infra
        ("sudo apt install x", "alias", C),
        ("sudo rm -rf /", "alias", C),
        ("shutdown /s", "alias", C),
        ("npm publish", "alias", C),
        ("twine upload dist/*", "alias", C),
        ("docker system prune -a", "alias", C),
        ("kubectl delete pod x", "alias", C),
        ("psql -c DROP TABLE users", "alias", C),
        # never
        ("mkfs.ext4 /dev/sda1", "alias", B),
        ("mkfs /dev/sda1", "alias", B),
        ("format C:", "alias", B),
        ("diskpart", "alias", B),
        ("", "rules", B),
    ],
)
def test_shield_table(argv: str, source: str, risk: Risk) -> None:
    intent = Intent(IntentKind.EXECUTE, argv=tuple(argv.split()), source=source)  # type: ignore[arg-type]
    assert Shield().assess(intent).risk == risk, Shield().assess(intent)


def test_extra_blocked_patterns() -> None:
    shield = Shield(extra_blocked_patterns=[r"terraform destroy"])
    intent = Intent(IntentKind.EXECUTE, argv=("terraform", "destroy"), source="alias")
    assert shield.assess(intent).risk == B


def test_llm_commands_always_need_confirmation() -> None:
    for argv in (("git", "status"), ("ls",), ("pytest",)):
        a = Shield().assess(Intent(IntentKind.EXECUTE, argv=argv, source="llm"))
        assert a.risk == C and "IA" in a.reasons[0]


@pytest.mark.parametrize(
    "argv",
    [
        ("ls", ";", "rm", "-rf", "/"),
        ("echo", "hi", "&&", "reboot"),
        ("cat", "x", "|", "sh"),
        ("echo", "$(whoami)"),
        ("echo", "`id`"),
        ("echo", "x", ">", "/etc/passwd"),
    ],
)
def test_shell_metacharacters_from_llm_or_alias_are_blocked(argv: tuple[str, ...]) -> None:
    for source in ("llm", "alias"):
        assert Shield().assess(Intent(IntentKind.EXECUTE, argv=argv, source=source)).risk == B  # type: ignore[arg-type]


def test_rule_commit_message_with_metacharacters_is_not_blocked() -> None:
    # argv from rules never goes through a shell, so "a | b" in a message is harmless
    intent = Intent(IntentKind.EXECUTE, argv=("git", "commit", "-m", "Fix a | b"), source="rules")
    assert Shield().assess(intent).risk == S


def test_inline_interpreters_need_confirmation() -> None:
    for argv in (("python", "-c", "print(1)"), ("bash", "-c", "ls"), ("powershell", "-Command", "ls"),
                 ("node", "-e", "1")):  # fmt: skip
        assert Shield().assess(Intent(IntentKind.EXECUTE, argv=argv, source="alias")).risk == C


def test_alias_flags() -> None:
    confirm = Intent(IntentKind.EXECUTE, argv=("make", "deploy"), source="alias", confirm=True)
    assert Shield().assess(confirm).risk == C
    shell = Intent(IntentKind.EXECUTE, argv=("sh", "-c", "a && b"), source="alias", shell=True)
    assert Shield().assess(shell).risk == C  # shell aliases: confirm, not blocked


def test_extra_patterns_and_non_execute() -> None:
    shield = Shield(extra_confirm_patterns=[r"\bdeploy\b"])
    assert shield.assess(Intent(IntentKind.EXECUTE, argv=("make", "deploy"))).risk == C
    assert shield.assess(Intent(IntentKind.DELEGATE, prompt="rm -rf /")).risk == S


def test_rules_to_shield_end_to_end() -> None:
    parser = RuleParser()
    shield = Shield()
    assert shield.assess(parser.parse("haz push forzado")).risk == C
    assert shield.assess(parser.parse("descarta los cambios")).risk == C
    assert shield.assess(parser.parse("estado de git")).risk == S
    assert shield.assess(parser.parse("abre cursor")).risk == S


def test_program_name() -> None:
    assert program_name(r"C:\Program Files\Git\bin\git.EXE") == "git"
    assert program_name("/usr/bin/npm.cmd") == "npm"


# --------------------------------------------------------------------------- confirmation


def make_flow(timeout: float = 10.0) -> tuple[ConfirmationFlow, FakeClock]:
    clock = FakeClock()
    return ConfirmationFlow(
        Intent(IntentKind.EXECUTE, argv=("git", "push", "-f")), timeout, clock
    ), clock


def test_confirmation_accepts_isolated_word() -> None:
    p = RuleParser()
    flow, _ = make_flow()
    assert flow.feed_intent(p.parse("confirmo")) == Decision.ACCEPTED


@pytest.mark.parametrize(
    "answer", ["no", "sí pero espera", "cancela", "corre los tests", "", "confirmas"]
)
def test_anything_else_cancels(answer: str) -> None:
    flow, _ = make_flow()
    assert flow.feed_intent(RuleParser().parse(answer)) == Decision.CANCELLED


def test_timeout_expires_even_if_confirmed_late() -> None:
    flow, clock = make_flow(10)
    assert flow.remaining_s == 10
    clock.advance(10.1)
    assert flow.check_timeout() == Decision.EXPIRED
    assert flow.feed_intent(Intent(IntentKind.CONFIRM)) == Decision.EXPIRED
    assert flow.remaining_s == 0


def test_keys_and_manual_expire() -> None:
    flow, _ = make_flow()
    assert flow.feed_key("Y") == Decision.ACCEPTED
    flow2, _ = make_flow()
    assert flow2.feed_key("x") == Decision.CANCELLED
    flow3, _ = make_flow()
    assert flow3.expire() == Decision.EXPIRED
    assert flow3.feed_key("y") == Decision.EXPIRED
