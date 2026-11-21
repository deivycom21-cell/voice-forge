from __future__ import annotations

import json
import subprocess
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from tests.helpers import FakeClock
from voice_forge.core.types import ContextSnapshot, IntentKind
from voice_forge.intent.engine import IntentEngine, build_prompt, clean_request, intent_from_llm
from voice_forge.intent.llm import (
    ClaudeCliBackend,
    FakeLLM,
    LLMBackend,
    LLMError,
    OllamaBackend,
    extract_json,
    is_loopback_url,
)
from voice_forge.intent.rules import RuleParser

# --------------------------------------------------------------------------- local fake Ollama server


class Handler(BaseHTTPRequestHandler):
    mode = "ok"
    requests: list[dict[str, Any]] = []

    def log_message(self, *args: Any) -> None:
        pass

    def _send(self, code: int, body: str) -> None:
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self._send(200, json.dumps({"models": []}))

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        Handler.requests.append(json.loads(self.rfile.read(length)))
        if Handler.mode == "ok":
            self._send(200, json.dumps({"message": {"content": '{"kind": "unknown"}'}}))
        elif Handler.mode == "garbage":
            self._send(200, "not json at all")
        elif Handler.mode == "nomessage":
            self._send(200, json.dumps({"done": True}))
        elif Handler.mode == "slow":
            time.sleep(1.5)
            self._send(200, "{}")
        else:
            self._send(500, "{}")


@pytest.fixture
def ollama() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    Handler.mode = "ok"
    Handler.requests = []
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_loopback_validation() -> None:
    assert is_loopback_url("http://127.0.0.1:11434") and is_loopback_url("http://localhost:1")
    assert is_loopback_url("http://[::1]:11434")
    assert not is_loopback_url("http://10.0.0.5:11434") and not is_loopback_url(
        "https://api.example.com"
    )
    with pytest.raises(LLMError, match="not local"):
        OllamaBackend("http://192.168.1.10:11434")
    OllamaBackend("http://192.168.1.10:11434", allow_remote=True)


def test_ollama_complete_sends_json_mode(ollama: str) -> None:
    backend = OllamaBackend(ollama, "tiny-model")
    assert isinstance(backend, LLMBackend)
    assert backend.available()
    assert backend.complete("SYS", "USER", timeout=5) == '{"kind": "unknown"}'
    req = Handler.requests[-1]
    assert req["model"] == "tiny-model" and req["format"] == "json" and req["stream"] is False
    assert req["messages"][0] == {"role": "system", "content": "SYS"}
    assert req["options"]["temperature"] == 0


@pytest.mark.parametrize("mode", ["garbage", "nomessage", "error"])
def test_ollama_bad_responses_raise_llm_error(ollama: str, mode: str) -> None:
    Handler.mode = mode
    with pytest.raises(LLMError):
        OllamaBackend(ollama).complete("s", "u", timeout=5)


def test_ollama_timeout(ollama: str) -> None:
    Handler.mode = "slow"
    with pytest.raises(LLMError):
        OllamaBackend(ollama).complete("s", "u", timeout=0.3)


def test_ollama_unavailable_is_cached() -> None:
    clock = FakeClock()
    backend = OllamaBackend("http://127.0.0.1:9", clock=clock, availability_ttl_s=30)
    assert backend.available() is False
    backend._avail = (clock(), True)  # cached value is reused within TTL
    assert backend.available() is True
    clock.advance(31)
    assert backend.available() is False


def test_claude_cli_backend() -> None:
    seen: dict[str, Any] = {}

    def fake_run(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        seen.update(kw, argv=argv)
        return subprocess.CompletedProcess(
            argv, 0, json.dumps({"result": '{"kind":"unknown"}'}), ""
        )

    b = ClaudeCliBackend("haiku", which=lambda x: "/bin/claude", run=fake_run)
    assert b.available()
    assert b.complete("SYS", "USER", timeout=3) == '{"kind":"unknown"}'
    assert seen["argv"] == ["/bin/claude", "-p", "--output-format", "json", "--model", "haiku"]
    assert "SYS" in seen["input"] and "USER" in seen["input"] and seen["timeout"] == 3


def test_claude_cli_backend_errors() -> None:
    assert not ClaudeCliBackend(which=lambda x: None).available()
    with pytest.raises(LLMError):
        ClaudeCliBackend(which=lambda x: None).complete("s", "u", timeout=1)
    failing = ClaudeCliBackend(
        which=lambda x: "c", run=lambda a, **k: subprocess.CompletedProcess(a, 1, "", "no auth")
    )
    with pytest.raises(LLMError, match="no auth"):
        failing.complete("s", "u", timeout=1)

    def timeout(a: Any, **k: Any) -> Any:
        raise subprocess.TimeoutExpired(a, 1)

    with pytest.raises(LLMError):
        ClaudeCliBackend(which=lambda x: "c", run=timeout).complete("s", "u", timeout=1)
    plain = ClaudeCliBackend(
        which=lambda x: "c", run=lambda a, **k: subprocess.CompletedProcess(a, 0, "texto", "")
    )
    assert plain.complete("s", "u", timeout=1) == "texto"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": {"b": 2}}\n```', {"a": {"b": 2}}),
        ('Sure! {"kind": "execute"} hope it helps', {"kind": "execute"}),
        ('{broken} {"ok": true}', {"ok": True}),
    ],
)
def test_extract_json(text: str, expected: dict[str, Any]) -> None:
    assert extract_json(text) == expected


@pytest.mark.parametrize("text", ["", "no json", "[1, 2]", "{unclosed"])
def test_extract_json_errors(text: str) -> None:
    with pytest.raises(LLMError):
        extract_json(text)


# --------------------------------------------------------------------------- engine


def test_rules_win_and_llm_is_never_called(py_ctx: ContextSnapshot) -> None:
    llm = FakeLLM(['{"kind": "execute", "argv": ["rm", "-rf", "/"]}'])
    intent = IntentEngine(RuleParser(), llm).resolve("corre los tests", py_ctx)
    assert intent.argv == ("pytest", "-q") and llm.calls == []


def test_unknown_goes_to_llm_and_is_tagged(py_ctx: ContextSnapshot) -> None:
    llm = FakeLLM(['{"kind": "execute", "argv": ["docker", "ps"], "label": "contenedores"}'])
    intent = IntentEngine(RuleParser(), llm).resolve("qué contenedores están corriendo", py_ctx)
    assert intent.kind == IntentKind.EXECUTE and intent.argv == ("docker", "ps")
    assert intent.source == "llm" and intent.label == "contenedores"
    system, user = llm.calls[0]
    assert "JSON" in system and "branch=main" in user and "qué contenedores" in user


@pytest.mark.parametrize(
    "response",
    ['{"kind": "execute", "argv": "ls -la"}', '{"kind": "execute", "argv": []}', '{"kind": "dance"}',
     "not json", LLMError("down"), RuntimeError("bug")],
)  # fmt: skip
def test_bad_llm_output_degrades_to_unknown(response: Any, py_ctx: ContextSnapshot) -> None:
    intent = IntentEngine(RuleParser(), FakeLLM([response])).resolve("haz magia", py_ctx)
    assert intent.kind == IntentKind.UNKNOWN


def test_unavailable_llm_is_not_called(py_ctx: ContextSnapshot) -> None:
    llm = FakeLLM(['{"kind":"execute","argv":["x"]}'], available=False)
    assert IntentEngine(RuleParser(), llm).resolve("haz magia", py_ctx).kind == IntentKind.UNKNOWN
    assert llm.calls == []

    class Exploding(FakeLLM):
        def available(self) -> bool:
            raise RuntimeError("boom")

    assert (
        IntentEngine(RuleParser(), Exploding()).resolve("haz magia", py_ctx).kind
        == IntentKind.UNKNOWN
    )


def test_delegation_without_llm_is_structured_deterministically(py_ctx: ContextSnapshot) -> None:
    intent = IntentEngine(RuleParser()).resolve("Claude, agrega validación al config", py_ctx)
    assert intent.kind == IntentKind.DELEGATE
    prompt = intent.prompt or ""
    assert prompt.startswith("## Tarea\nAgrega validación al config.")
    assert "- Rama: main" in prompt and "voice_forge/cli.py" in prompt
    assert "## Instrucciones" in prompt and "python (pyproject.toml)" in prompt


def test_delegation_with_llm_rewrite(py_ctx: ContextSnapshot) -> None:
    llm = FakeLLM(
        ['{"prompt": "Añade validación de esquema a Config con mensajes de error claros"}']
    )
    intent = IntentEngine(RuleParser(), llm).resolve(
        "claude eh agrega como validacion al config", py_ctx
    )
    assert "Añade validación de esquema a Config con mensajes de error claros." in (
        intent.prompt or ""
    )
    assert len(llm.calls) == 1 and "Spoken request" in llm.calls[0][1]


def test_delegation_llm_failure_falls_back(py_ctx: ContextSnapshot) -> None:
    intent = IntentEngine(RuleParser(), FakeLLM([LLMError("x")])).resolve(
        "claude arregla el bug", py_ctx
    )
    assert "Arregla el bug." in (intent.prompt or "")


def test_llm_delegate_is_not_rewritten_twice(py_ctx: ContextSnapshot) -> None:
    llm = FakeLLM(['{"kind": "delegate", "prompt": "Explica el módulo de audio"}'])
    intent = IntentEngine(RuleParser(), llm, lang="en").resolve("y eso del audio como va", py_ctx)
    assert intent.kind == IntentKind.DELEGATE and intent.source == "llm"
    assert len(llm.calls) == 1
    assert (intent.prompt or "").startswith("## Task\nExplica el módulo de audio.")


def test_intent_from_llm_validation() -> None:
    assert intent_from_llm({"kind": "execute", "argv": ["a", ""]}, "t") is None
    assert intent_from_llm({"kind": "execute", "argv": ["a"] * 40}, "t") is None
    assert intent_from_llm({"kind": "execute", "argv": ["a", 1]}, "t") is None
    d = intent_from_llm({"kind": "delegate", "prompt": ""}, "texto original")
    assert d is not None and d.prompt == "texto original"


def test_build_prompt_variants(tmp_path: Any) -> None:
    assert clean_request("  hola mundo ") == "Hola mundo."
    assert clean_request("¿qué pasa?") == "¿qué pasa?"
    assert clean_request("") == ""
    plain = build_prompt("x", None)
    assert "## Contexto" not in plain
    no_git = ContextSnapshot(cwd=tmp_path, recent_files=("a.py", "b.py"))
    assert "Archivos recientes: a.py, b.py" in build_prompt("x", no_git)
    many = ContextSnapshot(
        cwd=tmp_path, git_root=tmp_path, changed_files=tuple(f"f{i}" for i in range(20))
    )
    assert "(+5)" in build_prompt("x", many)
    assert "ninguno" in build_prompt("x", ContextSnapshot(cwd=tmp_path, git_root=tmp_path))
