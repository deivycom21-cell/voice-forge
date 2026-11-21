"""Optional LLM backends. Text only — audio never reaches them.

* :class:`OllamaBackend` — local Ollama over HTTP (loopback only by default).
* :class:`ClaudeCliBackend` — ``claude -p`` (bring your own key; opt-in).
* :class:`FakeLLM` — scripted responses for tests.
"""

from __future__ import annotations

import ipaddress
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse


class LLMError(RuntimeError):
    pass


@runtime_checkable
class LLMBackend(Protocol):
    name: str

    def available(self) -> bool: ...

    def complete(self, system: str, user: str, *, timeout: float) -> str: ...


def is_loopback_url(url: str) -> bool:
    host = urlparse(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class OllamaBackend:
    name = "ollama"

    def __init__(
        self,
        url: str = "http://127.0.0.1:11434",
        model: str = "qwen2.5:1.5b",
        *,
        allow_remote: bool = False,
        availability_ttl_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not allow_remote and not is_loopback_url(url):
            raise LLMError(
                f"ollama url {url!r} is not local; set llm.allow_remote_ollama = true to allow it"
            )
        self.url = url.rstrip("/")
        self.model = model
        self.ttl = availability_ttl_s
        self._clock = clock
        self._avail: tuple[float, bool] | None = None

    def _request(self, path: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(  # noqa: S310 - scheme validated (http/https loopback)
            self.url + path, data=data, headers={"Content-Type": "application/json"},
            method="POST" if data is not None else "GET",
        )  # fmt: skip
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                body = resp.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMError(f"ollama no responde: {exc}") from exc
        try:
            parsed = json.loads(body)
        except ValueError as exc:
            raise LLMError("respuesta inválida de ollama") from exc
        if not isinstance(parsed, dict):
            raise LLMError("respuesta inválida de ollama")
        return parsed

    def available(self) -> bool:
        now = self._clock()
        if self._avail is not None and now - self._avail[0] < self.ttl:
            return self._avail[1]
        try:
            self._request("/api/tags", None, timeout=0.5)
            ok = True
        except LLMError:
            ok = False
        self._avail = (now, ok)
        return ok

    def complete(self, system: str, user: str, *, timeout: float) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        resp = self._request("/api/chat", payload, timeout=timeout)
        try:
            return str(resp["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise LLMError("respuesta de ollama sin 'message.content'") from exc


class ClaudeCliBackend:
    """Uses the user's own ``claude`` CLI in print mode. Prompt goes through stdin."""

    name = "claude"

    def __init__(
        self,
        model: str = "",
        *,
        which: Callable[[str], str | None] = shutil.which,
        run: Callable[..., Any] = subprocess.run,
    ) -> None:
        self.model = model
        self._which = which
        self._run = run

    def available(self) -> bool:
        return self._which("claude") is not None

    def complete(self, system: str, user: str, *, timeout: float) -> str:
        exe = self._which("claude")
        if exe is None:
            raise LLMError("claude CLI no encontrado")
        argv = [exe, "-p", "--output-format", "json"]
        if self.model:
            argv += ["--model", self.model]
        try:
            proc = self._run(
                argv, input=f"{system}\n\n---\n\n{user}", capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout,
            )  # fmt: skip
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise LLMError(f"claude -p falló: {exc}") from exc
        if proc.returncode != 0:
            raise LLMError(
                f"claude -p terminó con código {proc.returncode}: {proc.stderr.strip()[:200]}"
            )
        try:
            return str(json.loads(proc.stdout)["result"])
        except (ValueError, KeyError, TypeError):
            return str(proc.stdout)


class FakeLLM:
    name = "fake"

    def __init__(
        self, responses: Iterable[str | Exception] = (), *, available: bool = True
    ) -> None:
        self.responses = list(responses)
        self._available = available
        self.calls: list[tuple[str, str]] = []

    def available(self) -> bool:
        return self._available

    def complete(self, system: str, user: str, *, timeout: float) -> str:
        self.calls.append((system, user))
        if not self.responses:
            raise LLMError("no scripted response")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object found in ``text`` (tolerates code fences / chatter)."""
    text = text.strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except ValueError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start : i + 1])
                        if isinstance(value, dict):
                            return value
                    except ValueError:
                        break
                    break
        start = text.find("{", start + 1)
    raise LLMError("la respuesta no contiene un objeto JSON válido")
