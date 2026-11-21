"""voice-forge command line interface."""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from voice_forge import __version__
from voice_forge.config import Config, ConfigError, global_config_path, load_config, set_value

app = typer.Typer(
    name="voice-forge",
    help="Copiloto de voz 100% offline para la terminal · 100% offline voice copilot for the terminal.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
models_app = typer.Typer(help="Gestiona los modelos Whisper locales / Manage local Whisper models.",
                         no_args_is_help=True)  # fmt: skip
config_app = typer.Typer(help="Configuración / Configuration.", no_args_is_help=True)
app.add_typer(models_app, name="models")
app.add_typer(config_app, name="config")

console = Console()
err = Console(stderr=True)

# Typer needs Optional[...] at runtime on 3.10 for some types; keep annotations explicit.
OptStr = Optional[str]  # noqa: UP045
OptPath = Optional[Path]  # noqa: UP045


def _version(value: bool) -> None:
    if value:
        console.print(f"voice-forge {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Muestra la versión."),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Log de depuración en archivo.")
    ] = False,
) -> None:
    if verbose:
        from platformdirs import user_log_dir

        log_dir = Path(user_log_dir("voice-forge", appauthor=False))
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.DEBUG, filename=log_dir / "voice-forge.log",
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")  # fmt: skip


def _load(cwd: Path | None = None) -> Config:
    try:
        return load_config(cwd)
    except ConfigError as exc:
        err.print(f"[red]Configuración inválida:[/red] {exc}")
        raise typer.Exit(2) from exc


# --------------------------------------------------------------------------- printer (non-HUD output)


def make_printer(out: Console) -> Any:
    from voice_forge.core import events as ev
    from voice_forge.core.types import IntentKind, Risk

    def printer(event: ev.Event) -> None:
        if isinstance(event, ev.Transcribed):
            out.print(f"[grey62]dijiste:[/grey62] “{event.transcript.text}”")
        elif isinstance(event, ev.IntentResolved):
            i = event.intent
            color = {Risk.SAFE: "green", Risk.CONFIRM: "dark_orange", Risk.BLOCKED: "red"}[
                event.risk
            ]
            if i.kind == IntentKind.EXECUTE:
                out.print(
                    f"[{color}]$ {' '.join(i.argv)}[/{color}]  [grey50]({i.source}, {i.name})[/grey50]"
                )
            elif i.kind == IntentKind.DELEGATE:
                what = "pregunta" if i.delegate_kind == "question" else "orden de código"
                out.print(f"[cyan]→ agente ({what})[/cyan]")
            else:
                out.print(f"[cyan]{i.kind.value}[/cyan] {i.display}")
            if event.reasons:
                out.print(f"[dark_orange]riesgo: {'; '.join(event.reasons)}[/dark_orange]")
        elif isinstance(event, ev.ActionFinished):
            r = event.result
            if r.stdout:
                out.print(r.stdout, markup=False, highlight=False)
            if r.stderr:
                out.print(r.stderr, style="grey62", markup=False, highlight=False)
            status = "[green]✓[/green]" if r.ok else "[red]✗[/red]"
            out.print(
                f"{status} {r.error or ('en segundo plano' if r.detached else f'código {r.returncode}')}"
            )
        elif isinstance(event, ev.Delegated):
            out.print(Syntax(event.prompt, "markdown", theme="ansi_dark", word_wrap=True))
            out.print(("[green]✓[/green] " if event.ok else "[red]✗[/red] ") + event.detail)
        elif isinstance(event, ev.Answered):
            from voice_forge.ui.hud import answer_panel

            out.print(f"[cyan]pregunta:[/cyan] {event.question}", highlight=False)
            if event.ok:
                out.print(answer_panel(event.answer))
                out.print(f"[green]✓[/green] {event.detail}")
            else:
                out.print(f"[red]✗[/red] {event.detail}")
        elif isinstance(event, ev.Blocked):
            out.print(f"[red]⛔ bloqueado:[/red] {'; '.join(event.reasons)}")
        elif isinstance(event, ev.ConfirmationRequested):
            out.print("[dark_orange]⚠ requiere confirmación[/dark_orange]")
        elif isinstance(event, ev.ConfirmationResolved):
            out.print(
                "[green]confirmado[/green]"
                if event.accepted
                else f"[grey62]cancelado ({event.reason})[/grey62]"
            )
        elif isinstance(event, (ev.Info,)):
            out.print(f"[grey74]{event.message}[/grey74]")
        elif isinstance(event, ev.NothingHeard):
            out.print("[grey62]no se oyó nada[/grey62]")
        elif isinstance(event, ev.ErrorEvent):
            out.print(f"[red]error ({event.where}):[/red] {event.message}")

    return printer


# --------------------------------------------------------------------------- listen


@app.command()
def listen(
    model: Annotated[OptStr, typer.Option(help="Modelo Whisper: tiny | base | small …")] = None,
    lang: Annotated[OptStr, typer.Option(help="Idioma de reconocimiento: es | en | auto.")] = None,
    target: Annotated[
        OptStr, typer.Option(help="Destino de delegación: clipboard | claude | print.")
    ] = None,
    llm: Annotated[
        OptStr, typer.Option(help="Cerebro IA opcional: none | ollama | claude.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Interpreta pero no ejecuta nada.")
    ] = False,
    no_tts: Annotated[bool, typer.Option("--no-tts", help="Sin voz de respuesta.")] = False,
    from_wav: Annotated[
        OptPath, typer.Option(help="Procesa un archivo WAV en vez del micrófono.")
    ] = None,
    hotkey: Annotated[OptStr, typer.Option(help="Atajo, p. ej. ctrl+alt+v.")] = None,
    mode: Annotated[OptStr, typer.Option(help="toggle | hold.")] = None,
    terminal: Annotated[
        bool, typer.Option("--terminal", help="Usa ENTER en la terminal en vez del atajo global.")
    ] = False,
    tray: Annotated[
        bool, typer.Option("--tray", help="Icono de estado en la bandeja (extra [tray]).")
    ] = False,
) -> None:
    """Escucha y ejecuta órdenes de voz · Listen and act on voice commands."""
    from voice_forge.app import Overrides, apply_overrides, build_orchestrator
    from voice_forge.audio.hotkey import HotkeyError, build_trigger
    from voice_forge.core import events as ev
    from voice_forge.stt import models
    from voice_forge.ui.hud import Hud, HudState

    for name, value, choices in (
        ("--lang", lang, ("es", "en", "auto")),
        ("--target", target, ("clipboard", "claude", "print")),
        ("--llm", llm, ("none", "ollama", "claude")),
        ("--mode", mode, ("toggle", "hold")),
    ):
        if value is not None and value not in choices:
            err.print(f"[red]{name} debe ser uno de: {', '.join(choices)}[/red]")
            raise typer.Exit(2)

    cfg = apply_overrides(_load(), Overrides(model=model, language=lang, target=target, llm=llm,  # type: ignore[arg-type]
                                             no_tts=no_tts))  # fmt: skip
    if hotkey:
        cfg.hotkey.combo = hotkey
    if mode:
        cfg.hotkey.mode = mode  # type: ignore[assignment]
    if terminal:
        cfg.hotkey.backend = "terminal"

    try:
        if not models.is_downloaded(cfg.stt.model):
            err.print(f"[red]El modelo '{cfg.stt.model}' no está descargado.[/red] "
                      f"Ejecuta: [bold]voice-forge models download {cfg.stt.model}[/bold]")  # fmt: skip
            raise typer.Exit(1)
    except models.ModelError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    bus = ev.EventBus()
    orch = build_orchestrator(cfg, bus=bus, dry_run=dry_run, from_wav=from_wav)

    with console.status(f"Cargando modelo Whisper '{cfg.stt.model}'…"):
        try:
            orch.transcriber.load()  # type: ignore[attr-defined]
        except Exception as exc:
            err.print(f"[red]No se pudo cargar el modelo:[/red] {exc}")
            raise typer.Exit(1) from exc

    if from_wav is not None:
        if not from_wav.is_file():
            err.print(f"[red]No existe: {from_wav}[/red]")
            raise typer.Exit(2)
        bus.subscribe(make_printer(console))
        outcome = orch.listen_once()
        if outcome.status == "confirm_requested":
            answer = typer.confirm("¿Confirmas?", default=False)
            outcome = orch.handle_text("confirmo" if answer else "no")
        raise typer.Exit(0 if outcome.status not in ("error", "failed") else 1)

    try:
        trigger = build_trigger(cfg.hotkey.combo, cfg.hotkey.mode, cfg.hotkey.backend)
    except HotkeyError as exc:
        err.print(f"[red]Atajo inválido:[/red] {exc}")
        raise typer.Exit(2) from exc

    ctx = orch.probe.snapshot()
    state = HudState(
        hint=trigger.hint or cfg.hotkey.combo.upper(), model=cfg.stt.model, llm=cfg.llm.backend,
        target=cfg.delegate.target, cwd=str(ctx.root), branch=ctx.branch or "", dry_run=dry_run,
    )  # fmt: skip
    hud = Hud(state, console)
    bus.subscribe(hud)
    tray_icon = None
    if tray:
        from voice_forge.ui.tray import TrayStatus, tray_available

        if tray_available():
            tray_icon = TrayStatus()
            tray_icon.start()
            bus.subscribe(tray_icon)
        else:
            err.print(
                "[yellow]Instala el extra [tray] para el icono: pip install 'voice-forge[tray]'[/yellow]"
            )
    if trigger.name == "terminal":
        console.print("[yellow]Atajo global no disponible: pulsa ENTER para hablar.[/yellow]")
    stop = threading.Event()
    try:
        with hud:
            orch.run(trigger, stop)
    except KeyboardInterrupt:
        stop.set()
    finally:
        orch.speaker.close()
        if tray_icon is not None:
            tray_icon.stop()
    console.print("[grey62]voice-forge detenido.[/grey62]")


# --------------------------------------------------------------------------- interpret


@app.command()
def interpret(
    text: Annotated[str, typer.Argument(help='Frase a interpretar, p. ej. "corre los tests".')],
    execute: Annotated[
        bool, typer.Option("--execute", "-x", help="Ejecuta de verdad (por defecto: dry-run).")
    ] = False,
    target: Annotated[OptStr, typer.Option(help="clipboard | claude | print.")] = None,
    llm: Annotated[OptStr, typer.Option(help="none | ollama | claude.")] = None,
) -> None:
    """Interpreta una frase escrita (sin micrófono) · Interpret typed text."""
    from voice_forge.app import Overrides, apply_overrides, build_orchestrator
    from voice_forge.core import events as ev
    from voice_forge.stt.transcriber import FakeTranscriber
    from voice_forge.tts.speaker import NullSpeaker

    cfg = apply_overrides(_load(), Overrides(target=target, llm=llm, no_tts=True))  # type: ignore[arg-type]
    bus = ev.EventBus()
    bus.subscribe(make_printer(console))
    orch = build_orchestrator(cfg, bus=bus, dry_run=not execute, transcriber=FakeTranscriber(),
                              speaker=NullSpeaker())  # fmt: skip
    outcome = orch.handle_text(text)
    if outcome.status == "confirm_requested":
        answer = execute and typer.confirm("¿Confirmas?", default=False)
        outcome = orch.handle_text("confirmo" if answer else "no")
    if outcome.status in ("error", "failed"):
        raise typer.Exit(1)


# --------------------------------------------------------------------------- last answer


@app.command()
def last(
    raw: Annotated[
        bool, typer.Option("--raw", help="Solo el texto de la respuesta (para tuberías).")
    ] = False,
) -> None:
    """Muestra la última respuesta de Claude · Show the last answer from Claude."""
    from rich.markdown import Markdown

    from voice_forge.actions.answers import last_answer_path, load_last_answer

    record = load_last_answer()
    if record is None:
        err.print(
            "[yellow]Todavía no hay respuestas guardadas.[/yellow] Haz una pregunta con "
            '[bold]--target claude[/bold] o [bold]--llm claude[/bold], p. ej. "Claude, ¿qué es un VAD?"'
        )
        raise typer.Exit(1)
    if raw:
        typer.echo(record.answer)
        return
    console.print(f"[cyan]pregunta:[/cyan] {record.question}", highlight=False)
    meta = " · ".join(x for x in (record.created_at, record.cwd) if x)
    if meta:
        console.print(f"[grey50]{meta}[/grey50]", highlight=False)
    console.print(Markdown(record.answer))
    console.print(f"[grey50]{last_answer_path()}[/grey50]", highlight=False)


# --------------------------------------------------------------------------- models


@models_app.command("list")
def models_list() -> None:
    """Lista los modelos disponibles y descargados."""
    from voice_forge.stt import models

    table = Table(title="Modelos Whisper (faster-whisper, int8)")
    for col in ("modelo", "descargado", "tamaño aprox.", "ruta"):
        table.add_column(col)
    for m in models.list_models():
        table.add_row(
            m.name,
            "✅" if m.downloaded else "—",
            f"{m.approx_mb} MB",
            str(m.path) if m.downloaded else "",
        )
    console.print(table)


@models_app.command("download")
def models_download(
    name: Annotated[str, typer.Argument(help="tiny | base | tiny.en | base.en | small")] = "base",
) -> None:
    """Descarga un modelo (única operación con red)."""
    from voice_forge.stt import models

    try:
        if models.is_downloaded(name):
            console.print(
                f"[green]✓[/green] '{name}' ya está descargado en {models.model_path(name)}"
            )
            return
        console.print(f"Descargando '{name}' (~{models.CATALOG[name][1]} MB) desde Hugging Face…")
        path = models.download(name)
    except models.ModelError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    except Exception as exc:
        err.print(f"[red]Error de descarga:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[green]✓[/green] listo: {path}")


@models_app.command("remove")
def models_remove(name: Annotated[str, typer.Argument()]) -> None:
    """Borra un modelo descargado."""
    from voice_forge.stt import models

    try:
        removed = models.remove(name)
    except models.ModelError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]✓[/green] '{name}' eliminado" if removed else f"'{name}' no estaba descargado"
    )


# --------------------------------------------------------------------------- doctor


def _rss_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / 1e6)
    except Exception:
        return None


@app.command()
def doctor(
    bench: Annotated[
        bool, typer.Option("--bench", help="Mide carga/latencia del modelo y memoria.")
    ] = False,
    mic_test: Annotated[
        bool, typer.Option("--mic-test", help="Graba 1 s y muestra el nivel.")
    ] = False,
) -> None:
    """Verifica micrófono, modelo, TTS, atajo y herramientas."""
    from voice_forge.audio.hotkey import detect_backend
    from voice_forge.audio.sources import list_input_devices
    from voice_forge.context.probe import ContextProbe
    from voice_forge.intent.llm import ClaudeCliBackend, OllamaBackend
    from voice_forge.stt import models
    from voice_forge.tts.speaker import available_engines

    cfg = _load()
    rows: list[tuple[str, str, str]] = []

    def add(check: str, ok: bool | None, detail: str) -> None:
        rows.append((check, "✅" if ok else ("⚠️" if ok is None else "❌"), detail))

    add(
        "Python",
        sys.version_info >= (3, 10),
        f"{platform.python_version()} · {platform.system()} {platform.release()}",
    )
    try:
        import sounddevice as sd

        devices = list_input_devices(sd)
        default_in = None
        with contextlib.suppress(Exception):
            default_in = sd.query_devices(kind="input")["name"]
        add(
            "Micrófono",
            bool(devices),
            f"{len(devices)} entradas · por defecto: {default_in or '—'}",
        )
        if mic_test and devices:
            from voice_forge.audio.vad import rms_db

            rec = sd.rec(16000, samplerate=16000, channels=1, dtype="float32")
            sd.wait()
            add("Nivel (1 s)", None, f"{rms_db(rec.reshape(-1)):.1f} dBFS (habla normal ≈ -30…-10)")
    except Exception as exc:
        hint = " → sudo apt install libportaudio2" if sys.platform.startswith("linux") else ""
        add("Micrófono", False, f"PortAudio/sounddevice no disponible: {exc}{hint}")

    try:
        downloaded = models.is_downloaded(cfg.stt.model)
        add(f"Modelo '{cfg.stt.model}'", downloaded,
            str(models.model_path(cfg.stt.model)) if downloaded else f"voice-forge models download {cfg.stt.model}")  # fmt: skip
    except models.ModelError as exc:
        downloaded = False
        add("Modelo", False, str(exc))

    if bench and downloaded:
        import numpy as np

        from voice_forge.app import build_transcriber

        rss0 = _rss_mb()
        t0 = time.perf_counter()
        tr = build_transcriber(cfg)
        tr.load()
        load_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        tr.transcribe(np.zeros(16000 * 3, dtype=np.float32))
        add("Carga / 3 s de audio", True, f"{load_s:.2f} s / {time.perf_counter() - t1:.2f} s")
        rss1 = _rss_mb()
        if rss0 is not None and rss1 is not None:
            add("Memoria (RSS)", rss1 < 500, f"{rss1:.0f} MB (modelo ≈ {rss1 - rss0:.0f} MB)")
        else:
            add("Memoria (RSS)", None, "instala el extra [bench] (psutil) para medir")

    if bench:
        from voice_forge.intent.rules import RuleParser

        parser = RuleParser(cfg.aliases)
        phrases = ["corre los tests", "estado de git", "haz push forzado", "algo sin sentido aqui"]
        t0 = time.perf_counter()
        for _ in range(250):
            for p in phrases:
                parser.parse(p)
        avg_ms = (time.perf_counter() - t0) * 1000 / 1000
        add("Parser de reglas", avg_ms < 100, f"{avg_ms:.3f} ms por frase")

    engines = available_engines()
    if not cfg.tts.enabled or cfg.tts.engine == "none":
        add("Voz (TTS)", None, "desactivada")
    else:
        add(
            "Voz (TTS)",
            bool(engines),
            ", ".join(engines) or "sin motor: pip install pyttsx3 / apt install espeak-ng",
        )

    backend = cfg.hotkey.backend if cfg.hotkey.backend != "auto" else detect_backend()
    add("Atajo global", backend == "pynput" or None,
        f"{cfg.hotkey.combo} ({cfg.hotkey.mode})" if backend == "pynput" else "no disponible (Wayland/sin X): se usará ENTER")  # fmt: skip

    if sys.platform.startswith("linux"):
        clip = any(shutil.which(x) for x in ("xclip", "xsel", "wl-copy"))
        add("Portapapeles", clip or None, "ok" if clip else "instala xclip, xsel o wl-clipboard")

    tools = ContextProbe().tools()
    add("Herramientas", "git" in tools or None, ", ".join(sorted(tools)) or "ninguna")

    if cfg.llm.backend == "ollama":
        try:
            ok = OllamaBackend(cfg.llm.ollama_url, cfg.llm.ollama_model,
                               allow_remote=cfg.llm.allow_remote_ollama).available()  # fmt: skip
            add("Ollama", ok, cfg.llm.ollama_url if ok else f"no responde en {cfg.llm.ollama_url}")
        except Exception as exc:
            add("Ollama", False, str(exc))
    elif cfg.llm.backend == "claude":
        add("claude -p", ClaudeCliBackend().available(), "texto → API de Anthropic (opt-in)")
    else:
        add("Cerebro IA", None, "desactivado (solo reglas, 100% offline)")

    table = Table(title=f"voice-forge doctor · {__version__}")
    table.add_column("verificación")
    table.add_column("")
    table.add_column("detalle")
    for row in rows:
        table.add_row(*row)
    console.print(table)
    if any(r[1] == "❌" for r in rows):
        raise typer.Exit(1)


# --------------------------------------------------------------------------- config


@config_app.command("show")
def config_show() -> None:
    """Muestra la configuración efectiva (global + proyecto)."""
    cfg = _load()
    console.print(
        f"[grey62]global:[/grey62] {cfg.global_file or '(sin archivo, valores por defecto)'}"
    )
    console.print(f"[grey62]proyecto:[/grey62] {cfg.project_file or '(ninguno)'}")
    console.print(Syntax(cfg.to_toml(), "toml", theme="ansi_dark"))


@config_app.command("path")
def config_path() -> None:
    """Ruta del archivo de configuración global."""
    console.print(str(global_config_path()))


@config_app.command("init")
def config_init(
    force: Annotated[bool, typer.Option("--force", help="Sobrescribe si existe.")] = False,
) -> None:
    """Crea el archivo de configuración global con los valores por defecto."""
    path = global_config_path()
    if path.exists() and not force:
        err.print(f"[yellow]Ya existe:[/yellow] {path} (usa --force para sobrescribir)")
        raise typer.Exit(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(Config().to_toml(), encoding="utf-8")
    console.print(f"[green]✓[/green] creado {path}")


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="seccion.clave, p. ej. stt.model")],
    value: Annotated[str, typer.Argument(help="Valor (literal TOML o texto).")],
) -> None:
    """Cambia un valor en la configuración global (validado)."""
    try:
        set_value(global_config_path(), key, value)
    except ConfigError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    console.print(f"[green]✓[/green] {key} = {value}")


@config_app.command("edit")
def config_edit() -> None:
    """Abre la configuración global en tu editor."""
    path = global_config_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(Config().to_toml(), encoding="utf-8")
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if editor:
        argv = [*editor.split(), str(path)]
    elif sys.platform == "win32":
        argv = ["notepad", str(path)]
    else:
        argv = ["xdg-open", str(path)]
    exe = shutil.which(argv[0])
    if exe is None:
        err.print(f"[red]No encontré el editor '{argv[0]}'. Archivo:[/red] {path}")
        raise typer.Exit(1)
    subprocess.run([exe, *argv[1:]], check=False)  # noqa: S603


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
