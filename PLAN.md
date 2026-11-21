# PLAN — voice-forge

> Copiloto y director de orquesta por voz, 100% offline, para desarrolladores que viven en la terminal
> (Claude Code, Cursor, Git, `swarm-pilot`, `pytest`…).

Estado: **v0.1.0 — fases 0–15 implementadas.** Ver [§8 Estado de implementación](#8-estado-de-implementación).

---

## 0. Principios no negociables

| # | Principio | Cómo se garantiza |
|---|-----------|-------------------|
| P1 | **El audio nunca sale de la máquina** | Whisper corre en CPU local. No existe ningún código que serialice audio hacia red. Test de guardia (`test_no_network_in_audio_path`) que parchea `socket.socket` y falla si el pipeline de audio→texto abre un socket. |
| P2 | **Offline por defecto** | La única operación de red es `voice-forge models download` (explícita). En runtime se fija `HF_HUB_OFFLINE=1` y `WhisperModel(..., local_files_only=True)`. Ollama es `localhost`. `claude -p` es **opt-in** y solo recibe **texto**. |
| P3 | **≤ 8 GB RAM totales en la máquina** | Presupuesto de memoria de voice-forge < 500 MB (ver §3). Modelo `tiny`/`base` en `int8`. Carga perezosa de todo lo pesado. |
| P4 | **Latencia de comando directo < 100 ms** (texto → acción decidida) | El parser por reglas no toca red, ni LLM, ni disco. Test de rendimiento en CI. |
| P5 | **Nada destructivo sin confirmación explícita** | El Escudo evalúa el *argv final* (no la frase), es deny-by-default para comandos generados por LLM y nunca se usa `shell=True`. |
| P6 | **Todo testeable sin micrófono, sin modelo y sin LLM** | Cada dependencia externa vive detrás de un `Protocol` con implementación falsa determinista. CI corre en Windows + Linux sin hardware de audio. |
| P7 | **Cero costos** | Solo dependencias OSS. Sin PyPI: instalación vía `pip install git+https://github.com/deivycom21-cell/voice-forge.git`. |

---

## 1. Arquitectura

### 1.1 Flujo de una orden

```
 [Hotkey global] ──► Recorder ──frames──► VAD (endpointing) ──audio──► Transcriber (faster-whisper)
                                                                              │ texto
                                                                              ▼
                         ContextProbe ──snapshot──► IntentEngine ◄── RuleParser (rápido, regex/fuzzy)
                                                         │        ◄── LLMBackend (opcional: ollama / claude -p)
                                                         ▼ Intent
                                                   SafetyShield ──(CONFIRM)──► confirmación voz/tecla
                                                         │ OK
                                       ┌─────────────────┼──────────────────┐
                                       ▼                 ▼                  ▼
                                  Executor          Delegator           (rechazo)
                              (subprocess argv)  (claude -p / clipboard)
                                       │                 │
                                       └──────► EventBus ◄┘ ──► HUD (rich)  +  Speaker (TTS)
```

Todos los componentes publican **eventos tipados** en un `EventBus` síncrono e in-process. El HUD y el TTS
son *suscriptores*; el núcleo no sabe que existen. Esto permite testear el núcleo con un bus que solo graba.

### 1.2 Mapa de paquetes

```
voice_forge/
├── __init__.py          # __version__
├── __main__.py          # python -m voice_forge
├── cli.py               # Typer: listen | models | doctor | config
├── config.py            # Config dataclasses, carga/merge TOML (global + proyecto)
├── core/
│   ├── types.py         # Intent, IntentKind, Risk, ActionResult, ContextSnapshot…
│   ├── events.py        # Eventos + EventBus
│   └── orchestrator.py  # Bucle principal: une todas las piezas (inyección de dependencias)
├── audio/
│   ├── sources.py       # Protocol AudioSource + MicSource (sounddevice) + WavFileSource
│   ├── synth.py         # Generador determinista de señales (tonos, ruido, silencio) — usado en tests
│   ├── vad.py           # VAD por energía (RMS + histéresis) + adaptador Silero opcional
│   ├── recorder.py      # Máquina de estados de grabación (pre-roll, fin por silencio, máx. duración)
│   └── hotkey.py        # Push-to-talk global (pynput) + fallback "Enter en la terminal"
├── stt/
│   ├── transcriber.py   # Protocol Transcriber + WhisperTranscriber (lazy) + FakeTranscriber
│   └── models.py        # Catálogo, descarga, rutas en caché, verificación
├── context/
│   └── probe.py         # Rama git, cambios, archivos recientes, marcadores de proyecto, herramientas
├── intent/
│   ├── normalize.py     # minúsculas, sin acentos, sin muletillas, números hablados
│   ├── rules.py         # Gramática ES/EN → Intent (determinista)
│   ├── llm.py           # Protocol LLMBackend + Ollama + ClaudeCli + Fake; salida JSON validada
│   └── engine.py        # Reglas primero, LLM como respaldo, plantillas de prompt
├── actions/
│   ├── shield.py        # Clasificador de riesgo SAFE / CONFIRM / BLOCKED
│   ├── executor.py      # CommandRunner (subprocess, argv, timeout) + FakeRunner
│   ├── delegate.py      # Envío a `claude -p` o portapapeles
│   └── confirm.py       # Flujo de confirmación (voz "confirmo" / tecla / timeout = cancelar)
├── tts/
│   └── speaker.py       # Protocol Speaker + Piper / pyttsx3 (SAPI5, espeak-ng) / espeak CLI / Null
└── ui/
    ├── hud.py           # rich.Live: estado, medidor de nivel, transcripción, comando, historial
    └── tray.py          # (opcional, extra [tray]) icono de estado con pystray
```

### 1.3 Contratos (Protocols) — la clave de la testabilidad

```python
class AudioSource(Protocol):
    sample_rate: int  # siempre 16_000 mono float32

    def frames(self) -> Iterator[np.ndarray]: ...  # bloques de 30 ms (480 muestras)
    def close(self) -> None: ...


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray, *, language: str | None) -> Transcript: ...


class LLMBackend(Protocol):
    name: str

    def available(self) -> bool: ...
    def complete(self, system: str, user: str, *, timeout: float) -> str: ...


class CommandRunner(Protocol):
    def run(self, argv: list[str], *, cwd: Path, timeout: float) -> ActionResult: ...


class Speaker(Protocol):
    def say(self, text: str) -> None: ...  # no bloqueante
```

Implementaciones falsas que viven en el propio paquete (no en tests) para que también sirvan en `--dry-run`:
`WavFileSource`, `SyntheticSource`, `FakeTranscriber`, `FakeLLM`, `FakeRunner`, `RecordingSpeaker`, `NullSpeaker`.

### 1.4 Modelo de intenciones

```python
class IntentKind(Enum):
    EXECUTE = "execute"  # comando directo (argv)
    DELEGATE = "delegate"  # prompt para un agente
    CONFIRM = "confirm"  # "sí / confirmo / dale / yes"
    CANCEL = "cancel"  # "no / cancela / para / stop"
    META = "meta"  # "repite", "historial", "silencio", "salir"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Intent:
    kind: IntentKind
    utterance: str  # transcripción original
    argv: tuple[str, ...] = ()  # para EXECUTE
    prompt: str | None = None  # para DELEGATE
    source: Literal["rules", "llm", "alias"] = "rules"
    confidence: float = 1.0
    label: str = ""  # texto corto para HUD/TTS: "git status"
```

### 1.5 Escudo de seguridad

- Opera sobre el **argv final** que se ejecutaría, nunca sobre la frase.
- Sin `shell=True` jamás. Si un comando (venga de LLM o de alias) contiene metacaracteres de shell
  (`; && || | > < $( \``), se clasifica **BLOCKED** a menos que el usuario lo haya declarado en un alias con
  `shell = true` explícito (y aun así → CONFIRM).
- Niveles:
  - **SAFE** → se ejecuta directamente (solo si viene de reglas o de alias de usuario).
  - **CONFIRM** → requiere "confirmo" por voz o tecla `y` en ≤ 10 s; el timeout **cancela**.
  - **BLOCKED** → nunca se ejecuta; se explica por qué.
- **Todo comando generado por LLM es como mínimo CONFIRM** (deny-by-default).
- Reglas destructivas iniciales (ampliables por config, nunca reducibles por debajo de la lista base):
  `rm -r/-rf/-fr`, `rmdir /s`, `del /s /q`, `Remove-Item -Recurse`, `git reset --hard`, `git push --force|-f|--force-with-lease|+refspec`,
  `git clean -f*`, `git checkout -- .`/`git restore .`, `git branch -D`, `git stash drop|clear`, `git rebase`,
  `git filter-branch`, `docker system prune`, `DROP TABLE|DATABASE`, `mkfs`, `dd`, `format`, `chmod -R`, `chown -R`, `sudo`.
- La confirmación verbal debe ser una palabra de confirmación **aislada** ("sí", "confirmo", "yes", "confirm");
  frases ambiguas ("sí pero espera") → cancelar.

### 1.6 Delegación a agente

1. Detección: frases que empiezan con "claude…", "dile a claude…", "agente…", "prompt…", "ask claude…",
   o cualquier frase larga (> 12 palabras) que no matchee una regla.
2. Estructuración:
   - **Sin LLM (determinista):** plantilla con `Objetivo`, `Contexto` (rama, archivos modificados, stack detectado),
     `Restricciones` y la petición literal limpia de muletillas.
   - **Con LLM (ollama local o `claude -p`):** se le pide reescribir a un prompt técnico; salida en JSON validado.
     Si falla o excede timeout → se usa la plantilla determinista (nunca se bloquea al usuario).
3. Destino (configurable, por defecto `clipboard`): `clipboard` | `claude` (`claude -p "<prompt>"` en el cwd) | `print`.

---

## 2. Decisiones técnicas

| Tema | Decisión | Motivo |
|------|----------|--------|
| STT | `faster-whisper`, `tiny`/`base`, `int8`, `beam_size=1`, `cpu_threads` configurable | Menos de 400 MB, rápido en CPU. |
| Sesgo de vocabulario | `initial_prompt` con jerga: "git, commit, push, pull, pytest, Claude, swarm-pilot, rama, merge…" | Whisper confunde términos técnicos; esto mejora el reconocimiento sin entrenar nada. |
| VAD en vivo | VAD **por energía** propio (RMS en dBFS + histéresis + *hangover*) | Determinista, sin compilación (webrtcvad no tiene ruedas fiables en Windows), testeable con señales sintéticas. Silero (incluido en faster-whisper) solo como `vad_filter` al transcribir. |
| Captura | `sounddevice.InputStream`, 16 kHz mono float32, bloques de 30 ms, cola acotada | PortAudio viene empaquetado en las ruedas de Windows; en Linux requiere `libportaudio2`. |
| Hotkey global | `pynput`; por defecto `<ctrl>+<alt>+v` (configurable). Modos `toggle` (default) y `hold` | `Ctrl+Space`/`Alt+V` chocan con IMEs y atajos de terminal; se pueden configurar. En **Wayland** pynput no captura globalmente → fallback automático a "pulsa Enter en la terminal". |
| CLI | `typer` | Ayuda limpia, autocompletado, testeable con `CliRunner`. |
| TUI | `rich.Live` | Ligero; los tests renderizan a `Console(record=True)`. |
| TTS | Cadena de prioridad: `piper` (extra opcional, voz neural) → `pyttsx3` (SAPI5 / espeak-ng) → `espeak-ng` CLI → `NullSpeaker` | Siempre hay algo que funciona; nunca falla por falta de TTS. Se habla en un hilo con cola para no bloquear. |
| Config | TOML. Global en `platformdirs.user_config_dir("voice-forge")/config.toml` + `.voice-forge.toml` por proyecto (merge) | El proyecto puede definir alias propios ("abre el dashboard" → `swarm-pilot dashboard`). |
| LLM local | Ollama vía `urllib` a `http://127.0.0.1:11434/api/chat` (stdlib, sin dependencias) | Cero dependencias extra. Host restringido a loopback salvo override explícito. |
| Python | `>=3.10` | Compatibilidad amplia; `tomllib` con fallback `tomli` en 3.10. |
| Build | `hatchling` | Simple, estándar PEP 621. |

---

## 3. Presupuestos

### 3.1 Memoria (objetivo < 500 MB RSS)

| Componente | Estimado |
|------------|----------|
| Intérprete + rich + typer + numpy | ~60 MB |
| faster-whisper `tiny` int8 (CTranslate2) | ~100–150 MB |
| faster-whisper `base` int8 | ~200–300 MB |
| Buffer de audio (máx. 30 s × 16 kHz × 4 B) | ~2 MB |
| pyttsx3 / SAPI5 | ~20 MB |
| **Total con `base`** | **≈ 400 MB** |

Verificado por `voice-forge doctor --bench` (mide RSS con `psutil` si está instalado, si no con `tracemalloc` aproximado).

### 3.2 Latencia

| Tramo | Objetivo |
|-------|----------|
| Fin de habla → VAD corta | 600 ms (configurable `silence_ms`) |
| Transcripción 3 s de audio, `tiny`, CPU 4 hilos | < 700 ms |
| Texto → Intent (reglas) | **< 5 ms** típico, **< 100 ms** garantizado |
| Intent → proceso lanzado | < 50 ms |

---

## 4. Estrategia de tests (deterministas, sin hardware)

- **Audio sintético** (`voice_forge/audio/synth.py`): `tone(freq, ms, amp)`, `silence(ms)`, `noise(ms, amp, seed)`,
  `concat(...)`, `to_wav(path)`. Semilla fija → mismas muestras en cualquier OS. Se generan **al vuelo** en
  `tmp_path`; no se versionan WAVs binarios (salvo uno pequeño de referencia en `tests/fixtures/`).
- **Micrófono falso**: `monkeypatch` de `sounddevice.InputStream` con una clase que invoca el `callback`
  con bloques de un array dado → prueba `MicSource` sin PortAudio.
- **Whisper falso**: `FakeTranscriber(script=["corre los tests", ...])` devuelve textos en orden, o
  `FakeTranscriber(mapping={hash_audio: texto})`. `WhisperTranscriber` se prueba con `WhisperModel` mockeado
  (verifica parámetros: `int8`, `beam_size`, `language`, `initial_prompt`, `local_files_only`).
- **LLM falso**: `FakeLLM(responses=[...])`; Ollama se prueba contra un `http.server` local en un hilo
  (puerto efímero), incluyendo timeouts y JSON inválido. `claude -p` con `subprocess.run` mockeado.
- **Ejecutor falso**: `FakeRunner` graba los argv; jamás se ejecuta nada real en tests unitarios.
- **Git real en `tmp_path`** para `context/` (se omite con `skip` si no hay `git`).
- **Tabla de frases** (`tests/fixtures/utterances.toml`, sin dependencia YAML): ~80 casos ES/EN con
  su intención esperada, incluyendo errores típicos de Whisper ("corre los test", "estado del git", "guitar status").
- **Snapshots del HUD**: render a texto plano de ancho fijo (100 col) y comparación con archivo `.txt`.
- **Marcadores pytest**: `model` (requiere modelo descargado), `mic` (requiere micrófono), `slow`. Por defecto
  `-m "not model and not mic"`.
- **CI**: GitHub Actions, matriz `ubuntu-latest` × `windows-latest` × Python `3.10`/`3.12`; `ruff check`,
  `ruff format --check`, `mypy voice_forge`, `pytest --cov` (umbral 85% excluyendo adaptadores de hardware).

---

## 5. Fases atómicas

Cada fase termina con: código + tests verdes en local + `ruff` limpio. Ninguna fase depende de hardware.

### Fase 0 — Andamiaje ✅ (esta entrega)
- Estructura de carpetas, `pyproject.toml`, `README.md` bilingüe, `PLAN.md`, `.gitignore`, `LICENSE`.
- **Pendiente para cerrar F0:** `cli.py` mínimo con `--version`, workflow CI, `tests/test_smoke.py`.
- **Tests:** `test_version_matches_pyproject`, `test_cli_version` (CliRunner).

### Fase 1 — Tipos núcleo y bus de eventos (`core/types.py`, `core/events.py`)
- Dataclasses inmutables (`Intent`, `ActionResult`, `ContextSnapshot`, `Transcript`), enums, eventos
  (`ListeningStarted`, `LevelUpdate`, `SpeechEnded`, `Transcribed`, `IntentResolved`, `ConfirmationRequested`,
  `ActionFinished`, `Error`).
- `EventBus.subscribe/publish`; un suscriptor que lanza excepción no rompe a los demás.
- **Tests:** orden de entrega, aislamiento de errores, `RecordingBus` para aserciones.

### Fase 2 — Configuración (`config.py`)
- Dataclasses con defaults; carga global + proyecto (busca `.voice-forge.toml` subiendo hasta la raíz git);
  validación con mensajes claros; `Config.to_toml()`.
- Secciones: `[audio]`, `[stt]`, `[hotkey]`, `[intent]`, `[llm]`, `[delegate]`, `[tts]`, `[safety]`, `[aliases]`.
- **Tests:** defaults, merge con precedencia, valores inválidos → `ConfigError` con ruta del campo,
  alias de usuario con comando destructivo se marca para confirmación, roundtrip TOML.

### Fase 3 — Audio sintético + VAD + grabador (`audio/synth.py`, `vad.py`, `recorder.py`, `WavFileSource`)
- `EnergyVAD`: umbral en dBFS, calibración de ruido de fondo con los primeros N frames, histéresis.
- `Recorder`: estados `IDLE → ARMED → SPEECH → TRAILING → DONE`; pre-roll de 300 ms; `min_speech_ms`;
  `silence_ms` para cortar; `max_seconds` duro; cancelación.
- **Tests (WAV sintéticos):**
  - silencio puro → no produce segmento (timeout de "no se oyó nada").
  - 500 ms silencio + 1 s tono + 800 ms silencio → corta en ±1 frame del punto esperado; duración del segmento ≈ 1.3 s (incluye pre-roll).
  - clic de 40 ms (< `min_speech_ms`) → ignorado.
  - pausa corta (300 ms) dentro de la frase → **no** corta.
  - ruido de fondo constante −45 dBFS + voz → calibración evita falsos positivos.
  - habla continua 40 s → corte en `max_seconds`.
  - `WavFileSource` lee 8/16/44.1 kHz y estéreo → re-muestrea a 16 kHz mono (resampleo lineal con numpy).

### Fase 4 — Micrófono y hotkey (`audio/sources.py::MicSource`, `audio/hotkey.py`)
- `MicSource` con cola acotada (descarta lo más viejo si se llena, cuenta *overflows*).
- `HotkeyListener` con modos `toggle`/`hold`; detección de Wayland → `TerminalTrigger` (Enter).
- **Tests:** `InputStream` falso que alimenta el callback; parseo de combinaciones (`"ctrl+alt+v"` → formato pynput);
  `hold` emite start/stop; fallback seleccionado cuando `WAYLAND_DISPLAY` está definido (monkeypatch env).

### Fase 5 — STT (`stt/transcriber.py`, `stt/models.py`)
- `WhisperTranscriber` carga perezosa, thread-safe, `vad_filter=True`, `initial_prompt` con vocabulario.
- `models`: catálogo (`tiny`, `base`, `tiny.en`, `base.en`), `download(name)` vía `huggingface_hub.snapshot_download`,
  `is_downloaded`, `path_for`, tamaño en disco.
- **Tests:** `WhisperModel` mockeado verifica kwargs; `snapshot_download` mockeado; `FakeTranscriber`;
  guardia P1 (sin sockets durante `transcribe`). Test marcado `model`: transcribe un WAV sintetizado por `espeak-ng`
  si está disponible (opcional, fuera de CI por defecto).

### Fase 6 — Conciencia de contexto (`context/probe.py`)
- `ContextProbe.snapshot(cwd) -> ContextSnapshot`: raíz git, rama, ahead/behind, archivos modificados
  (`git status --porcelain=v1 -z`), 10 archivos más recientes (ignorando `.git`, `node_modules`, `.venv`, `__pycache__`, `dist`),
  marcadores (`pyproject.toml`, `package.json`, `Cargo.toml`, `go.mod`…), herramientas (`shutil.which` de
  `git`, `claude`, `swarm-pilot`, `aether`, `pytest`, `npm`, `cursor`, `code`, `ollama`).
- Caché con TTL de 2 s; cada llamada a git con timeout de 1 s; nunca lanza excepción (degrada a campos vacíos).
- **Tests:** repo git en `tmp_path` (rama, archivo modificado, untracked), directorio sin git, `which` mockeado,
  timeout de git simulado, caché respeta TTL (reloj inyectado).

### Fase 7 — Parser por reglas (`intent/normalize.py`, `intent/rules.py`)
- Normalización: minúsculas, quitar acentos, puntuación, muletillas ("eh", "este", "por favor", "please", "porfa"),
  números hablados básicos ("uno…diez").
- Gramática inicial ES/EN (cada regla → argv o acción):
  - git: estado, diff, log corto, rama actual, cambiar a rama X, crear rama X, add todo, commit con mensaje "…",
    pull, push (normal = SAFE), stash.
  - tests: "corre los tests" → `pytest` (o `npm test` según contexto), "tests de <archivo>", "último test fallido" → `pytest --lf`.
  - herramientas: "abre claude", "abre cursor/code aquí", "abre el dashboard" → alias configurable (`swarm-pilot dashboard`).
  - control: confirmar, cancelar, repetir, historial, silencio/voz, salir.
  - delegación: prefijos "claude…", "dile a claude…", "agente…", "prompt…".
- Fuzzy matching con `difflib.SequenceMatcher` sobre frases clave (umbral 0.82) para errores de Whisper.
- El contexto modifica el resultado: "corre los tests" usa `pytest` si hay `pyproject.toml`/`pytest`, `npm test` si hay `package.json`.
- **Tests:** tabla de ~80 frases → intent esperado; idempotencia de normalización; fuzzy ("guitar status" → `git status`);
  frase no reconocida → `UNKNOWN`; **benchmark**: 1 000 frases en < 100 ms total (≈ 0.1 ms c/u) y peor caso < 100 ms.

### Fase 8 — Escudo de seguridad (`actions/shield.py`, `actions/confirm.py`)
- `Shield.assess(argv, source) -> Assessment(risk, reasons)`.
- `ConfirmationFlow`: máquina de estados con reloj inyectado; acepta `CONFIRM`/`CANCEL`/tecla; timeout → cancelar.
- **Tests (tabla adversarial):** `git push -f`, `git push origin +main`, `git push --force-with-lease`, `rm -r -f x`,
  `rm -fr x`, `Remove-Item x -Recurse -Force`, `git reset --hard HEAD~1`, `git clean -fdx`, `sudo …`,
  argv con `;`/`&&`/`$(...)` → BLOCKED; `git status`, `pytest -q`, `git push` → SAFE; cualquier argv con `source="llm"` ≥ CONFIRM;
  confirmación ambigua → cancelar; timeout → cancelar.

### Fase 9 — Ejecución y delegación (`actions/executor.py`, `actions/delegate.py`)
- `SubprocessRunner`: argv, `cwd`, timeout, captura de stdout/stderr (truncada a N líneas para el HUD), código de salida;
  en Windows resuelve `.cmd/.bat` con `shutil.which` sin `shell=True`. Comandos de larga duración (abrir claude/cursor)
  se lanzan *detached*.
- `Delegator`: `clipboard` (pyperclip), `claude` (`claude -p <prompt>` con salida en streaming al HUD), `print`.
- **Tests:** `FakeRunner`; `SubprocessRunner` real con `sys.executable -c "print('ok')"` (portable) y timeout con
  `time.sleep`; pyperclip mockeado; claude ausente → error claro, no excepción.

### Fase 10 — LLM opcional (`intent/llm.py`, `intent/engine.py`)
- `OllamaBackend` (urllib, `format: "json"`), `ClaudeCliBackend` (`claude -p --output-format json`), `FakeLLM`.
- `IntentEngine.resolve(text, ctx)`: reglas → si `UNKNOWN` o delegación larga → LLM (si está habilitado) → validación de esquema
  → fallback a plantilla determinista. Los comandos devueltos por LLM se marcan `source="llm"` (→ Escudo ≥ CONFIRM).
- **Tests:** servidor HTTP local con respuestas grabadas (OK, JSON roto, 500, timeout); `subprocess.run` mockeado para claude;
  la regla siempre gana al LLM; LLM deshabilitado → nunca se invoca (contador = 0); prompt de sistema incluye el contexto.

### Fase 11 — Voz de respuesta (`tts/speaker.py`)
- Selector por disponibilidad + config; hilo con cola; frases cortas en catálogo ES/EN ("Listo", "Corriendo tests",
  "¿Confirmas?", "Cancelado", "No te entendí"); `mute` en caliente.
- **Tests:** selector con imports mockeados; `RecordingSpeaker`; la cola no bloquea al llamador; error del motor → degrada a `NullSpeaker`.

### Fase 12 — HUD (`ui/hud.py`, `ui/tray.py`)
- Panel superior: estado (● escuchando / ◌ transcribiendo / ◆ pensando / ? confirmación) + medidor de nivel.
- Centro: transcripción y comando traducido con color por riesgo (verde/ámbar/rojo).
- Inferior: historial de las últimas 8 acciones con código de salida y duración.
- `tray.py` solo si está instalado el extra `[tray]`.
- **Tests:** alimentar una secuencia de eventos a `HudState` (puro) y comparar el render de texto con snapshots.

### Fase 13 — Orquestador (`core/orchestrator.py`)
- Une todo por inyección de dependencias; bucle: esperar trigger → grabar → transcribir → contexto → intent →
  escudo → (confirmar) → ejecutar/delegar → hablar.
- **Tests E2E deterministas:**
  1. `SyntheticSource` (tono) + `FakeTranscriber("corre los tests")` → `FakeRunner` recibe `["pytest"]`, Speaker dijo "Corriendo tests", eventos en orden.
  2. `"git push force"` → `ConfirmationRequested`; siguiente turno `"confirmo"` → se ejecuta.
  3. igual que (2) pero `"no"` → no se ejecuta nada.
  4. `"claude, refactoriza el parser para que use dataclasses"` → portapapeles contiene prompt con rama y archivos.
  5. Silencio total → evento "no se oyó nada", sin transcribir.
  6. Excepción en el transcriptor → evento `Error`, el bucle sigue vivo.

### Fase 14 — CLI (`cli.py`)
- `voice-forge listen [--model tiny|base] [--lang es|en|auto] [--dry-run] [--no-tts] [--target clipboard|claude|print] [--from-wav PATH]`
  - `--dry-run`: decide pero no ejecuta (útil para probar gramática).
  - `--from-wav`: usa un archivo en lugar del micrófono (demos y debugging).
- `voice-forge models download|list|remove [NAME]`.
- `voice-forge doctor [--bench]`: micrófono (lista dispositivos, prueba 1 s de nivel), modelo (descargado + carga),
  TTS (motor elegido), hotkey (backend disponible / Wayland), herramientas detectadas, Ollama alcanzable. Salida en tabla ✅/⚠️/❌.
- `voice-forge config show|path|init|set KEY VALUE|edit`.
- **Tests:** `CliRunner` con todas las dependencias mockeadas; códigos de salida; `doctor` sin micrófono reporta ❌ sin crashear.

### Fase 15 — Documentación y release v0.1.0
- README completo (ES/EN) con GIF de demo, tabla de comandos de voz, guía de alias, troubleshooting (PortAudio en Linux, Wayland, SAPI5).
- `scripts/`: `gen_fixtures.py` (regenera WAV de referencia), `bench_intent.py`, `record_sample.py` (graba muestra local para debugging).
- `CHANGELOG.md`, tag `v0.1.0`; verificar `pip install git+https://…@v0.1.0` en VM limpia Windows 11 y Ubuntu.

### Futuro (fuera de v0.1)
- Palabra de activación offline (wake word) opcional.
- Modo "dictado" para escribir en el editor activo.
- Integración profunda con `swarm-pilot` (estado de agentes por voz).
- Gramáticas como plugins (`entry_points`).

---

## 6. Riesgos y mitigaciones

| Riesgo | Mitigación |
|--------|------------|
| Whisper `tiny` transcribe mal términos técnicos en español | `initial_prompt` con vocabulario + fuzzy matching + opción `base`. |
| Hotkey global no funciona en Wayland / requiere permisos | Fallback automático a Enter en terminal; documentado en `doctor`. |
| `sounddevice` sin PortAudio en Linux | `doctor` detecta y sugiere `sudo apt install libportaudio2`; import perezoso para que el resto de la CLI funcione. |
| Comando destructivo ejecutado por error de transcripción | Escudo sobre argv, confirmación aislada, timeout = cancelar, LLM ≥ CONFIRM. |
| `claude -p` envía texto fuera de la máquina | Opt-in explícito, indicado en el HUD con un icono; jamás audio. |
| Dependencias pesadas en CI (ctranslate2/onnxruntime) | Imports perezosos; caché de pip en CI. |

---

## 7. Definición de "hecho" (por fase)

- [ ] Tests nuevos deterministas (sin red, sin micrófono, sin modelo) y verdes en Windows + Linux.
- [ ] `ruff check`, `ruff format --check`, `mypy voice_forge` sin errores.
- [ ] Ninguna llamada `subprocess` con `shell=True`.
- [ ] Docstrings públicas y entrada en `CHANGELOG.md`.

---

## 8. Estado de implementación

| Fase | Estado | Notas / desviaciones respecto al plan |
|------|--------|----------------------------------------|
| 0 Andamiaje | ✅ | CI en `.github/workflows/ci.yml` (Windows + Linux × Py 3.10/3.12, ruff, mypy estricto, cobertura ≥ 85%). |
| 1 Núcleo + bus | ✅ | `core/types.py`, `core/events.py` (`RecordingBus` para tests). |
| 2 Config | ✅ | Validación con ruta del campo; `set_value` valida antes de escribir. |
| 3 Audio + VAD | ✅ | Se añadió `audio/wav.py`. El segmento de referencia dura ≈ 1.53 s (no 1.3 s) porque se conservan 200 ms de silencio final, que ayudan a Whisper. |
| 4 Micrófono + hotkey | ✅ | `ManualTrigger` para tests. En modo `hold`, si sueltas antes de armar el stop, se corta al instante. |
| 5 STT | ✅ | Test real opcional: `pytest -m model` (voz SAPI5/espeak + Whisper `base`). |
| 6 Contexto | ✅ | También upstream y detached HEAD (para `git push -u origin <rama>`). |
| 7 Reglas | ✅ | 82 frases en la tabla, incluidas salidas reales de Whisper. Corrección regex de palabras ("acamit" → commit). |
| 8 Escudo | ✅ | También `mkfs.*`, `sudo` (re-evalúa el comando envuelto), intérpretes con código en línea, `docker`/`kubectl`/`publish`. |
| 9 Ejecución + delegación | ✅ | Los hijos reciben `PYTHONUTF8=1` (Windows usaba la codepage ANSI y rompía la "ñ"). El prompt para `claude -p` va por **stdin**. |
| 10 LLM opcional | ✅ | Una delegación que ya viene del LLM no se reescribe dos veces. |
| 11 TTS | ✅ | `wait_idle()` evita que el micrófono grabe nuestro propio "¿Confirmas?". |
| 12 HUD | ✅ | Los tests comprueban fragmentos del render en texto plano en vez de snapshots completos (más robusto entre terminales). |
| 13 Orquestador | ✅ | Los 6 escenarios del plan y 13 más (bucle real con trigger, expiración, repeat re-evaluado por el escudo…). |
| 14 CLI | ✅ | Comando extra `interpret "frase" [-x]` para probar la gramática escribiendo. |
| 15 Docs + release | ✅ | README bilingüe, CHANGELOG, scripts (`bench_intent.py`, `record_sample.py`, `gen_samples.py` en lugar de `gen_fixtures.py`). Publicado en GitHub con el tag `v0.1.0`. |
| 16 Órdenes vs. preguntas | ✅ | `intent/classify.py` (determinista). Las preguntas van limpias a `claude -p` (`--target claude` o `--llm claude`), la respuesta completa aparece en el panel `salida`, se lee la primera frase por TTS y `voice-forge last` la recupera (`actions/answers.py`). |

### Mediciones reales (Windows 11, CPU, modelo `base` int8)

| Métrica | Objetivo | Medido |
|---------|----------|--------|
| RSS total del proceso con el modelo cargado | < 500 MB | **231 MB** (modelo ≈ 172 MB) |
| Parser de reglas, por frase | < 100 ms | **~0.2 ms** (peor caso con fuzzy ~0.3 ms) |
| Carga del modelo `base` | — | 2.4 s (una vez, al arrancar `listen`) |
| Frases habladas reales (voz SAPI5 "Helena") reconocidas | — | 6/6 con `base` (4/6 con `tiny`) |

### Calidad

- 444 tests deterministas (+3 opcionales con modelo real).
- `ruff check`, `ruff format --check` y `mypy --strict` sin errores.
