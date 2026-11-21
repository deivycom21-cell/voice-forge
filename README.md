# voice-forge 🎙️⚒️

> **ES** · Copiloto y director de orquesta por voz, **100% offline**, para desarrolladores que trabajan en la terminal
> (Claude Code, Cursor, Git, `swarm-pilot`, `pytest`…).
> **EN** · A **100% offline** voice copilot and conductor for terminal-first developers.

[Español](#español) · [English](#english) · [Plan / Roadmap](PLAN.md) · [Changelog](CHANGELOG.md)

```
 ● ESCUCHANDO   ▮▮▮▮▮▮▮▮▮▮▮▮▯▯▯▯▯▯▯▯▯▯▯▯          voice-forge · CTRL+ALT+V · base · llm:none · offline
 C:\dev\mi-proyecto  ⎇ main
╭──────────────────────────────────────────────────────────────────────────────────────────────╮
│ dijiste    “haz push forzado”                                                                │
│ comando    $ git push --force-with-lease                                                     │
│ estado     ⚠ git push forzado/destructivo rescribe historial remoto — di "confirmo" (10s)    │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯
```

---

## Español

### ¿Qué es?

`voice-forge` escucha tu voz cuando pulsas un atajo de teclado y la **transcribe en tu máquina** con Whisper.
Luego convierte lo que dijiste en acciones: comandos de Git, tests, abrir herramientas, o un prompt técnico
limpio para Claude Code. **Ningún audio sale de tu equipo.**

### Características

| | |
|---|---|
| 🔒 **100% offline** | `faster-whisper` (`tiny`/`base`, int8) en CPU. Medido: **~230 MB de RAM** con `base`. Funciona en equipos de 8 GB. |
| ⚡ **Instantáneo** | El parser de reglas (ES/EN) resuelve una frase en **~0.2 ms**, sin red ni IA. |
| 🧭 **Consciente del contexto** | Lee la rama, los archivos modificados, el tipo de proyecto (Python/Node/Rust/Go) y las herramientas instaladas. "Corre los tests" ejecuta `pytest` o `npm test` según el proyecto. |
| 🛡️ **Escudo de seguridad** | `rm -rf`, `reset --hard`, `push --force` y similares **siempre** piden "confirmo". Si no respondes a tiempo, se cancela. Nunca se usa `shell=True`. |
| 🤖 **IA opcional** | `ollama` local o `claude -p` (con tu propia clave) para frases que las reglas no entienden y para pulir prompts. Desactivada por defecto. |
| 🗣️ **Voz de respuesta offline** | Piper, SAPI5 (Windows) o espeak-ng (Linux). Confirmaciones breves, no bloquea. |
| 🖥️ **HUD en terminal** | Estado, nivel de micrófono, transcripción, comando, salida e historial. |

### Instalación

Requiere Python 3.10+.

```bash
pip install git+https://github.com/deivycom21-cell/voice-forge.git

# extras opcionales: voz neural Piper, icono en la bandeja, medición de memoria
pip install "voice-forge[piper,tray,bench] @ git+https://github.com/deivycom21-cell/voice-forge.git"
```

**Linux:** necesitas PortAudio y, para la voz, espeak-ng:

```bash
sudo apt install libportaudio2 espeak-ng xclip
```

### Primeros pasos

```bash
voice-forge models download base   # única operación con red: descarga el modelo (~145 MB) una vez
voice-forge doctor                 # verifica micrófono, modelo, voz, atajo y herramientas
voice-forge listen                 # pulsa CTRL+ALT+V, habla y suelta
```

Otros comandos útiles:

```bash
voice-forge listen --dry-run                 # interpreta pero no ejecuta nada
voice-forge listen --target claude           # las delegaciones van directo a `claude -p`
voice-forge listen --mode hold               # mantener pulsado para hablar
voice-forge listen --terminal                # usa ENTER en vez del atajo global (Wayland, SSH)
voice-forge listen --from-wav orden.wav      # procesa un archivo en vez del micrófono
voice-forge interpret "corre los tests"      # prueba la gramática escribiendo (dry-run)
voice-forge interpret "estado de git" -x     # …y ejecútalo de verdad
voice-forge doctor --bench                   # mide carga del modelo, latencia y memoria
voice-forge config show | init | set stt.model tiny | edit
```

### Qué puedes decir

| Dices | Ejecuta |
|---|---|
| "corre los tests" · "ejecuta las pruebas" | `pytest -q` / `npm test` / `cargo test` / `go test ./...` |
| "corre los tests fallidos" | `pytest -q --lf` |
| "corre los tests del parser" | `pytest -q -k parser` |
| "corre el linter" | `ruff check .` / `npm run lint` |
| "estado de git" · "¿qué ha cambiado?" | `git status --short --branch` |
| "muestra el diff" | `git diff --stat` |
| "últimos cinco commits" | `git log --oneline -n 5` |
| "¿en qué rama estoy?" · "lista las ramas" | `git branch --show-current` · `git branch --list` |
| "crea rama feature barra login" | `git switch -c feature/login` |
| "cambia a la rama main" | `git switch main` |
| "agrega todo" | `git add -A` |
| "commit con mensaje arregla el parser" | `git commit -m "Arregla el parser"` |
| "trae los cambios" · "sube los cambios" | `git pull --ff-only` · `git push` (con `-u origin <rama>` si es nueva) |
| "haz push forzado" ⚠️ | `git push --force-with-lease` (pide confirmación) |
| "descarta los cambios" ⚠️ | `git reset --hard` (pide confirmación) |
| "haz stash" · "recupera el stash" | `git stash push` · `git stash pop` |
| "abre claude" · "abre cursor" | nueva terminal con `claude` · `cursor .` |
| "abre el dashboard" | `swarm-pilot dashboard` (alias editable) |
| "**Claude,** agrega validación al config" | prompt estructurado → portapapeles / `claude -p` |
| "dile a Claude que…" · "agente…" | ídem |
| "confirmo" / "cancela" | responde a una confirmación |
| "repite" · "historial" · "silencio" · "ayuda" · "salir" | control de voice-forge |

Las frases largas que no encajan en ninguna regla se tratan como petición al agente. La gramática tolera
errores típicos de Whisper ("guitar status", "corre los testes", "acamit con mensaje…").

### Delegación a agentes

Cuando le hablas a Claude, voice-forge construye un prompt así (sin IA, de forma determinista):

```markdown
## Tarea
Agrega validación al módulo de configuración.

## Contexto
- Directorio: /home/tú/proyecto
- Rama: feature/config
- Archivos modificados: voice_forge/config.py, tests/test_config.py
- Proyecto: python (pyproject.toml)

## Instrucciones
- Haz cambios mínimos y enfocados en la tarea.
- Ejecuta los tests relevantes antes de terminar.
- Al final, resume qué cambiaste y por qué.
```

Destino configurable: `clipboard` (por defecto), `claude` (`claude -p`, el prompt va por **stdin**) o `print`.
Con `llm.backend = "ollama"` un modelo local pule la redacción antes.

### Configuración

Global: `voice-forge config path`. Por proyecto: `.voice-forge.toml` en la raíz del repo (se fusiona encima).

```toml
[stt]
model = "base"          # tiny | base | small | tiny.en | base.en
language = "es"         # es | en | auto

[hotkey]
combo = "ctrl+alt+v"
mode = "toggle"         # toggle | hold

[llm]
backend = "none"        # none | ollama | claude
ollama_model = "qwen2.5:1.5b"

[delegate]
target = "clipboard"    # clipboard | claude | print

[safety]
confirm_timeout_s = 10
extra_confirm_patterns = ["\\bdeploy\\b"]
extra_blocked_patterns = ["terraform destroy"]

[aliases]
"abre el dashboard" = "swarm-pilot dashboard"
"despliega en staging" = { cmd = "make deploy ENV=staging", confirm = true }
```

### Privacidad y seguridad

- El audio **nunca** se envía a la red. Un test de CI bloquea los sockets y recorre el camino audio→texto→intención.
- La única conexión de red es `voice-forge models download`. En ejecución se fuerza `HF_HUB_OFFLINE=1`.
- `ollama` solo se acepta en `localhost` salvo que pongas `allow_remote_ollama = true`.
- `claude -p` es opcional y recibe únicamente **texto**. El HUD lo indica.
- El escudo de seguridad revisa el comando final. Todo comando propuesto por una IA pide confirmación.
  Las metacaracteres de shell (`;`, `&&`, `|`, `$( )`…) en comandos de IA o alias se bloquean.

### Solución de problemas

| Problema | Solución |
|---|---|
| `PortAudio library not found` | Linux: `sudo apt install libportaudio2` |
| El atajo no responde (Wayland) | Los hooks globales no funcionan en Wayland: usa `voice-forge listen --terminal` (ENTER) |
| No hay voz de respuesta | `voice-forge doctor`. En Linux: `sudo apt install espeak-ng` |
| El portapapeles falla en Linux | `sudo apt install xclip` (X11) o `wl-clipboard` (Wayland) |
| Transcribe mal términos técnicos | Usa `base` en vez de `tiny` y añade palabras a `stt.vocabulary` |
| Corta antes de que termine de hablar | Sube `audio.silence_ms` (p. ej. 1000) |
| No detecta mi voz (micrófono bajo) | Baja `audio.abs_threshold_db` o `audio.margin_db`; mira el nivel con `voice-forge doctor --mic-test` |

---

## English

### What is it?

`voice-forge` listens when you press a hotkey and **transcribes your voice on your machine** with Whisper.
It then turns what you said into actions: Git commands, tests, launching tools, or a clean technical
prompt for Claude Code. **No audio ever leaves your computer.**

### Features

- **100% offline:** `faster-whisper` (`tiny`/`base`, int8) on CPU. Measured **~230 MB RAM** with `base`. Runs on 8 GB machines.
- **Instant:** the rule parser (Spanish + English) resolves a phrase in **~0.2 ms**, with no network or AI.
- **Context-aware:** reads the branch, modified files, project type (Python/Node/Rust/Go) and installed tools.
- **Safety shield:** `rm -rf`, `reset --hard`, `push --force`… **always** require you to say "confirm". Timeout cancels.
- **Optional AI:** local `ollama` or `claude -p` (your own key) for free-form requests and prompt polishing. Off by default.
- **Offline voice feedback:** Piper, SAPI5 (Windows) or espeak-ng (Linux).
- **Terminal HUD** built with `rich`.

### Install

```bash
pip install git+https://github.com/deivycom21-cell/voice-forge.git
# Linux only:
sudo apt install libportaudio2 espeak-ng xclip
```

### Quick start

```bash
voice-forge models download base   # the only networked step
voice-forge doctor
voice-forge listen                 # press CTRL+ALT+V and speak
voice-forge interpret "run the tests"   # try the grammar by typing
```

Set `intent.ui_language = "en"` and `stt.language = "en"` (or `"auto"`) for English feedback and recognition.

### What you can say

"run the tests" · "rerun failed tests" · "run the linter" · "git status" · "show me the diff" ·
"last 3 commits" · "what branch am I on" · "create branch hotfix dash auth" · "switch to branch develop" ·
"add all changes" · "commit with message fix the tokenizer" · "pull the latest changes" · "push the changes" ·
"force push" ⚠️ · "discard all changes" ⚠️ · "open claude" · "open cursor" · "open the dashboard" ·
"**Claude,** add type hints to the CLI" · "ask Claude to …" · "confirm" / "cancel" · "repeat" · "history" ·
"mute" · "quit".

### Privacy

Audio never leaves your machine. The only network access is `voice-forge models download`. Ollama is
loopback-only by default. `claude -p` is opt-in and receives text only.

---

## Development

```bash
git clone https://github.com/deivycom21-cell/voice-forge.git
cd voice-forge
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                    # ~370 deterministic tests: no microphone, model, LLM or network
pytest -m model           # optional: real Whisper model + system TTS voice
ruff check . && ruff format --check . && mypy voice_forge
python scripts/bench_intent.py
```

Architecture, design decisions and test strategy: [PLAN.md](PLAN.md).

## License

[MIT](LICENSE)
