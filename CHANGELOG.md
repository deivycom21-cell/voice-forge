# Changelog

## 0.1.0 — 2026-11-21

Primera versión funcional / First working release.

### Añadido / Added
- **Audio:** captura con `sounddevice`, VAD por energía con calibración de ruido e histéresis,
  grabador con pre-roll, corte por silencio, duración máxima y cancelación. Fuente WAV (`--from-wav`).
- **Push-to-talk:** atajo global (`pynput`, modos `toggle`/`hold`) con fallback a ENTER en Wayland o sin entorno gráfico.
- **STT offline:** `faster-whisper` int8 en CPU, carga perezosa, vocabulario técnico como `initial_prompt`,
  `HF_HUB_OFFLINE=1` en ejecución. Gestor de modelos (`models download|list|remove`).
- **Contexto:** rama, upstream, ahead/behind, archivos modificados y recientes, tipo de proyecto, herramientas instaladas (caché con TTL).
- **Intenciones:** gramática determinista ES/EN (~30 reglas) con normalización, tolerancia a errores de Whisper,
  coincidencia aproximada y alias por proyecto. Motor híbrido con `ollama`/`claude -p` opcionales.
- **Escudo de seguridad:** clasificación SAFE / CONFIRM / BLOCKED sobre el argv final; confirmación por voz con timeout.
- **Ejecución y delegación:** subprocess sin shell (salida UTF-8), lanzamiento en segundo plano, prompts estructurados
  al portapapeles, a `claude -p` (por stdin) o a la salida.
- **Voz de respuesta:** Piper / pyttsx3 (SAPI5, espeak-ng) / espeak-ng en un hilo de trabajo que no bloquea.
- **HUD** con `rich` e icono de bandeja opcional.
- **CLI:** `listen`, `interpret`, `models`, `doctor [--bench --mic-test]`, `config show|path|init|set|edit`.
- ~370 tests deterministas (audio sintético, fakes de micrófono, Whisper, LLM y ejecución) + tests opcionales con el modelo real.
- CI en GitHub Actions (Windows + Linux, Python 3.10 y 3.12).
