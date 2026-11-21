"""Generate spoken command WAVs with the system's offline TTS voice.

Windows: SAPI5 via pyttsx3 · Linux: espeak-ng. Useful for demos and for
``pytest -m model`` style checks without speaking into a microphone.

Usage: python scripts/gen_samples.py [out_dir] [--lang es|en]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PHRASES = {
    "es": {
        "tests": "Corre los tests",
        "estado": "Estado de git",
        "push_forzado": "Haz push forzado",
        "commit": "Haz commit con mensaje arregla el parser",
        "rama": "Crea una rama llamada feature login",
        "claude": "Claude, agrega validación al módulo de configuración",
    },
    "en": {
        "tests": "Run the tests",
        "status": "Git status",
        "force_push": "Force push",
        "claude": "Claude, add validation to the config module",
    },
}


def synthesize_all(out: Path, lang: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    phrases = PHRASES[lang]
    if sys.platform == "win32":
        import pyttsx3

        engine = pyttsx3.init()
        tag = "ES-" if lang == "es" else "EN-"
        for voice in engine.getProperty("voices"):
            if tag in voice.id.upper():
                engine.setProperty("voice", voice.id)
                break
        engine.setProperty("rate", 170)
        for name, text in phrases.items():
            engine.save_to_file(text, str(out / f"{name}.wav"))
        engine.runAndWait()
    else:
        exe = shutil.which("espeak-ng") or shutil.which("espeak")
        if exe is None:
            sys.exit("espeak-ng not found: sudo apt install espeak-ng")
        for name, text in phrases.items():
            subprocess.run([exe, "-v", lang, "-w", str(out / f"{name}.wav"), text], check=True)  # noqa: S603
    for name in phrases:
        print(out / f"{name}.wav")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="samples")
    ap.add_argument("--lang", choices=["es", "en"], default="es")
    args = ap.parse_args()
    synthesize_all(Path(args.out), args.lang)


if __name__ == "__main__":
    main()
