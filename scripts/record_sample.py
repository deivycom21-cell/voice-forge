"""Record one utterance from the microphone (VAD-trimmed) into a WAV file.

Handy to debug recognition locally and to replay it later with
``voice-forge listen --from-wav sample.wav --dry-run``. The audio never leaves your machine.

Usage: python scripts/record_sample.py [out.wav]
"""

from __future__ import annotations

import sys
from pathlib import Path

from voice_forge.audio.recorder import Recorder
from voice_forge.audio.sources import MicSource
from voice_forge.audio.wav import write_wav


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "sample.wav")
    print("Habla ahora… / Speak now… (stops after ~0.7 s of silence)")
    result = Recorder(start_timeout_s=10).record(MicSource())
    if not result.has_audio or result.audio is None:
        print(f"Nada grabado ({result.reason}).")
        sys.exit(1)
    write_wav(out, result.audio)
    print(f"Guardado {out} · {result.duration_s:.2f} s · motivo de corte: {result.reason}")


if __name__ == "__main__":
    main()
