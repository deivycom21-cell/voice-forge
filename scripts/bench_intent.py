"""Benchmark the deterministic intent parser.

Usage: python scripts/bench_intent.py [iterations]
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

from voice_forge.config import DEFAULT_ALIASES
from voice_forge.core.types import ContextSnapshot
from voice_forge.intent.rules import RuleParser

PHRASES = [
    "corre los tests",
    "estado de git",
    "haz push forzado",
    "commit con mensaje arregla el parser",
    "crea rama feature barra login",
    "claude, refactoriza el módulo de audio",
    "estao de git",  # fuzzy hit
    "frase que no significa nada para el parser",  # fuzzy miss
]


def main() -> None:
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    parser = RuleParser(DEFAULT_ALIASES)
    ctx = ContextSnapshot(cwd=Path.cwd(), markers=("pyproject.toml",), tools=frozenset({"pytest"}))
    print(f"{'phrase':<48} {'mean µs':>9} {'p99 µs':>9} {'max µs':>9}")
    for phrase in PHRASES:
        samples = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            parser.parse(phrase, ctx)
            samples.append((time.perf_counter_ns() - t0) / 1000)
        samples.sort()
        p99 = samples[int(len(samples) * 0.99) - 1]
        print(f"{phrase[:47]:<48} {statistics.mean(samples):9.1f} {p99:9.1f} {samples[-1]:9.1f}")


if __name__ == "__main__":
    main()
