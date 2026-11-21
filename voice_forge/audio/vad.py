"""Energy based voice activity detection.

Deterministic and dependency-free so it can be tested with synthetic signals.

Algorithm
---------
* Level of each frame in dBFS (RMS).
* The first ``calibration_frames`` estimate the noise floor (25th percentile),
  capped at ``max_noise_floor_db`` so that talking right away still works.
  No speech is reported while calibrating (the recorder's pre-roll keeps that audio).
* Speech threshold = ``max(abs_threshold_db, noise_floor + margin_db)``.
* Hysteresis: once in speech, stay in speech while level > threshold - hysteresis_db.
* While not in speech the noise floor slowly tracks quieter levels.
"""

from __future__ import annotations

import math

import numpy as np

from voice_forge.audio.synth import Audio

SILENCE_DB = -200.0


def rms_db(frame: Audio) -> float:
    if len(frame) == 0:
        return SILENCE_DB
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
    return 20.0 * math.log10(rms) if rms > 1e-10 else SILENCE_DB


class EnergyVAD:
    def __init__(
        self,
        abs_threshold_db: float = -50.0,
        margin_db: float = 12.0,
        calibration_frames: int = 10,
        hysteresis_db: float = 3.0,
        max_noise_floor_db: float = -40.0,
    ) -> None:
        self.abs_threshold_db = abs_threshold_db
        self.margin_db = margin_db
        self.calibration_frames = calibration_frames
        self.hysteresis_db = hysteresis_db
        self.max_noise_floor_db = max_noise_floor_db
        self.reset()

    def reset(self) -> None:
        self._calib: list[float] = []
        self.noise_floor_db: float | None = None if self.calibration_frames > 0 else SILENCE_DB
        self._in_speech = False
        self.last_level_db = SILENCE_DB

    @property
    def calibrating(self) -> bool:
        return self.noise_floor_db is None

    @property
    def threshold_db(self) -> float:
        floor = self.noise_floor_db if self.noise_floor_db is not None else SILENCE_DB
        return max(self.abs_threshold_db, floor + self.margin_db)

    def is_speech(self, frame: Audio) -> bool:
        level = rms_db(frame)
        self.last_level_db = level
        if self.noise_floor_db is None:
            self._calib.append(level)
            if len(self._calib) >= self.calibration_frames:
                floor = float(np.percentile(self._calib, 25))
                self.noise_floor_db = min(floor, self.max_noise_floor_db)
            return False
        threshold = self.threshold_db
        if self._in_speech:
            self._in_speech = level > threshold - self.hysteresis_db
        else:
            self._in_speech = level > threshold
            if not self._in_speech and level < threshold:
                # track the floor slowly (only quieter-than-threshold frames)
                self.noise_floor_db = min(
                    self.max_noise_floor_db, 0.95 * self.noise_floor_db + 0.05 * level
                )
        return self._in_speech
