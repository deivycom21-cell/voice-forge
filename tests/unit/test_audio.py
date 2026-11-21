"""Audio pipeline tests with deterministic synthetic signals (no microphone)."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from voice_forge.audio import synth
from voice_forge.audio.recorder import Recorder
from voice_forge.audio.sources import ArraySource, MicSource, WavFileSource, list_input_devices
from voice_forge.audio.vad import SILENCE_DB, EnergyVAD, rms_db
from voice_forge.audio.wav import load_for_stt, read_wav, resample, write_wav

SR = synth.SAMPLE_RATE


# --------------------------------------------------------------------------- synth / wav


def test_synth_is_deterministic() -> None:
    assert np.array_equal(synth.noise(100, seed=3), synth.noise(100, seed=3))
    assert not np.array_equal(synth.noise(100, seed=3), synth.noise(100, seed=4))
    assert len(synth.tone(250)) == 4000
    assert synth.speechlike(300).dtype == np.float32
    assert len(synth.mix(synth.tone(100), synth.tone(200))) == 3200
    assert len(synth.concat()) == 0


def test_rms_db_levels() -> None:
    assert rms_db(synth.silence(30)) == SILENCE_DB
    assert rms_db(np.zeros(0, dtype=np.float32)) == SILENCE_DB
    assert rms_db(synth.tone(300, amp=0.3)) == pytest.approx(-13.5, abs=0.2)
    assert rms_db(synth.noise(1000, amp=synth.db_to_amp(-45))) == pytest.approx(-45, abs=0.5)


def test_wav_roundtrip(tmp_path: Path) -> None:
    audio = synth.tone(500, amp=0.5)
    path = write_wav(tmp_path / "a.wav", audio)
    back, sr = read_wav(path)
    assert sr == SR
    assert np.max(np.abs(back - audio)) < 1e-3


def test_load_for_stt_resamples_stereo_44k(tmp_path: Path) -> None:
    import wave

    t = np.arange(44100) / 44100
    left = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    stereo = np.stack([left, left], axis=1)
    p = tmp_path / "st.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(stereo.tobytes())
    audio = load_for_stt(p)
    assert len(audio) == SR
    assert rms_db(audio) == pytest.approx(-13.5, abs=0.3)


def test_read_8bit_and_resample_identity(tmp_path: Path) -> None:
    import wave

    p = tmp_path / "u8.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(8000)
        wf.writeframes(bytes([128, 255, 0, 128]))
    data, sr = read_wav(p)
    assert sr == 8000 and data[0] == 0.0 and data[1] > 0.9 and data[2] == -1.0
    assert len(resample(data, 8000, 16000)) == 8
    same = synth.tone(10)
    assert resample(same, SR, SR) is not None and len(resample(same, SR, SR)) == len(same)


# --------------------------------------------------------------------------- VAD


def frames(audio: synth.Audio, block: int = 480) -> Iterator[synth.Audio]:
    for i in range(0, len(audio), block):
        yield audio[i : i + block]


def test_vad_calibrates_and_detects() -> None:
    vad = EnergyVAD(calibration_frames=10)
    sig = synth.concat(synth.silence(300), synth.tone(300))
    flags = [vad.is_speech(f) for f in frames(sig)]
    assert flags[:10] == [False] * 10  # never speech while calibrating
    assert all(flags[10:])
    assert not vad.calibrating


def test_vad_noise_floor_avoids_false_positive() -> None:
    vad = EnergyVAD(abs_threshold_db=-50)
    sig = synth.noise(3000, amp=synth.db_to_amp(-45), seed=7)
    assert not any(vad.is_speech(f) for f in frames(sig))
    assert vad.threshold_db > -45


def test_vad_hysteresis_keeps_speech_slightly_below_threshold() -> None:
    vad = EnergyVAD(calibration_frames=0, abs_threshold_db=-30, hysteresis_db=6)
    assert vad.is_speech(synth.tone(30, amp=0.3))  # -13.5 dB
    assert vad.is_speech(synth.tone(30, amp=synth.db_to_amp(-33) * 1.414))  # -33 dB rms > -36
    assert not vad.is_speech(synth.silence(30))
    assert not vad.is_speech(
        synth.tone(30, amp=synth.db_to_amp(-33) * 1.414)
    )  # needs -30 to re-enter


# --------------------------------------------------------------------------- Recorder


def record(audio: synth.Audio, **kw: Any) -> Any:
    rec_kw = {k: kw.pop(k) for k in list(kw) if k in ("stop", "cancel", "start_timeout_s")}
    return Recorder(**kw).record(ArraySource(audio), **rec_kw)


def test_cuts_after_trailing_silence_at_expected_point() -> None:
    sig = synth.concat(synth.silence(500), synth.tone(1000), synth.silence(800))
    res = record(sig)
    assert res.reason == "silence"
    # pre-roll starts 300 ms before the tone; keeps 200 ms (+1 frame) of trailing silence
    assert res.duration_s == pytest.approx(1.53, abs=0.031)
    assert res.elapsed_ms == pytest.approx(2220, abs=30)


def test_silence_only_produces_no_segment() -> None:
    res = record(synth.silence(3000), start_timeout_s=1.0)
    assert res.reason == "no_speech" and res.audio is None and not res.has_audio
    assert res.elapsed_ms == pytest.approx(1000, abs=30)


def test_short_click_is_ignored() -> None:
    sig = synth.concat(synth.silence(500), synth.tone(40), synth.silence(1500))
    res = record(sig, start_timeout_s=10)
    assert res.reason == "no_speech"


def test_short_pause_inside_phrase_does_not_cut() -> None:
    sig = synth.concat(synth.silence(400), synth.tone(600), synth.silence(300), synth.tone(600),
                       synth.silence(1000))  # fmt: skip
    res = record(sig)
    assert res.reason == "silence"
    assert res.duration_s > 1.5  # both halves in one segment


def test_background_noise_with_speech() -> None:
    noise = synth.noise(3500, amp=synth.db_to_amp(-45), seed=1)
    speech = synth.concat(synth.silence(800), synth.speechlike(1200, seed=2), synth.silence(1500))
    res = record(synth.mix(noise, speech))
    assert res.reason == "silence"
    assert 1.2 <= res.duration_s <= 1.9


def test_max_duration() -> None:
    res = record(synth.tone(40_000), max_seconds=5)
    assert res.reason == "max_duration"
    assert res.duration_s == pytest.approx(5.0, abs=0.031)


def test_source_end_while_speaking() -> None:
    res = record(synth.concat(synth.silence(400), synth.tone(800)))
    assert res.reason == "source_end" and res.has_audio


class StopAfter:
    """Source that sets an event after N frames."""

    def __init__(self, audio: synth.Audio, n: int, event: threading.Event) -> None:
        self.inner = ArraySource(audio)
        self.sample_rate = SR
        self.n = n
        self.event = event
        self.closed = False

    def frames(self) -> Iterator[synth.Audio]:
        for i, f in enumerate(self.inner.frames()):
            if i == self.n:
                self.event.set()
            yield f

    def close(self) -> None:
        self.closed = True


def test_stop_event_finalizes_with_audio() -> None:
    stop = threading.Event()
    sig = synth.concat(synth.silence(300), synth.tone(3000))
    res = Recorder().record(StopAfter(sig, 40, stop), stop=stop)
    assert res.reason == "stopped" and res.has_audio


def test_stop_before_speech_confirmed_keeps_preroll_if_heard() -> None:
    stop = threading.Event()
    sig = synth.concat(synth.silence(300), synth.tone(3000))
    res = Recorder(min_speech_ms=500).record(StopAfter(sig, 12, stop), stop=stop)
    assert res.reason == "stopped" and res.has_audio


def test_cancel_discards() -> None:
    cancel = threading.Event()
    sig = synth.concat(synth.silence(300), synth.tone(3000))
    res = Recorder().record(StopAfter(sig, 20, cancel), cancel=cancel)
    assert res.reason == "cancelled" and res.audio is None


def test_level_callback_per_frame() -> None:
    levels: list[float] = []
    Recorder(on_level=levels.append).record(ArraySource(synth.silence(300)), start_timeout_s=0.3)
    assert len(levels) == 10


# --------------------------------------------------------------------------- sources


def test_array_source_pads_last_block_and_closes() -> None:
    src = ArraySource(synth.tone(100), block_ms=30)
    blocks = list(src.frames())
    assert len(blocks) == 4 and all(len(b) == 480 for b in blocks)
    src2 = ArraySource(synth.tone(1000))
    it = src2.frames()
    next(it)
    src2.close()
    assert list(it) == []


def test_wav_file_source_adds_tail_silence(tmp_path: Path) -> None:
    p = write_wav(tmp_path / "x.wav", synth.tone(300))
    src = WavFileSource(p, tail_silence_ms=300)
    assert len(src.audio) == pytest.approx(SR * 0.6, abs=1)


class FakeStream:
    instances: list[FakeStream] = []

    def __init__(self, *, samplerate: int, channels: int, dtype: str, blocksize: int, device: Any,
                 callback: Any, audio: synth.Audio) -> None:  # fmt: skip
        assert (samplerate, channels, dtype) == (16000, 1, "float32")
        self.blocksize = blocksize
        self.device = device
        self.callback = callback
        self.audio = audio
        self.started = self.stopped = self.closed = False
        FakeStream.instances.append(self)

    def start(self) -> None:
        self.started = True
        for i in range(0, len(self.audio), self.blocksize):
            block = self.audio[i : i + self.blocksize].reshape(-1, 1)
            self.callback(block, len(block), None, "input overflow" if i == 0 else None)

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def fake_sd(audio: synth.Audio) -> SimpleNamespace:
    return SimpleNamespace(InputStream=lambda **kw: FakeStream(**kw, audio=audio))


def test_mic_source_with_fake_portaudio_records_and_closes_stream() -> None:
    sig = synth.concat(synth.silence(500), synth.tone(800), synth.silence(1000))
    mic = MicSource(sd_module=fake_sd(sig), device="3")
    res = Recorder().record(mic)
    assert res.reason == "silence" and res.has_audio
    stream = FakeStream.instances[-1]
    assert stream.started and stream.stopped and stream.closed
    assert stream.blocksize == 480 and stream.device == "3"


def test_mic_source_overflow_drops_oldest() -> None:
    mic = MicSource(sd_module=None, max_queue=5)
    for i in range(10):
        mic._callback(np.full((480, 1), i, dtype=np.float32), 480, None, None)
    assert mic.overflows == 5
    assert [int(mic._queue.get_nowait()[0]) for _ in range(5)] == [5, 6, 7, 8, 9]


def test_list_input_devices() -> None:
    sd = SimpleNamespace(query_devices=lambda: [
        {"name": "Mic", "max_input_channels": 1}, {"name": "Speakers", "max_input_channels": 0}])  # fmt: skip
    assert list_input_devices(sd) == [{"index": 0, "name": "Mic", "channels": 1}]

    def boom() -> None:
        raise OSError("PortAudio library not found")

    assert list_input_devices(SimpleNamespace(query_devices=boom)) == []
