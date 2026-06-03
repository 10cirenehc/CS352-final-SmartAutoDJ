"""Shared test fixtures: synthetic click tracks at known tempos.

Using synthetic audio (not real songs) lets the analyze -> align -> mix path be
tested deterministically and without shipping copyrighted audio, per the
project's de-risk rule ("test each module on short, consistent clips first").
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

SR = 22050  # lower SR keeps tests fast; pipeline works at any SR


def make_click_track(bpm: float, duration: float = 12.0, sr: int = SR, seed: int = 0) -> np.ndarray:
    """A metronome-like click track in 4/4 with accented downbeats.

    Each beat is a short decaying noise burst; every 4th beat (the downbeat) is
    louder, which is what the librosa downbeat-phase heuristic keys on. A faint
    tonal bed is added so the signal isn't near-silent between clicks.
    """
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    y = 0.01 * np.sin(2 * np.pi * 110 * np.arange(n) / sr).astype(np.float32)
    beat_dur = 60.0 / bpm
    burst_len = int(0.04 * sr)
    env = np.exp(-np.linspace(0, 10, burst_len)).astype(np.float32)
    n_beats = int(duration / beat_dur)
    for k in range(n_beats):
        i = int(k * beat_dur * sr)
        gain = 1.0 if (k % 4 == 0) else 0.6  # accent downbeats
        burst = gain * rng.standard_normal(burst_len).astype(np.float32) * env
        end = min(i + burst_len, n)
        y[i:end] += burst[: end - i]
    peak = float(np.max(np.abs(y))) or 1.0
    return (y / peak * 0.9).astype(np.float32)


@pytest.fixture
def click_tracks(tmp_path):
    """Write two click tracks (120 and 124 BPM) to disk; return their paths.

    124/120 = 1.033, inside the default 10% tempo tolerance, so tier-2 stretch
    engages — exercising the time-stretch path.
    """
    a_path = tmp_path / "a_120.wav"
    b_path = tmp_path / "b_124.wav"
    sf.write(a_path, make_click_track(120.0, seed=1), SR)
    sf.write(b_path, make_click_track(124.0, seed=2), SR)
    return str(a_path), str(b_path), SR
