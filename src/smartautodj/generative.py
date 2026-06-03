"""Tier-3 auxiliary "bridge" layer.

In the full system this layer is a short AI clip (MusicGen / Stable Audio)
generated in Colab. For this build cycle we synthesize a stand-in *procedurally*
so the whole trim/normalize/align/mix path is exercised and reproducible with no
external dependency. ``prepare_clip`` is the seam: real Colab clips flow through
the exact same function (trim -> normalize -> length-match) before mixing.

The bridge is **additive** — it never replaces the algorithmic transition.

Two stand-in kinds, both beat-aware:
  * ``riser`` — swelling filtered noise + a rising sine sweep that peaks at the
    drop (the moment B takes over). A staple EDM transition effect.
  * ``drum_fill`` — short noise-burst hits on the beat grid that get denser
    toward the end, like a fill leading into the downbeat.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from . import DEFAULT_SR
from .io import peak_normalize


def make_placeholder_bridge(
    kind: str,
    dur: float,
    sr: int = DEFAULT_SR,
    bpm: float = 120.0,
    seed: int = 0,
    gain_dbfs: float = -9.0,
) -> np.ndarray:
    """Synthesize a beat-aware bridge of length ``dur`` seconds.

    Deterministic given ``seed`` (project convention: reproducible runs).
    Returned at a modest level (``gain_dbfs``) so it sits under the mix.
    """
    rng = np.random.default_rng(seed)
    n = max(int(round(dur * sr)), 1)
    if kind == "drum_fill":
        y = _drum_fill(n, sr, bpm, rng)
    else:  # default / "riser"
        y = _riser(n, sr, rng)
    return peak_normalize(y, target_dbfs=gain_dbfs)


def prepare_clip(
    clip: np.ndarray, sr: int, dur: float, gain_dbfs: float = -9.0
) -> np.ndarray:
    """Trim/pad ``clip`` to ``dur`` seconds and normalize — the shared seam.

    Real Colab-generated clips use this same function so the local pipeline
    treats them as static assets it only trims/normalizes/length-matches.
    """
    clip = np.asarray(clip, dtype=np.float32)
    n = int(round(dur * sr))
    if clip.size >= n:
        clip = clip[:n]
    else:
        clip = np.pad(clip, (0, n - clip.size))
    # Short fades in/out so the bridge doesn't click at its edges.
    clip = _edge_fade(clip, sr)
    return peak_normalize(clip, target_dbfs=gain_dbfs)


# --------------------------------------------------------------------------- #
# Synthesis primitives
# --------------------------------------------------------------------------- #
def _riser(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Filtered-noise swell + rising sine sweep, peaking at the end."""
    t = np.linspace(0.0, 1.0, n)
    # Amplitude swells quadratically into the drop.
    amp = t ** 2

    # Noise that brightens over time: crossfade a dull (low-passed) copy into a
    # bright (high-passed) copy so perceived "energy" rises toward the drop.
    noise = rng.standard_normal(n).astype(np.float32)
    low = _filt(noise, sr, 800.0, "low")
    high = _filt(noise, sr, 2000.0, "high")
    bright = low * (1.0 - t) + high * t
    noise_layer = bright * amp

    # Rising sine sweep (pitch glides up) layered underneath.
    f0, f1 = 200.0, 2000.0
    freq = f0 * (f1 / f0) ** t  # exponential glide
    phase = np.cumsum(2.0 * np.pi * freq / sr)
    sweep = np.sin(phase).astype(np.float32) * amp * 0.5

    return (0.7 * noise_layer + 0.3 * sweep).astype(np.float32)


def _drum_fill(n: int, sr: int, bpm: float, rng: np.random.Generator) -> np.ndarray:
    """Noise-burst hits on the beat grid, getting denser toward the end."""
    y = np.zeros(n, dtype=np.float32)
    beat_dur = 60.0 / max(bpm, 1.0)
    dur = n / sr

    # Subdivide more finely as the fill progresses: quarter -> eighth -> sixteenth.
    hits = []
    t = 0.0
    while t < dur:
        prog = t / dur if dur > 0 else 0.0
        subdiv = 1 if prog < 0.5 else (2 if prog < 0.8 else 4)
        step = beat_dur / subdiv
        hits.append(t)
        t += step

    for ht in hits:
        i = int(round(ht * sr))
        burst_len = int(0.05 * sr)
        env = np.exp(-np.linspace(0, 12, burst_len)).astype(np.float32)
        burst = rng.standard_normal(burst_len).astype(np.float32) * env
        end = min(i + burst_len, n)
        y[i:end] += burst[: end - i]
    return y


def _filt(y: np.ndarray, sr: int, cutoff: float, btype: str, order: int = 4) -> np.ndarray:
    if y.size <= order * 3:
        return y.astype(np.float32)
    wn = np.clip(cutoff / (sr / 2.0), 1e-4, 0.99)
    sos = butter(order, wn, btype=btype, output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def _edge_fade(y: np.ndarray, sr: int, ms: float = 10.0) -> np.ndarray:
    k = min(int(sr * ms / 1000.0), len(y) // 2)
    if k <= 0:
        return y
    ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
    y = y.copy()
    y[:k] *= ramp
    y[-k:] *= ramp[::-1]
    return y
