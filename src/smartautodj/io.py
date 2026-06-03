"""Audio I/O: load to mono float32 at a canonical sample rate, save WAV, normalize.

Course topic — Loudness & Amplitude: ``peak_normalize`` works in dBFS so level
adjustments are expressed the way loudness is perceived/specified, not as raw
linear gains.
"""

from __future__ import annotations

import os

import librosa
import numpy as np
import soundfile as sf

from . import DEFAULT_SR


def load_audio(path: str, sr: int = DEFAULT_SR) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 resampled to ``sr``.

    Returns ``(y, sr)`` where ``y`` is a 1-D float32 array in roughly [-1, 1].
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"audio file not found: {path}")
    # librosa.load handles decoding (via soundfile/audioread+ffmpeg), downmix
    # to mono, and resampling to our canonical pipeline rate in one call.
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), sr


def save_wav(path: str, y: np.ndarray, sr: int = DEFAULT_SR) -> str:
    """Write ``y`` to a 16-bit PCM WAV, creating parent dirs. Returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    y = np.asarray(y, dtype=np.float32)
    # Guard against clipping on export: only scale down if we exceed full-scale.
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak
    sf.write(path, y, sr, subtype="PCM_16")
    return path


def peak_normalize(y: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    """Scale ``y`` so its peak sits at ``target_dbfs`` (decibels full-scale).

    A silent signal is returned unchanged. ``target_dbfs`` should be <= 0.
    """
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 0.0:
        return y
    target_amp = 10.0 ** (target_dbfs / 20.0)
    return (y * (target_amp / peak)).astype(np.float32)


def to_samples(t_sec: float, sr: int = DEFAULT_SR) -> int:
    """Convert a time in seconds to an integer sample index."""
    return int(round(float(t_sec) * sr))
