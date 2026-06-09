"""Post-processing: make the additive bridge *sit* in the transition like a DJ.

Loudness-matching alone makes the bridge audible but it still piles on top of the
two full tracks — crowded, muffled, "not like a transition". Real DJs create
*space* for a transition element. Three composable techniques do that here, in
the order a mix engineer would apply them:

1. ``loudness_match`` — stage the bridge by **RMS** (not peak) relative to the
   program, so a transient air-horn/stab and a sustained riser both sit at a
   predictable, audible level (Loudness topic).
2. ``soft_limit`` — tanh peak-limit the bridge so its transients don't dominate
   the headroom budget and force the whole mix quieter (the "sounds low" issue).
3. ``sidechain_duck`` — duck the two songs *under* the bridge's envelope so the
   transition **breathes** instead of stacking three full layers.
4. ``spectral_carve`` (optional) — a per-frequency sidechain: dip the program in
   the bands where the bridge is energetic, so it's not muffled. This is the
   Fourier/spectrogram tie-in (STFT magnitude masking).

``integrate_bridge`` orchestrates them and returns the overlap-region signal plus
an info dict for the sidecar. All steps are gentle, commented signal processing —
a teaching artifact, not a black box.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import istft, lfilter, stft


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def loudness_match(layer: np.ndarray, reference: np.ndarray, rel_db: float = -3.0) -> np.ndarray:
    """Scale ``layer`` so its RMS sits ``rel_db`` relative to ``reference``'s RMS.

    Staging by loudness (not peak) makes any element — transient or sustained —
    sit at a predictable, audible level; peak-normalizing a high-crest stab leaves
    it ~10 dB low in RMS and masked.
    """
    layer = np.asarray(layer, dtype=np.float32)
    lr, rr = _rms(layer), _rms(reference)
    if lr <= 1e-9 or rr <= 1e-9:
        return layer
    target = rr * (10.0 ** (rel_db / 20.0))
    return (layer * (target / lr)).astype(np.float32)


def soft_limit(x: np.ndarray, ceiling: float) -> np.ndarray:
    """Smooth (tanh) peak limit: ``y = ceiling * tanh(x / ceiling)``.

    Small signals pass ~linearly; peaks asymptote to ``±ceiling``. This caps the
    bridge's transients (so they don't blow the headroom budget and force the
    whole transition quieter) while preserving most of its RMS.
    """
    x = np.asarray(x, dtype=np.float32)
    c = max(float(ceiling), 1e-4)
    return (c * np.tanh(x / c)).astype(np.float32)


def envelope(x: np.ndarray, sr: int, smooth_ms: float = 80.0) -> np.ndarray:
    """One-pole (vectorised) amplitude follower on ``|x|`` — the ducking control."""
    x = np.abs(np.asarray(x, dtype=np.float64))
    if x.size == 0:
        return x.astype(np.float32)
    a = float(np.exp(-1.0 / max(sr * smooth_ms / 1000.0, 1.0)))
    env = lfilter([1.0 - a], [1.0, -a], x)
    return env.astype(np.float32)


def sidechain_duck(
    program: np.ndarray, control: np.ndarray, sr: int,
    depth_db: float = 6.0, smooth_ms: float = 80.0,
) -> np.ndarray:
    """Duck ``program`` by up to ``depth_db`` following ``control``'s envelope.

    Where the bridge (control) is loudest the program dips to ``-depth_db``;
    where it's silent the program is untouched — so the transition opens up.
    """
    program = np.asarray(program, dtype=np.float32)
    env = envelope(control, sr, smooth_ms)
    peak = float(np.max(env)) if env.size else 0.0
    if peak <= 1e-9:
        return program
    env_norm = env / peak
    floor = 10.0 ** (-abs(depth_db) / 20.0)
    gain = 1.0 - (1.0 - floor) * env_norm
    return (program * gain).astype(np.float32)


def spectral_carve(
    program: np.ndarray, control: np.ndarray, sr: int,
    depth_db: float = 6.0, n_fft: int = 2048,
) -> np.ndarray:
    """Per-frequency sidechain: dip ``program`` bins where ``control`` is energetic.

    STFT both signals; wherever the bridge has strong magnitude in a time-frequency
    cell, attenuate the program there (down to ``-depth_db``) so the bridge isn't
    masked and the result isn't muffled. iSTFT back. The Fourier/spectrogram topic
    made audible.
    """
    program = np.asarray(program, dtype=np.float32)
    if program.size < n_fft:
        return program
    ctrl = np.asarray(control, dtype=np.float32)
    if ctrl.size < program.size:
        ctrl = np.pad(ctrl, (0, program.size - ctrl.size))
    _, _, P = stft(program, fs=sr, nperseg=n_fft, noverlap=n_fft // 2)
    _, _, C = stft(ctrl[: program.size], fs=sr, nperseg=n_fft, noverlap=n_fft // 2)
    cmag = np.abs(C)
    cmax = float(cmag.max()) if cmag.size else 0.0
    if cmax <= 1e-9:
        return program
    cnorm = cmag / cmax  # 0..1 per time-frequency cell
    floor = 10.0 ** (-abs(depth_db) / 20.0)
    gain = 1.0 - (1.0 - floor) * cnorm  # carve where the bridge is strong
    _, y = istft(P[:, : gain.shape[1]] * gain, fs=sr, nperseg=n_fft, noverlap=n_fft // 2)
    y = np.asarray(y, dtype=np.float32)
    if y.size >= program.size:
        return y[: program.size]
    return np.pad(y, (0, program.size - y.size))


def integrate_bridge(
    program: np.ndarray, bridge: np.ndarray, sr: int, *,
    rel_db: float = -3.0, duck_db: float = 6.0, limit: bool = True, spectral: bool = True,
) -> tuple[np.ndarray, dict]:
    """Blend ``bridge`` into the overlap ``program`` (a_proc + b_proc).

    Order: loudness-match the bridge -> soft-limit its peaks -> duck the program
    under it -> (optional) spectral carve -> sum. Returns ``(overlap, info)``.
    """
    program = np.asarray(program, dtype=np.float32)
    bridge = np.asarray(bridge, dtype=np.float32)

    bridge = loudness_match(bridge, program, rel_db)
    if limit:
        # Cap bridge peaks near the program's peak so they don't dominate headroom.
        ceiling = max(float(np.max(np.abs(program))) if program.size else 0.0, 1e-3)
        bridge = soft_limit(bridge, ceiling)

    ducked = program
    if duck_db > 0:
        ducked = sidechain_duck(program, bridge, sr, depth_db=duck_db)
    if spectral:
        ducked = spectral_carve(ducked, bridge, sr, depth_db=duck_db)

    overlap = (ducked + bridge).astype(np.float32)
    info = {
        "bridge_rel_db": rel_db,
        "bridge_duck_db": duck_db if duck_db > 0 else None,
        "bridge_limit": bool(limit),
        "bridge_spectral_carve": bool(spectral),
        "bridge_peak": round(float(np.max(np.abs(bridge))) if bridge.size else 0.0, 4),
    }
    return overlap, info
