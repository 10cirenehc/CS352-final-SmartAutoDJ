"""Fade curves and EQ-style filtering for the transition region.

Two building blocks:

  * **Fade curves** — linear (tier-1 baseline) or equal-power (tier-2+). Equal
    power uses sin/cos so ``fade_out**2 + fade_in**2 == 1`` and the summed
    loudness stays roughly constant through the crossfade (Loudness topic).
  * **Bass swap** — instead of a time-varying filter, split each track into a
    low and a high band with complementary Butterworth filters, then hand the
    low band off from A to B with a short *equal-power* crossfade centred on the
    swap point. Only one bassline is dominant at a time (no two basslines
    fighting) but the **total** low-band energy stays ~constant through the
    swap — never dropping to silence — which is the classic DJ EQ move
    (Convolution & Filtering). An earlier linear schedule sent both basslines to
    zero at the midpoint, producing an audible bass dropout; the equal-power
    handoff fixes that.

``process_overlap`` returns the two region signals already fully processed
(fades + EQ applied) so ``mix`` only has to sum them.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .types import TransitionPlan


def fade_curves(
    n: int, shape: str = "equal_power", sharpness: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(fade_out, fade_in)`` gain envelopes of length ``n``.

    ``shape="linear"`` ramps straight; ``shape="equal_power"`` uses cos/sin so
    constant power is preserved across the overlap. ``sharpness > 1`` warps time
    toward the midpoint (tanh) so the crossover is **quicker / less gradual** —
    the two tracks hold, then swap fast (a punchier, more DJ-like cut).
    """
    if n <= 1:
        return np.ones(max(n, 0)), np.ones(max(n, 0))
    t = np.linspace(0.0, 1.0, n)
    if sharpness and sharpness > 1.0:
        k = float(sharpness)
        t = 0.5 * (1.0 + np.tanh(k * (t - 0.5)) / np.tanh(k * 0.5))
    if shape == "linear":
        return (1.0 - t), t
    # equal-power (constant-power) crossfade
    return np.cos(t * np.pi / 2.0), np.sin(t * np.pi / 2.0)


def bass_swap_curves(
    n: int, center: float = 0.5, width: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(bass_out, bass_in)`` low-band gains for an equal-power swap.

    A's bass holds full, then hands off to B's bass over a short window of
    fractional ``width`` centred at ``center``. Inside the window the gains
    follow cos/sin so ``bass_out**2 + bass_in**2 == 1`` — total low-band energy
    stays ~constant (no midpoint dropout). Outside it, exactly one bass is on.
    """
    if n <= 1:
        return np.ones(max(n, 0)), np.zeros(max(n, 0))
    t = np.linspace(0.0, 1.0, n)
    half = max(width, 1e-6) / 2.0
    lo, hi = center - half, center + half
    # Local position within the handoff window, clipped to [0, 1].
    u = np.clip((t - lo) / (hi - lo), 0.0, 1.0)
    bass_out = np.cos(u * np.pi / 2.0)
    bass_in = np.sin(u * np.pi / 2.0)
    return bass_out.astype(np.float32), bass_in.astype(np.float32)


def split_bands(
    y: np.ndarray, sr: int, cutoff: float = 200.0, order: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``y`` into ``(low, high)`` bands at ``cutoff`` Hz.

    The low band is a zero-phase Butterworth low-pass (``sosfiltfilt``, no phase
    smearing); the high band is its exact complement ``y - low``. Defining the
    high band this way guarantees perfect reconstruction (``low + high == y``),
    so equal band gains reproduce the original signal — important for a clean
    bass swap.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.size <= order * 3:  # too short to filter; treat as all-low
        return y.copy(), np.zeros_like(y)
    wn = np.clip(cutoff / (sr / 2.0), 1e-4, 0.99)
    sos_low = butter(order, wn, btype="low", output="sos")
    low = sosfiltfilt(sos_low, y).astype(np.float32)
    high = (y - low).astype(np.float32)
    return low, high


def process_overlap(
    a_tail: np.ndarray, b_head: np.ndarray, sr: int, plan: TransitionPlan
) -> tuple[np.ndarray, np.ndarray]:
    """Apply fades (+ optional bass swap) to the two overlapping segments.

    Returns ``(a_proc, b_proc)`` of equal length, ready to be summed by ``mix``.
    """
    n = int(min(len(a_tail), len(b_head)))
    a_tail = np.asarray(a_tail[:n], dtype=np.float32)
    b_head = np.asarray(b_head[:n], dtype=np.float32)
    fade_out, fade_in = fade_curves(n, plan.fade_shape, getattr(plan, "fade_sharpness", 1.0))

    eq = plan.eq_params or {}
    if eq.get("kind") != "bass_swap":
        # Tier-1 style: plain volume crossfade, full-band.
        return a_tail * fade_out, b_head * fade_in

    cutoff = float(eq.get("cutoff_hz", 200.0))
    order = int(eq.get("order", 4))
    center = float(eq.get("swap_center", 0.5))
    width = float(eq.get("swap_width", 0.2))
    a_low, a_high = split_bands(a_tail, sr, cutoff, order)
    b_low, b_high = split_bands(b_head, sr, cutoff, order)

    # Equal-power low-band handoff: A's bass hands off to B's over a short window
    # so the *total* bass never drops out (the old linear schedule crossed both
    # through zero at the midpoint -> hollow dropout).
    bass_out, bass_in = bass_swap_curves(n, center, width)

    a_proc = a_low * bass_out + a_high * fade_out
    b_proc = b_low * bass_in + b_high * fade_in
    return a_proc.astype(np.float32), b_proc.astype(np.float32)
