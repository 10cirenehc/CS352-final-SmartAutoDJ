"""Tempo matching, time-stretching, and downbeat-aligned region selection.

This stage decides *where* the two tracks overlap and lines their bars up:

  1. ``decide_stretch`` — if A and B are within a tolerance, time-stretch B onto
     A's tempo (DJs only beat-match when tempos are close).
  2. ``select_transition_region`` — anchor the overlap on a **downbeat** near the
     end of A (preferring a section boundary) and the **first downbeat** of B,
     so bar 1 of B lands on bar 1 of the outgoing phrase.
  3. ``matched_beats_in_overlap`` — pair A/B beats inside the overlap so
     ``evaluate`` can measure beat-alignment error.

Course topics: Beat/Downbeat tracking, Self-Similarity (section-aware anchor).
"""

from __future__ import annotations

import librosa
import numpy as np

from .types import AnalysisResult, TransitionPlan

METER = 4  # assume 4/4 throughout (matches the librosa downbeat derivation)


def decide_stretch(bpm_a: float, bpm_b: float, tol: float = 0.10) -> float:
    """Return the time-stretch *rate* to apply to B so it plays at A's tempo.

    ``librosa.effects.time_stretch(y, rate=r)`` yields new_tempo = old_tempo * r,
    so the rate that moves B from ``bpm_b`` to ``bpm_a`` is ``bpm_a / bpm_b``.
    If the tempos differ by more than ``tol`` (fractional), stretching would be
    too audible/unnatural, so we leave B alone (rate 1.0).
    """
    if bpm_a <= 0 or bpm_b <= 0:
        return 1.0
    rate = bpm_a / bpm_b
    return rate if abs(rate - 1.0) <= tol else 1.0


def _refine_stretch(a: AnalysisResult, b: AnalysisResult, tol: float) -> float:
    """Stretch rate from actual median beat periods, not the rounded BPM.

    The tempo estimator reports a rounded global BPM, but we align on the beat
    *arrays*. Deriving the rate from the median inter-beat interval (IBI) of
    each track keeps the stretch consistent with the beats we actually measure
    error against, which removes slow drift across a long overlap. Falls back to
    the BPM-based estimate when there are too few beats, and is still gated by
    the same tempo tolerance.
    """
    if a.beats.size < 3 or b.beats.size < 3:
        return decide_stretch(a.bpm, b.bpm, tol)
    period_a = _beat_period(a.beats)
    period_b = _beat_period(b.beats)
    if period_a <= 0 or period_b <= 0:
        return decide_stretch(a.bpm, b.bpm, tol)
    rate = period_b / period_a  # stretching B by this rate makes its period == A's
    return rate if abs(rate - 1.0) <= tol else 1.0


def _beat_period(beats: np.ndarray) -> float:
    """Least-squares beat period: the slope of beat-index -> time.

    Fitting all beats (vs taking a median of consecutive differences) minimizes
    accumulated phase drift across the overlap when the tracker's beats are
    slightly jittery.
    """
    idx = np.arange(beats.size, dtype=float)
    slope, _ = np.polyfit(idx, np.asarray(beats, dtype=float), 1)
    return float(slope)


def apply_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Time-stretch ``y`` by ``rate`` (no-op when rate ~= 1)."""
    if abs(rate - 1.0) < 1e-3:
        return np.asarray(y, dtype=np.float32)
    return librosa.effects.time_stretch(np.asarray(y, dtype=np.float32), rate=rate)


def scale_times(times: np.ndarray, rate: float) -> np.ndarray:
    """Map event times into a stretched timeline: t -> t / rate."""
    if abs(rate - 1.0) < 1e-3:
        return np.asarray(times, dtype=float)
    return np.asarray(times, dtype=float) / rate


def select_transition_region(
    a: AnalysisResult,
    b: AnalysisResult,
    bars: int = 8,
    tempo_tol: float = 0.10,
    tier: int = 2,
) -> TransitionPlan:
    """Plan a beat-aligned transition from A into B.

    Returns a :class:`TransitionPlan`. ``stretch_ratio`` is the rate the caller
    should apply to B's audio (via :func:`apply_stretch`) before mixing; the
    plan's B-side times are already expressed in that stretched timeline.
    """
    rate = _refine_stretch(a, b, tempo_tol)
    target_bpm = a.bpm  # after (optional) stretch both sit at A's tempo

    # B's beats/downbeats in the post-stretch timeline.
    b_downbeats = scale_times(b.downbeats, rate)
    b_duration = b.duration / rate if abs(rate - 1.0) >= 1e-3 else b.duration

    bar_dur = METER * 60.0 / target_bpm if target_bpm > 0 else 2.0
    overlap = bars * bar_dur

    # Don't ask for more overlap than either track can supply.
    overlap = float(min(overlap, 0.9 * a.duration, 0.9 * b_duration))

    start_a = _choose_outgoing_downbeat(a, overlap)
    start_b = _choose_incoming_downbeat(b_downbeats)

    region_a = (start_a, min(start_a + overlap, a.duration))
    region_b = (start_b, min(start_b + overlap, b_duration))
    # Use the actually-available overlap (whichever side is shorter).
    overlap = float(min(region_a[1] - region_a[0], region_b[1] - region_b[0]))
    region_a = (start_a, start_a + overlap)
    region_b = (start_b, start_b + overlap)

    anchors_a = a.downbeats[
        (a.downbeats >= region_a[0] - 1e-6) & (a.downbeats <= region_a[1] + 1e-6)
    ]
    anchors_b = b_downbeats[
        (b_downbeats >= region_b[0] - 1e-6) & (b_downbeats <= region_b[1] + 1e-6)
    ]

    return TransitionPlan(
        tier=tier,
        overlap_sec=overlap,
        region_a=region_a,
        region_b=region_b,
        anchor_downbeats_a=anchors_a,
        anchor_downbeats_b=anchors_b,
        stretch_ratio=rate,
        fade_shape="equal_power",
        eq_params={"kind": "bass_swap", "cutoff_hz": 200.0, "order": 4},
        bridge=None,
    )


def baseline_plan(duration_a: float, duration_b: float, overlap_sec: float = 4.0) -> TransitionPlan:
    """Tier-1 plan: a fixed overlap of A's tail with B's head, no alignment."""
    overlap = float(min(overlap_sec, 0.9 * duration_a, 0.9 * duration_b))
    return TransitionPlan(
        tier=1,
        overlap_sec=overlap,
        region_a=(duration_a - overlap, duration_a),
        region_b=(0.0, overlap),
        stretch_ratio=1.0,
        fade_shape="linear",
        eq_params={},
        bridge=None,
    )


def matched_beats_in_overlap(
    a: AnalysisResult, b: AnalysisResult, plan: TransitionPlan
) -> tuple[np.ndarray, np.ndarray]:
    """Pair A/B beats that fall inside the overlap, on a common output clock.

    Both beat sets are shifted so the overlap starts at t=0: A beats by
    ``-region_a.start`` and (stretched) B beats by ``-region_b.start``. Each A
    beat is matched to its nearest B beat. Used by ``evaluate`` to compute
    beat-alignment error. Returns ``(a_times, b_times)`` arrays of equal length.
    """
    a0, a1 = plan.region_a
    b0, b1 = plan.region_b
    a_beats = a.beats[(a.beats >= a0) & (a.beats <= a1)] - a0
    b_beats = scale_times(b.beats, plan.stretch_ratio)
    b_beats = b_beats[(b_beats >= b0) & (b_beats <= b1)] - b0
    if a_beats.size == 0 or b_beats.size == 0:
        return np.array([]), np.array([])
    nearest = b_beats[np.argmin(np.abs(a_beats[:, None] - b_beats[None, :]), axis=1)]
    return a_beats, nearest


# --------------------------------------------------------------------------- #
# Anchor selection helpers
# --------------------------------------------------------------------------- #
def _choose_outgoing_downbeat(a: AnalysisResult, overlap: float) -> float:
    """Latest A downbeat that leaves ``overlap`` seconds before the end.

    Among feasible downbeats, prefer one near a section boundary (a musically
    meaningful exit point); otherwise take the latest feasible downbeat.
    """
    db = a.downbeats
    feasible = db[db <= a.duration - overlap + 1e-6]
    if feasible.size == 0:
        return float(max(0.0, a.duration - overlap))

    boundaries = _section_boundaries(a)
    if boundaries.size:
        # Restrict to a late-ish window, then snap to the nearest boundary.
        window = feasible[feasible >= feasible.max() - 2 * overlap]
        cand = window if window.size else feasible
        dists = np.min(np.abs(cand[:, None] - boundaries[None, :]), axis=1)
        if float(dists.min()) <= overlap / 2:  # only if a boundary is genuinely close
            return float(cand[int(np.argmin(dists))])
    return float(feasible.max())


def _choose_incoming_downbeat(b_downbeats: np.ndarray) -> float:
    """First downbeat of B (skip a leading downbeat at t=0 if a later one exists)."""
    if b_downbeats.size == 0:
        return 0.0
    nonzero = b_downbeats[b_downbeats > 1e-3]
    return float(nonzero[0]) if nonzero.size else float(b_downbeats[0])


def _section_boundaries(a: AnalysisResult) -> np.ndarray:
    starts = [s["start"] for s in a.sections] if a.sections else []
    return np.asarray(sorted(set(starts)), dtype=float)
