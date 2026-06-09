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

import warnings

import librosa
import numpy as np

from . import structure as structure_mod
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
    """Robust beat period: median of *inlier* inter-beat intervals.

    A least-squares slope of beat-index -> time looks appealing for jitter, but
    it is catastrophically wrong when the tracker **misses beats** (common on
    real music): a dropped beat leaves a ~2x gap, the index no longer matches
    ``k * period``, and the slope inflates — which once inverted our stretch
    direction on real songs. Instead we take the median IBI (robust to the
    occasional 2x/0.5x outlier from a missed/extra beat), then average only the
    intervals within [0.5x, 1.5x] of it to also smooth out small jitter.
    """
    beats = np.asarray(beats, dtype=float)
    diffs = np.diff(beats)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.0
    med = float(np.median(diffs))
    inliers = diffs[(diffs >= 0.5 * med) & (diffs <= 1.5 * med)]
    return float(np.mean(inliers)) if inliers.size else med


def apply_stretch(y: np.ndarray, rate: float, sr: int = 44100) -> np.ndarray:
    """Time-stretch ``y`` by ``rate`` (no-op when rate ~= 1).

    Prefers **RubberBand** (via ``pyrubberband``) — it is transient-aware and
    DJ-grade, avoiding the "phasiness"/transient-smearing that librosa's phase
    vocoder produces (librosa's own docs recommend RubberBand for quality).
    Falls back to librosa's phase vocoder if the ``rubberband`` CLI / wrapper is
    unavailable, so the pipeline still runs everywhere.

    ``pyrubberband.time_stretch(y, sr, rate)`` makes the output ``rate`` times
    *faster* (duration / rate), matching librosa's ``time_stretch`` convention.
    """
    y = np.asarray(y, dtype=np.float32)
    if abs(rate - 1.0) < 1e-3:
        return y
    try:
        import pyrubberband as pyrb

        out = pyrb.time_stretch(y, sr, rate)
        return np.asarray(out, dtype=np.float32)
    except Exception as exc:  # rubberband binary or wrapper missing
        warnings.warn(
            f"pyrubberband unavailable ({exc}); falling back to librosa phase "
            "vocoder (lower quality). Install rubberband: `brew install rubberband`."
        )
        return librosa.effects.time_stretch(y, rate=rate)


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
    cue_method: str = "energy",
    phrase_bars: int = 4,
    y_a: np.ndarray | None = None,
    y_b: np.ndarray | None = None,
    sr: int | None = None,
    cue_a: float | None = None,
    cue_b: float | None = None,
    fade_sharpness: float = 1.0,
    shape: str = "blend",
) -> TransitionPlan:
    """Plan a structure-aware, beat-aligned transition from A into B.

    The incoming point in B is chosen so we drop into a *high-energy* section
    (not B's quiet intro): ``cue_method="energy"`` takes the first phrase
    downbeat whose section has full energy (Phase 1); ``cue_method="novelty"``
    ranks phrase downbeats by fused novelty + energy + harmonic compatibility
    with A (Phase 2). See :mod:`smartautodj.structure`.

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

    if cue_method == "match" and y_a is not None and y_b is not None and sr:
        # Jointly pick A-exit + B-entry by beat-window cross-similarity + energy.
        start_a, start_b_orig, sel_b = structure_mod.choose_transition_pair(
            a, b, y_a, y_b, sr, overlap, phrase_bars=phrase_bars
        )
    else:
        start_a = _choose_outgoing_downbeat(a, overlap)
        # Choose B's switch point in B's ORIGINAL timeline (where its energy/novelty
        # features live), then map it into the post-stretch timeline.
        start_b_orig, sel_b = _choose_incoming_switch(a, b, overlap, cue_method, phrase_bars)
    # Manual cue override (seconds into each track) — snapped to the nearest
    # downbeat so it stays beat-aligned. Lets a user place the transition exactly
    # (e.g. on the chorus) when auto-detection can't find it.
    if cue_a is not None:
        start_a = _snap_downbeat(a.downbeats, cue_a, a.duration - overlap)
        sel_b = {**(sel_b or {}), "manual_cue_a": round(float(cue_a), 2)}
    if cue_b is not None:
        start_b_orig = _snap_downbeat(b.downbeats, cue_b, b.duration - overlap)
        sel_b = {**(sel_b or {}), "manual_cue_b": round(float(cue_b), 2)}
    start_b = float(scale_times(np.array([start_b_orig]), rate)[0])
    selection = {
        "cue_method": cue_method,
        "phrase_bars": phrase_bars,
        "incoming": sel_b,
        "key_a": getattr(a, "key", {}),
        "key_b": getattr(b, "key", {}),
        "harmonic_compatibility": round(
            structure_mod.key_compatibility(getattr(a, "key", {}), getattr(b, "key", {})), 3
        ),
    }

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
        shape=shape,
        fade_shape="equal_power",
        fade_sharpness=fade_sharpness,
        eq_params={"kind": "bass_swap", "cutoff_hz": 200.0, "order": 4},
        bridge=None,
        selection=selection,
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
def _snap_downbeat(downbeats: np.ndarray, t: float, t_max: float) -> float:
    """Nearest downbeat to ``t`` that still leaves room (<= ``t_max``)."""
    db = np.asarray(downbeats, dtype=float)
    db = db[db <= max(t_max, 0.0) + 1e-6]
    if db.size == 0:
        return float(max(0.0, t_max))
    return float(db[int(np.argmin(np.abs(db - t)))])


def _choose_outgoing_downbeat(a: AnalysisResult, overlap: float) -> float:
    """Latest A downbeat that leaves ``overlap`` seconds before the end.

    Among feasible downbeats, prefer one near a section boundary (a musically
    meaningful exit point); otherwise take the latest feasible downbeat.
    """
    db = a.downbeats
    feasible = db[db <= a.duration - overlap + 1e-6]
    if feasible.size == 0:
        return float(max(0.0, a.duration - overlap))

    # Prefer exiting from a *sustained-energy* part of A (avoid mixing out during
    # a breakdown/quiet tail). Keep candidates whose next bar is at least half of
    # A's peak energy; if that empties the set, fall back to all feasible.
    rms = getattr(a, "rms", np.array([]))
    hop_sec = getattr(a, "rms_hop_sec", 0.0)
    if rms.size and hop_sec > 0:
        bar = METER * 60.0 / max(a.bpm, 1.0)
        peak = float(np.percentile(rms, 95))
        energetic = np.array(
            [t for t in feasible
             if structure_mod.windowed_energy(rms, hop_sec, t, t + bar) >= 0.5 * peak]
        )
        if energetic.size:
            feasible = energetic

    boundaries = _section_boundaries(a)
    if boundaries.size:
        # Restrict to a late-ish window, then snap to the nearest boundary.
        window = feasible[feasible >= feasible.max() - 2 * overlap]
        cand = window if window.size else feasible
        dists = np.min(np.abs(cand[:, None] - boundaries[None, :]), axis=1)
        if float(dists.min()) <= overlap / 2:  # only if a boundary is genuinely close
            return float(cand[int(np.argmin(dists))])
    return float(feasible.max())


def _choose_incoming_switch(
    a: AnalysisResult,
    b: AnalysisResult,
    overlap: float,
    cue_method: str,
    phrase_bars: int,
) -> tuple[float, dict]:
    """Structure-aware incoming switch point (B's ORIGINAL timeline).

    Delegates to :mod:`structure`: energy-threshold (Phase 1) or novelty-ranked
    (Phase 2). Both skip B's quiet intro and land on a full-energy phrase.
    """
    if cue_method == "novelty":
        cands = structure_mod.find_switch_points(
            b, overlap, role="incoming", phrase_bars=phrase_bars,
            other_key=getattr(a, "key", {}),
        )
        if cands:
            top = cands[0]
            return float(top["time"]), {"reason": "top_novelty", "ranked": cands}
        # fall through to energy if novelty produced nothing
    t, info = structure_mod.choose_incoming_switch(
        b, overlap, phrase_bars=phrase_bars
    )
    return t, info


def _section_boundaries(a: AnalysisResult) -> np.ndarray:
    starts = [s["start"] for s in a.sections] if a.sections else []
    return np.asarray(sorted(set(starts)), dtype=float)
