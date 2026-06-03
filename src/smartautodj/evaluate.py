"""Objective metrics over the transition region, written into the JSON sidecar.

Three metrics, mirroring the proposal's evaluation plan:

  * **beat_alignment_error** — mean |Δt| between matched A/B beats inside the
    overlap. Tier 2/3 should beat tier 1 (Beat tracking topic).
  * **tempo_match_quality** — BPM gap before vs after time-stretching.
  * **loudness_continuity** — largest sudden level jump across the overlap; a
    smooth transition has no abrupt steps (Loudness & Amplitude topic).

All three are cheap because the beats/region are already computed upstream.
"""

from __future__ import annotations

import librosa
import numpy as np

from .align import matched_beats_in_overlap
from .types import AnalysisResult, TransitionPlan


def beat_alignment_error(a: AnalysisResult, b: AnalysisResult, plan: TransitionPlan) -> dict:
    """Mean absolute time error between matched A/B beats in the overlap."""
    a_times, b_times = matched_beats_in_overlap(a, b, plan)
    if a_times.size == 0:
        return {"mean_abs_error_sec": None, "n_matched_beats": 0}
    err = np.abs(a_times - b_times)
    return {
        "mean_abs_error_sec": round(float(err.mean()), 5),
        "max_abs_error_sec": round(float(err.max()), 5),
        "n_matched_beats": int(a_times.size),
    }


def tempo_match_quality(bpm_a: float, bpm_b: float, stretch_ratio: float) -> dict:
    """BPM disagreement before vs after stretching B (rate = stretch_ratio)."""
    before = abs(bpm_a - bpm_b)
    after = abs(bpm_a - bpm_b * stretch_ratio)
    return {
        "bpm_a": round(float(bpm_a), 3),
        "bpm_b": round(float(bpm_b), 3),
        "stretch_ratio": round(float(stretch_ratio), 5),
        "bpm_gap_before": round(float(before), 3),
        "bpm_gap_after": round(float(after), 3),
    }


def loudness_continuity(
    y: np.ndarray, info: dict, sr: int, window_sec: float = 1.0, hop_sec: float = 0.25
) -> dict:
    """Largest sudden level step (dB) in the *short-term loudness* across the overlap.

    A long smoothing window (~1 s, EBU R128-style "short-term" loudness) is used
    on purpose: a per-beat RMS frame would measure kick-vs-gap dynamics rather
    than the A->B handoff. Smoothing to ~1 s removes the beat-rate ripple and
    leaves the overall energy trajectory, so a hard cut or a mid-crossfade power
    dip shows up as a large step while a constant-power transition stays flat.

    Reports the max absolute step between consecutive (hop ~0.25 s) frames inside
    the overlap, plus the level standard deviation over the overlap.
    """
    y = np.asarray(y, dtype=np.float32)
    win = max(int(sr * window_sec), 1)
    hop = max(int(sr * hop_sec), 1)
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop)[0]
    db = librosa.amplitude_to_db(rms + 1e-8, ref=1.0)

    start_f = max(info["overlap_start_sample"] // hop, 0)
    end_f = min(
        (info["overlap_start_sample"] + info["overlap_len_sample"]) // hop + 1, len(db)
    )
    window = db[start_f:end_f]
    if window.size < 2:
        return {"max_db_jump": None, "std_db": None}
    return {
        "max_db_jump": round(float(np.max(np.abs(np.diff(window)))), 3),
        "std_db": round(float(np.std(window)), 3),
    }


def evaluate_all(
    a: AnalysisResult,
    b: AnalysisResult,
    plan: TransitionPlan,
    y_out: np.ndarray,
    info: dict,
    sr: int,
) -> dict:
    """Aggregate all metrics into one dict for the sidecar."""
    return {
        "beat_alignment": beat_alignment_error(a, b, plan),
        "tempo_match": tempo_match_quality(a.bpm, b.bpm, plan.stretch_ratio),
        "loudness_continuity": loudness_continuity(y_out, info, sr),
    }
