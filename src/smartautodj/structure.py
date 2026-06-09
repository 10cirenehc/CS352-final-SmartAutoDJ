"""Frequency-domain structure analysis and structure-aware cue points.

This module is where the project's spectral / MFCC / chroma analysis lives, and
where the "land in a high-energy section, not B's intro" fix is implemented.

Two layers:

1. **Per-track features** (``analyze_structure``):
     * **Loudness** — short-time RMS energy envelope (Loudness topic).
     * **Timbre** — MFCCs (MFCC topic).
     * **Harmony** — chroma, plus a Krumhansl-Schmuckler **key estimate**
       (Pitch / Chroma topic).
     * **Rhythm** — onset-strength envelope.
   The four feature streams are fused into a single **novelty curve** (where the
   music *changes* — high novelty = a structural boundary, the classic
   self-similarity / Foote idea, Self-Similarity topic).

2. **Cue-point selection** (``choose_incoming_switch`` / ``find_switch_points``):
   pick *where* to drop into B. The Milestone-1 code blindly took B's first
   downbeat, which is usually the quiet intro. Here we score downbeats on a
   phrase grid by energy (Phase 1) or by fused novelty + energy + harmonic
   compatibility (Phase 2), following the DJ "switch point" literature
   (Zehren et al. 2022): switch points sit where loudness/timbre/harmony/rhythm
   novelty is high and the incoming section has full energy.
"""

from __future__ import annotations

import librosa
import numpy as np

HOP = 512  # analysis hop for frame-level features

# --------------------------------------------------------------------------- #
# Krumhansl-Schmuckler key profiles (major / minor), normalised below.
# --------------------------------------------------------------------------- #
_KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
_PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# --------------------------------------------------------------------------- #
# Per-track feature extraction
# --------------------------------------------------------------------------- #
def analyze_structure(y: np.ndarray, sr: int, hop: int = HOP) -> dict:
    """Compute energy, fused novelty and a key estimate for one track.

    Returns a dict with:
      ``rms`` (normalised 0..1 energy envelope), ``hop_sec`` (seconds/frame),
      ``novelty`` (normalised 0..1 change curve, same frame grid), ``key``.
    """
    y = np.asarray(y, dtype=np.float32)
    hop_sec = hop / sr

    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_n = _norm01(rms)

    novelty = _fused_novelty(y, sr, hop)
    key = estimate_key(y, sr)

    return {"rms": rms_n, "hop_sec": float(hop_sec), "novelty": novelty, "key": key}


def _fused_novelty(y: np.ndarray, sr: int, hop: int) -> np.ndarray:
    """Fuse loudness/timbre/harmony/rhythm *flux* into one novelty curve.

    For each feature stream we take the frame-to-frame change (flux), normalise,
    and average. Spikes mark where several musical dimensions change at once —
    i.e. structural boundaries (intro->verse, build->drop). This is a cheap,
    robust stand-in for a full self-similarity-matrix + checkerboard-kernel
    novelty (the conceptual reference), and is enough to score downbeats.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)[None, :]
    rms = librosa.feature.rms(y=y, hop_length=hop)

    # Align all streams to the same number of frames.
    n = min(mfcc.shape[1], chroma.shape[1], onset.shape[1], rms.shape[1])
    streams = [mfcc[:, :n], chroma[:, :n], onset[:, :n], rms[:, :n]]

    fluxes = []
    for f in streams:
        f = _standardize_rows(f)
        d = np.sqrt(np.sum(np.diff(f, axis=1) ** 2, axis=0))  # L2 change per frame
        d = np.concatenate([[0.0], d])
        fluxes.append(_norm01(d))
    fused = np.mean(np.vstack(fluxes), axis=0)
    # Light smoothing so single-frame jitter doesn't dominate.
    fused = _smooth(fused, win=5)
    return _norm01(fused).astype(np.float32)


def estimate_key(y: np.ndarray, sr: int) -> dict:
    """Estimate global key via Krumhansl-Schmuckler chroma correlation.

    Returns ``{"name", "tonic", "mode", "confidence"}`` (e.g. "A minor").
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    profile = chroma.mean(axis=1)
    if profile.sum() <= 0:
        return {"name": "unknown", "tonic": None, "mode": None, "confidence": 0.0}
    profile = profile / profile.sum()

    maj = _KS_MAJOR / _KS_MAJOR.sum()
    minr = _KS_MINOR / _KS_MINOR.sum()
    best = (-2.0, 0, "major")
    for tonic in range(12):
        r_maj = _corr(profile, np.roll(maj, tonic))
        r_min = _corr(profile, np.roll(minr, tonic))
        if r_maj > best[0]:
            best = (r_maj, tonic, "major")
        if r_min > best[0]:
            best = (r_min, tonic, "minor")
    score, tonic, mode = best
    return {
        "name": f"{_PITCHES[tonic]} {mode}",
        "tonic": int(tonic),
        "mode": mode,
        "confidence": round(float(max(score, 0.0)), 3),
    }


def key_compatibility(key_a: dict, key_b: dict) -> float:
    """Harmonic-compatibility score in [0, 1] between two key estimates.

    Mirrors DJ "harmonic mixing" (Camelot wheel): same key = 1.0, perfect-fifth
    neighbours and relative major/minor are highly compatible, distant keys low.
    """
    if not key_a or not key_b:
        return 0.5
    ta, tb = key_a.get("tonic"), key_b.get("tonic")
    ma, mb = key_a.get("mode"), key_b.get("mode")
    if ta is None or tb is None:
        return 0.5
    if ta == tb and ma == mb:
        return 1.0
    # Relative major/minor (e.g. C major <-> A minor): tonics 3 semitones apart.
    if ma != mb and ((ta - tb) % 12 in (3, 9)):
        return 0.9
    if ma == mb:
        # Circle-of-fifths distance (0..6); fifths neighbours score high.
        cof = abs(((ta - tb) * 7) % 12)
        cof = min(cof, 12 - cof)
        return float(np.clip(1.0 - cof / 6.0, 0.0, 1.0))
    return 0.3


# --------------------------------------------------------------------------- #
# Cue-point selection
# --------------------------------------------------------------------------- #
def windowed_energy(rms: np.ndarray, hop_sec: float, t0: float, t1: float) -> float:
    """Mean RMS energy over the time window ``[t0, t1)`` (0 if out of range)."""
    if rms.size == 0 or hop_sec <= 0 or t1 <= t0:
        return 0.0
    i0 = int(max(0, np.floor(t0 / hop_sec)))
    i1 = int(min(rms.size, np.ceil(t1 / hop_sec)))
    if i1 <= i0:
        return 0.0
    return float(np.mean(rms[i0:i1]))


def _phrase_grid(downbeats: np.ndarray, phrase_bars: int) -> np.ndarray:
    """Candidate switch points: every ``phrase_bars``-th downbeat (a phrase)."""
    db = np.asarray(downbeats, dtype=float)
    db = db[db > 1e-3]  # drop a leading downbeat at t=0 (track start)
    if db.size == 0:
        return np.asarray(downbeats, dtype=float)[:1]
    grid = db[:: max(phrase_bars, 1)]
    return grid if grid.size else db[:1]


def choose_incoming_switch(
    b,
    overlap: float,
    phrase_bars: int = 4,
    energy_frac: float = 0.6,
) -> tuple[float, dict]:
    """Phase-1 cue: earliest high-energy phrase downbeat in B (skip the intro).

    Walks B's phrase grid and returns the first downbeat whose following section
    has energy >= ``energy_frac`` x B's peak energy — i.e. the first point where
    B has really "kicked in". Falls back to the highest-energy feasible phrase
    point, then to B's first downbeat. ``b`` is an ``AnalysisResult`` carrying
    ``rms``/``rms_hop_sec`` (original, un-stretched timeline).
    """
    db = np.asarray(b.downbeats, dtype=float)
    grid = _phrase_grid(db, phrase_bars)
    rms = getattr(b, "rms", np.array([]))
    hop_sec = getattr(b, "rms_hop_sec", 0.0)

    if rms.size == 0 or hop_sec <= 0:
        # No energy info -> old behaviour (first non-zero downbeat).
        nz = db[db > 1e-3]
        return (float(nz[0]) if nz.size else float(db[0] if db.size else 0.0)), {
            "method": "energy", "reason": "no_rms_fallback",
        }

    bar = 4 * 60.0 / max(b.bpm, 1.0)
    win = min(2 * bar, max(overlap, bar))  # energy of the next ~2 bars

    feasible = [t for t in grid if t + overlap <= b.duration + 1e-6]
    cand = feasible if feasible else list(grid)
    energies = [(t, windowed_energy(rms, hop_sec, t, t + win)) for t in cand]

    # Threshold relative to the *loudest phrase* (not the loudest frame): the
    # incoming section should be at least ``energy_frac`` as loud as B's biggest
    # section. This robustly separates a quiet intro from where B kicks in.
    peak_phrase = max((e for _, e in energies), default=0.0)
    threshold = energy_frac * peak_phrase

    for t, e in energies:
        if e >= threshold:
            return float(t), {
                "method": "energy", "reason": "first_high_energy_phrase",
                "energy": round(e, 5), "threshold": round(threshold, 5),
                "phrase_bars": phrase_bars,
            }
    # Nothing crossed the bar -> take the most energetic feasible phrase point.
    if energies:
        t, e = max(energies, key=lambda x: x[1])
        return float(t), {
            "method": "energy", "reason": "max_energy_phrase",
            "energy": round(e, 5), "threshold": round(threshold, 5),
        }
    nz = db[db > 1e-3]
    return (float(nz[0]) if nz.size else 0.0), {"method": "energy", "reason": "fallback"}


def find_switch_points(
    track,
    overlap: float,
    role: str = "incoming",
    phrase_bars: int = 4,
    other_key: dict | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Phase-2 cue: rank phrase downbeats by fused novelty + energy (+ harmony).

    For ``role="incoming"`` we want a point that (a) starts a new section
    (high novelty just before it), (b) has full energy after it, and (c) is
    harmonically compatible with the outgoing track (``other_key``). Returns a
    list of ``{"time", "score", "energy", "novelty", ...}`` sorted best-first.
    """
    db = np.asarray(track.downbeats, dtype=float)
    grid = _phrase_grid(db, phrase_bars)
    rms = getattr(track, "rms", np.array([]))
    nov = getattr(track, "novelty", np.array([]))
    hop_sec = getattr(track, "rms_hop_sec", 0.0)
    bar = 4 * 60.0 / max(track.bpm, 1.0)
    win = min(2 * bar, max(overlap, bar))

    feasible = [t for t in grid if t + overlap <= track.duration + 1e-6]
    cand = feasible if feasible else list(grid)
    if not cand:
        return []

    harm = key_compatibility(getattr(track, "key", {}), other_key) if other_key else 0.5
    scored = []
    for t in cand:
        e = windowed_energy(rms, hop_sec, t, t + win) if rms.size else 0.0
        # Novelty just *before* the downbeat = a boundary lands on this bar.
        nvl = windowed_energy(nov, hop_sec, t - bar, t) if nov.size else 0.0
        score = 0.5 * e + 0.3 * nvl + 0.2 * harm
        scored.append(
            {
                "time": round(float(t), 4),
                "score": round(float(score), 4),
                "energy": round(float(e), 4),
                "novelty": round(float(nvl), 4),
                "harm_compat": round(float(harm), 3),
            }
        )
    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:top_k]


def _cos01(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity mapped to [0, 1] (0.5 if either vector is empty)."""
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-9 or nv < 1e-9:
        return 0.5
    return (float(np.dot(u, v) / (nu * nv)) + 1.0) / 2.0


def _window_chroma(chroma: np.ndarray, hop_sec: float, times, overlap: float) -> np.ndarray:
    """Unit-norm mean chroma vector for each window ``[t, t+overlap]`` (N x 12)."""
    out = []
    for t in times:
        i0 = max(0, int(t / hop_sec))
        i1 = max(min(chroma.shape[1], int((t + overlap) / hop_sec)), i0 + 1)
        v = chroma[:, i0:i1].mean(axis=1)
        out.append(v / (np.linalg.norm(v) + 1e-9))
    return np.asarray(out) if out else np.zeros((0, chroma.shape[0]))


def chorus_scores(
    chroma: np.ndarray, hop_sec: float, cand_times, all_times, overlap: float,
) -> np.ndarray:
    """Repetition (chorus-likeness) score in [0,1] for each candidate window.

    Self-similarity *within* a song (HW3): the chorus/hook is the section that
    REPEATS, so a window scores high if it's highly similar (chroma cosine) to
    OTHER, non-adjacent windows elsewhere in the track. Energy can't separate
    verse from chorus when a track is loudness-compressed; repetition can.
    """
    cand = _window_chroma(chroma, hop_sec, cand_times, overlap)
    ref = _window_chroma(chroma, hop_sec, all_times, overlap)
    if cand.shape[0] == 0 or ref.shape[0] == 0:
        return np.zeros(len(cand_times))
    sims = cand @ ref.T  # cosine (both unit-norm): (n_cand, n_ref)
    all_arr = np.asarray(all_times, dtype=float)
    reps = []
    for i, t in enumerate(cand_times):
        row = sims[i].copy()
        row[np.abs(all_arr - t) < 10.0] = -1.0  # exclude self + near neighbours
        valid = row[row > -1.0]
        top = np.sort(valid)[-3:] if valid.size else np.array([0.0])  # 3 best matches elsewhere
        reps.append(float(np.mean(top)))
    return _norm01(np.asarray(reps))


def choose_transition_pair(
    a, b, y_a: np.ndarray, y_b: np.ndarray, sr: int,
    overlap: float, phrase_bars: int = 4, hop: int = HOP,
) -> tuple[float, float, dict]:
    """Jointly pick (A-exit, B-entry) downbeats that *match* and have energy.

    The Infinite-Jukebox / self-similarity idea (HW3) applied across two songs:
    for each candidate exit phrase in A and entry phrase in B, compare the two
    overlap windows with **beat-window chroma (harmony) + MFCC (timbre) cosine
    similarity**, and reward energy on *both* sides (exit from a climax, drop into
    a strong section — so it isn't flat). Returns ``(start_a, start_b_orig, info)``
    with ``start_b_orig`` in B's ORIGINAL (un-stretched) timeline.
    """
    y_a = np.asarray(y_a, dtype=np.float32)
    y_b = np.asarray(y_b, dtype=np.float32)
    hop_sec = hop / sr
    chroma_a = librosa.feature.chroma_cqt(y=y_a, sr=sr, hop_length=hop)
    mfcc_a = librosa.feature.mfcc(y=y_a, sr=sr, n_mfcc=13, hop_length=hop)
    chroma_b = librosa.feature.chroma_cqt(y=y_b, sr=sr, hop_length=hop)
    mfcc_b = librosa.feature.mfcc(y=y_b, sr=sr, n_mfcc=13, hop_length=hop)

    def win_feat(chroma, mfcc, t0):
        i0 = max(0, int(t0 / hop_sec))
        i1 = min(chroma.shape[1], int((t0 + overlap) / hop_sec))
        i1 = max(i1, i0 + 1)
        return chroma[:, i0:i1].mean(axis=1), mfcc[1:, i0:i1].mean(axis=1)  # drop c0

    rms_a, hsa = getattr(a, "rms", np.array([])), getattr(a, "rms_hop_sec", 0.0)
    rms_b, hsb = getattr(b, "rms", np.array([])), getattr(b, "rms_hop_sec", 0.0)

    # A exit candidates: phrase downbeats in the back half (play most of A) that
    # leave `overlap` before the end. B entry candidates: high-energy phrases
    # (skip the intro) that leave `overlap` before B's end.
    grid_a = _phrase_grid(np.asarray(a.downbeats, float), phrase_bars)
    a_cands = [t for t in grid_a if 0.5 * a.duration <= t <= a.duration - overlap + 1e-6]
    if not a_cands:
        a_cands = [t for t in grid_a if t <= a.duration - overlap + 1e-6] or [max(0.0, a.duration - overlap)]
    a_cands = a_cands[-16:]  # nearest to the end
    # Keep only high-energy A exits (exit from a CHORUS, not a verse/breakdown).
    a_en = [(t, windowed_energy(rms_a, hsa, t, t + overlap)) for t in a_cands]
    peak_a = max((e for _, e in a_en), default=0.0)
    a_cands = [t for t, e in a_en if e >= 0.7 * peak_a] or [t for t, _ in a_en]

    grid_b = _phrase_grid(np.asarray(b.downbeats, float), phrase_bars)
    b_feas = [t for t in grid_b if t + overlap <= b.duration + 1e-6] or list(grid_b)
    b_en = [(t, windowed_energy(rms_b, hsb, t, t + overlap)) for t in b_feas]
    peak_b = max((e for _, e in b_en), default=0.0)
    # Drop into a CHORUS-level section of B (raise the energy floor).
    b_cands = [t for t, e in b_en if e >= 0.7 * peak_b] or [t for t, _ in b_en]
    b_cands = b_cands[:16]

    harm = key_compatibility(getattr(a, "key", {}), getattr(b, "key", {}))
    # Chorus-likeness (repetition) of each candidate, vs all phrase windows.
    all_a = [t for t in grid_a if t + overlap <= a.duration + 1e-6] or list(grid_a)
    rep_a = dict(zip(a_cands, chorus_scores(chroma_a, hop_sec, a_cands, all_a, overlap)))
    rep_b = dict(zip(b_cands, chorus_scores(chroma_b, hop_sec, b_cands, b_feas, overlap)))

    best = None
    for ta in a_cands:
        ca, ma = win_feat(chroma_a, mfcc_a, ta)
        e_a = windowed_energy(rms_a, hsa, ta, ta + overlap) if rms_a.size else 0.5
        for tb in b_cands:
            cb, mb = win_feat(chroma_b, mfcc_b, tb)
            e_b = windowed_energy(rms_b, hsb, tb, tb + overlap) if rms_b.size else 0.5
            cross = 0.6 * _cos01(ca, cb) + 0.4 * _cos01(ma, mb)
            chorus = 0.5 * (rep_a.get(ta, 0.0) + rep_b.get(tb, 0.0))
            # Land near a CHORUS (repetition) on both sides, with a smooth blend;
            # energy is a light tiebreak (it's near-flat on compressed tracks).
            score = (0.38 * chorus + 0.24 * cross + 0.12 * e_b + 0.10 * e_a
                     + 0.16 * harm)
            cand = (score, ta, tb, e_a, e_b, cross, chorus)
            if best is None or score > best[0]:
                best = cand

    if best is None:  # degenerate (no candidates) — caller will fall back
        return float(max(0.0, a.duration - overlap)), float(b.downbeats[0] if len(b.downbeats) else 0.0), {
            "method": "match", "reason": "no_candidates",
        }
    score, ta, tb, e_a, e_b, cross, chorus = best
    info = {
        "method": "match", "reason": "chorus_match",
        "score": round(score, 4), "chorus": round(chorus, 4),
        "energy_a": round(e_a, 4), "energy_b": round(e_b, 4),
        "cross_sim": round(cross, 4), "harm_compat": round(harm, 3), "phrase_bars": phrase_bars,
    }
    return float(ta), float(tb), info


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #
def _norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _standardize_rows(f: np.ndarray) -> np.ndarray:
    mu = f.mean(axis=1, keepdims=True)
    sd = f.std(axis=1, keepdims=True) + 1e-9
    return (f - mu) / sd


def _smooth(x: np.ndarray, win: int = 5) -> np.ndarray:
    if win <= 1 or x.size < win:
        return x
    k = np.ones(win) / win
    return np.convolve(x, k, mode="same")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)
