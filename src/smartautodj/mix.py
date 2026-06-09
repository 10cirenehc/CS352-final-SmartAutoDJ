"""Assemble the final output: A intro + processed overlap + B outro.

The mix is sample-accurate: each region boundary is converted to an integer
sample index once. There are two render shapes (``plan.shape``):

  * **blend** (default) — an overlap-add of the fade/EQ-processed segments from
    ``transition`` plus the optional tier-3 bridge. A and B play together over
    the whole region (a symmetric crossfade).

        | A before overlap |  overlap (A_proc + B_proc + bridge)  | B after overlap |
                           ^ overlap_start (in output samples)

  * **cut** — a *sequential* quick-cut: A fades out fast over a short seam, the
    buildup effect leads over that seam, then B is concatenated (dropped) in on
    the downbeat, its bass swelling in via a high-pass filter-in.

        | A before seam |  seam (A fade-out + effect)  | B from the drop (filter-swept in) |
                        ^ overlap_start                ^ B's drop downbeat (on a bar by construction)
"""

from __future__ import annotations

import numpy as np

from .io import to_samples
from .postprocess import integrate_bridge, loudness_match, soft_limit  # noqa: F401 (re-export)
from .transition import (
    cut_bridge_envelope,
    cut_fade_out,
    fade_curves,
    make_whoosh,
    process_overlap,
    split_bands,
    swept_bandpass_svf,
    sweep_ramp,
)
from .types import TransitionPlan


def mix_transition(
    a: np.ndarray,
    b: np.ndarray,
    plan: TransitionPlan,
    sr: int,
    bridge: np.ndarray | None = None,
    bridge_gain_db: float = -3.0,
    bridge_duck_db: float = 6.0,
    bridge_limit: bool = True,
    bridge_spectral: bool = True,
    cut_sweep_sec: float | None = None,
    cut_woosh: str = "noise",
    cut_xfade_sec: float | None = None,
    cut_fade_frac: float = 0.65,
) -> tuple[np.ndarray, dict]:
    """Render the full transition. ``b`` must already be tempo-stretched.

    Returns ``(y_out, info)`` where ``info`` records the overlap placement in
    output samples (for visualization / evaluation). ``plan.shape`` selects the
    render: ``"blend"`` (symmetric crossfade) or ``"cut"`` (sequential quick-cut).
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    a0 = np.clip(to_samples(plan.region_a[0], sr), 0, len(a))
    a1 = np.clip(to_samples(plan.region_a[1], sr), a0, len(a))
    b0 = np.clip(to_samples(plan.region_b[0], sr), 0, len(b))
    b1 = np.clip(to_samples(plan.region_b[1], sr), b0, len(b))

    if getattr(plan, "shape", "blend") == "cut":
        return _mix_cut(
            a, b, a0, a1, b0, sr,
            bridge=bridge, bridge_gain_db=bridge_gain_db, bridge_limit=bridge_limit,
            cut_sweep_sec=cut_sweep_sec, cut_woosh=cut_woosh, cut_xfade_sec=cut_xfade_sec,
            cut_fade_frac=cut_fade_frac,
        )

    a_intro = a[:a0]
    a_tail = a[a0:a1]
    b_head = b[b0:b1]
    b_outro = b[b1:]

    a_proc, b_proc = process_overlap(a_tail, b_head, sr, plan)
    n = len(a_proc)
    overlap = a_proc + b_proc

    bridge_info: dict = {}
    if bridge is not None and len(bridge):
        bridge = np.asarray(bridge, dtype=np.float32)
        bridge = bridge[:n] if len(bridge) >= n else np.pad(bridge, (0, n - len(bridge)))
        # Make the bridge sit like a DJ transition: loudness-match -> limit ->
        # duck the two songs under it -> spectral carve (see postprocess.py). The
        # headroom guard below still catches any over-full-scale peak.
        overlap, bridge_info = integrate_bridge(
            overlap, bridge, sr,
            rel_db=bridge_gain_db, duck_db=bridge_duck_db,
            limit=bridge_limit, spectral=bridge_spectral,
        )

    overlap_peak = float(np.max(np.abs(overlap))) if n else 0.0

    y_out = np.concatenate([a_intro, overlap, b_outro]).astype(np.float32)

    # Headroom guard: if summing A+B(+bridge) pushed the overlap past full scale,
    # scale the WHOLE output down (not just the overlap) so we never hard-clip
    # mid-transition — clipping is itself an audible "warped"/crunchy artifact —
    # while preserving relative levels across the seams (no loudness jump).
    ceiling = 0.99
    applied_gain = 1.0
    peak = float(np.max(np.abs(y_out))) if y_out.size else 0.0
    if peak > ceiling:
        applied_gain = ceiling / peak
        y_out = (y_out * applied_gain).astype(np.float32)

    info = {
        "overlap_start_sample": int(len(a_intro)),
        "overlap_len_sample": int(n),
        "out_len_sample": int(len(y_out)),
        "overlap_peak": round(overlap_peak, 4),
        "headroom_gain": round(applied_gain, 4),
        **bridge_info,  # bridge_rel_db / duck / limit / spectral / peak (if a bridge)
    }
    return y_out, info


def _mix_cut(
    a: np.ndarray, b: np.ndarray, a0: int, a1: int, b0: int, sr: int, *,
    bridge: np.ndarray | None, bridge_gain_db: float, bridge_limit: bool,
    cut_sweep_sec: float | None, cut_woosh: str = "noise",
    cut_xfade_sec: float | None = None, cut_fade_frac: float = 0.65,
    woosh_lo: float = 200.0, woosh_hi: float = 12000.0, woosh_q: float = 8.0,
) -> tuple[np.ndarray, dict]:
    """Quick-cut render: fade A out -> riser over the seam -> hand off into B.

    The cut is *sequential*: A's tail (``a[a0:a1]``, the seam) fades out fast and
    the riser leads over it. The hand-off into B (the "END") is smoothed two ways:

      * **crossfade** — the riser's tail (last ``cut_xfade_sec``) is equal-power
        crossfaded with B's entrance instead of a hard cut, so the riser's energy
        hands *into* B rather than stopping dead (this is the "smoosh").
      * **woosh** — the swoosh into the drop. ``"noise"`` (default) adds a
        swept-resonant-noise whoosh over the seam tail (a real moving resonance that
        sweeps up the spectrum) while B drops full underneath. ``"bandpass"`` sweeps a
        resonant band-pass on B's own entrance, crossfading up to full B.
        ``"highpass"`` keeps highs and swells bass in; ``"lowpass"`` enters dark and
        opens up; ``"none"`` is dry.

    B still *starts* on its downbeat ``b0`` (only its first bar overlaps/eases in),
    so the drop stays on the grid.
    """
    a_intro = a[:a0]
    a_tail = a[a0:a1]
    seam_n = len(a_tail)

    # Seam = A fading out fast. Keep a copy of A's pre-fade tail as the loudness
    # reference for the effect (so the effect sits at a song-relative level even
    # though A itself is nearly gone by the drop).
    seam = (a_tail * cut_fade_out(seam_n, fade_frac=cut_fade_frac)).astype(np.float32)

    bridge_info: dict = {}
    if bridge is not None and len(bridge) and seam_n > 1:
        eff = np.asarray(bridge, dtype=np.float32)
        eff = eff[:seam_n] if len(eff) >= seam_n else np.pad(eff, (0, seam_n - len(eff)))
        # Stage the effect by loudness relative to A's (un-faded) tail, then swell
        # it into the drop. Over the seam it is the lead (A is dying, B not in yet)
        # so there's no ducking — it should be prominent.
        eff = loudness_match(eff, a_tail, rel_db=bridge_gain_db)
        eff = eff * cut_bridge_envelope(seam_n)
        if bridge_limit:
            ceiling = max(float(np.max(np.abs(a_tail))) if a_tail.size else 0.0, 1e-3)
            eff = soft_limit(eff, ceiling)
        seam = (seam + eff).astype(np.float32)
        # Soft-limit the *seam* locally so the effect's transients don't push it
        # over full scale and force the global headroom guard to turn the WHOLE
        # song down (the seam is the only place the effect adds energy).
        if bridge_limit:
            seam = soft_limit(seam, 0.99)
        bridge_info = {
            "bridge_rel_db": bridge_gain_db,
            "bridge_envelope": "rising_into_drop",
            "bridge_limit": bool(bridge_limit),
            "bridge_peak": round(float(np.max(np.abs(eff))) if eff.size else 0.0, 4),
        }

    # Woosh length (None -> half the seam; a number incl. 0 used as-is).
    woosh_raw = seam_n // 2 if cut_sweep_sec is None else int(cut_sweep_sec * sr)

    # "noise" woosh: a swept-resonant-noise whoosh (real moving resonance) added over
    # the seam tail on top of the riser, swelling into the drop. B drops full
    # underneath via the crossfade below. This is the aggressive, audible swoosh.
    if cut_woosh == "noise" and seam_n > 1:
        wn = int(np.clip(woosh_raw, 0, seam_n))
        if wn > 1:
            whoosh = make_whoosh(wn, sr, woosh_lo, woosh_hi, woosh_q)
            whoosh = loudness_match(whoosh, a_tail, rel_db=bridge_gain_db)
            seam[seam_n - wn:] = (seam[seam_n - wn:] + whoosh).astype(np.float32)
            if bridge_limit:
                seam = soft_limit(seam, 0.99)

    # B enters from b0 onward (everything from the downbeat, not just an outro).
    b_rest = np.array(b[b0:], dtype=np.float32, copy=True)
    # Filter-sweep B's entrance (level-preserving — real B audio, no normalize).
    sweep_n = int(np.clip(woosh_raw, 0, len(b_rest)))
    if sweep_n > 1 and cut_woosh in ("highpass", "lowpass", "bandpass"):
        if cut_woosh == "bandpass":
            # B wooshes in through a swept resonant band-pass, crossfading up to full B.
            swept = swept_bandpass_svf(b_rest[:sweep_n], sr, woosh_lo, woosh_hi, woosh_q)
            r = sweep_ramp(sweep_n)
            b_rest[:sweep_n] = (swept * (1.0 - r) + b_rest[:sweep_n] * r).astype(np.float32)
        elif cut_woosh == "lowpass":
            # B enters dark/muffled and brightens to full (highs swell in).
            low, high = split_bands(b_rest[:sweep_n], sr, cutoff=300.0, order=4)
            b_rest[:sweep_n] = (low + high * sweep_ramp(sweep_n)).astype(np.float32)
        else:  # "highpass": highs present from the start, bass swells in
            low, high = split_bands(b_rest[:sweep_n], sr, cutoff=200.0, order=4)
            b_rest[:sweep_n] = (high + low * sweep_ramp(sweep_n)).astype(np.float32)

    # Hand off the seam into B. With a crossfade, overlap-add the riser tail (fading
    # out) with B's entrance (fading in) so the energy hands over instead of cutting
    # dead; otherwise a tiny declick on a hard concatenation.
    # None -> default (a quarter of the seam); a number (incl. 0 = hard cut) used as-is.
    xfade_n = seam_n // 4 if cut_xfade_sec is None else int(cut_xfade_sec * sr)
    xfade_n = int(np.clip(xfade_n, 0, min(seam_n, len(b_rest))))
    if xfade_n > 1:
        fo, fi = fade_curves(xfade_n, "equal_power")
        tail = seam[seam_n - xfade_n:] * fo
        head = b_rest[:xfade_n] * fi
        cross = (tail + head).astype(np.float32)
        y_out = np.concatenate([a_intro, seam[: seam_n - xfade_n], cross, b_rest[xfade_n:]])
        y_out = y_out.astype(np.float32)
    else:
        dc = int(min(0.005 * sr, len(b_rest)))  # declick the hard cut
        if dc > 1:
            b_rest[:dc] *= np.linspace(0.0, 1.0, dc, dtype=np.float32)
        y_out = np.concatenate([a_intro, seam, b_rest]).astype(np.float32)
    seam_peak = float(np.max(np.abs(seam))) if seam_n else 0.0

    # Same headroom guard as the blend path.
    ceiling = 0.99
    applied_gain = 1.0
    peak = float(np.max(np.abs(y_out))) if y_out.size else 0.0
    if peak > ceiling:
        applied_gain = ceiling / peak
        y_out = (y_out * applied_gain).astype(np.float32)

    info = {
        "shape": "cut",
        "overlap_start_sample": int(len(a_intro)),
        "overlap_len_sample": int(seam_n),
        "out_len_sample": int(len(y_out)),
        "overlap_peak": round(seam_peak, 4),
        "headroom_gain": round(applied_gain, 4),
        "sweep_sample": int(sweep_n),
        "woosh": cut_woosh,
        "xfade_sample": int(xfade_n),
        **bridge_info,
    }
    return y_out, info
