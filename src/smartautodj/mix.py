"""Assemble the final output: A intro + processed overlap + B outro.

The mix is sample-accurate: each region boundary is converted to an integer
sample index once, and the overlap is an overlap-add of the fade/EQ-processed
segments from ``transition`` plus the optional tier-3 bridge.

Output timeline:

    | A before overlap |  overlap (A_proc + B_proc + bridge)  | B after overlap |
                       ^ overlap_start (in output samples)
"""

from __future__ import annotations

import numpy as np

from .io import to_samples
from .postprocess import integrate_bridge, loudness_match  # noqa: F401 (re-export)
from .transition import process_overlap
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
) -> tuple[np.ndarray, dict]:
    """Render the full transition. ``b`` must already be tempo-stretched.

    Returns ``(y_out, info)`` where ``info`` records the overlap placement in
    output samples (for visualization / evaluation).
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    a0 = np.clip(to_samples(plan.region_a[0], sr), 0, len(a))
    a1 = np.clip(to_samples(plan.region_a[1], sr), a0, len(a))
    b0 = np.clip(to_samples(plan.region_b[0], sr), 0, len(b))
    b1 = np.clip(to_samples(plan.region_b[1], sr), b0, len(b))

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
