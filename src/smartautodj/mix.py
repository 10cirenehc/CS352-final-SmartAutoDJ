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
from .transition import process_overlap
from .types import TransitionPlan


def mix_transition(
    a: np.ndarray,
    b: np.ndarray,
    plan: TransitionPlan,
    sr: int,
    bridge: np.ndarray | None = None,
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

    if bridge is not None and len(bridge):
        bridge = np.asarray(bridge, dtype=np.float32)
        bridge = bridge[:n] if len(bridge) >= n else np.pad(bridge, (0, n - len(bridge)))
        overlap = overlap + bridge

    y_out = np.concatenate([a_intro, overlap, b_outro]).astype(np.float32)
    info = {
        "overlap_start_sample": int(len(a_intro)),
        "overlap_len_sample": int(n),
        "out_len_sample": int(len(y_out)),
    }
    return y_out, info
