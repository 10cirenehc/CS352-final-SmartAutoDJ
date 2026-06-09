"""Local client for remote (Modal) generation of the tier-3 bridge.

The heavy generative model (audiocraft **MusicGen-Style** on a CUDA GPU) runs on
**Modal**, defined in ``infra/modal_bridge.py``. This module is the thin local
seam the pipeline calls when run with ``--generate``: it encodes the boundary-
reference audio, invokes the deployed Modal function, and writes the returned clip
to ``assets/generated/`` so the existing :func:`generative.resolve_bridge` path
loads it as a real AI clip (``source="ai-clip"``) and mixes it.

``modal`` is an **optional** dependency (extra ``gen``) and is imported lazily, so
the rest of the pipeline never requires it. Without ``--generate`` nothing here
runs and the procedural fallback is used.

Setup (once):  ``pip install -e ".[gen]"`` → ``modal setup`` →
``modal deploy infra/modal_bridge.py``.  See CLAUDE.md §5.
"""

from __future__ import annotations

import io
import os

import numpy as np
import soundfile as sf

MODAL_APP = "smartautodj-bridge"
MODAL_FN = "generate_bridge_remote"


def generate_bridge_via_modal(
    ref_wav: bytes, prompt: str, duration: float, params: dict | None = None
) -> bytes:
    """Call the deployed Modal function; return the generated WAV bytes.

    Requires the ``gen`` extra (``pip install modal``) and a deployed app
    (``modal deploy infra/modal_bridge.py``). ``--generate`` is an explicit opt-in,
    so a missing/unconfigured Modal raises a clear ``RuntimeError`` (loud, not a
    silent fall back to procedural) — run without ``--generate`` for the procedural
    bridge.
    """
    try:
        import modal  # optional dep; only needed with --generate
    except ImportError as e:
        raise RuntimeError(
            "modal not installed — run `pip install -e '.[gen]'` and "
            "`modal deploy infra/modal_bridge.py` (see CLAUDE.md §5)."
        ) from e
    fn = modal.Function.from_name(MODAL_APP, MODAL_FN)
    return fn.remote(ref_wav, prompt, float(duration), params or {})


def generate_bridge_to_file(
    ref_y: np.ndarray,
    sr: int,
    prompt: str,
    duration: float,
    out_path: str,
    params: dict | None = None,
) -> str:
    """Generate a bridge clip on Modal and write it to ``out_path``; return the path.

    ``ref_y`` is the boundary-reference audio (mono float32) from
    :func:`generative.extract_boundary_reference`. It is encoded to WAV bytes, sent
    to the Modal MusicGen-Style function, and the returned clip is written where the
    pipeline expects it (``assets/generated/<pair>_bridge.wav``) so ``resolve_bridge``
    picks it up on the same run.
    """
    buf = io.BytesIO()
    sf.write(buf, np.asarray(ref_y, dtype=np.float32), sr, format="WAV")
    clip_bytes = generate_bridge_via_modal(buf.getvalue(), prompt, duration, params)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(clip_bytes)
    return out_path
