"""Core data structures passed between pipeline stages.

Two small dataclasses carry everything the stages need:

  * ``AnalysisResult`` — the musical description of one track (tempo, beats,
    downbeats, structure). Produced by ``analysis.analyze``.
  * ``TransitionPlan`` — the decision of *where* and *how* to transition from
    A into B. Produced by ``align`` and consumed by ``transition``/``mix``.

Both expose ``to_dict`` so the pipeline can write a JSON sidecar next to every
output WAV (a project requirement: results must be inspectable and reusable by
the evaluation code).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _arr(x) -> np.ndarray:
    """Coerce to a 1-D float array (times in seconds)."""
    return np.asarray(x, dtype=float).reshape(-1)


@dataclass
class AnalysisResult:
    """Musical analysis of a single track.

    Attributes
    ----------
    path : str
        Source audio path (for provenance in the sidecar).
    sr : int
        Sample rate the track was analysed at.
    duration : float
        Track duration in seconds.
    bpm : float
        Estimated global tempo.
    beats : np.ndarray
        Beat times in seconds.
    downbeats : np.ndarray
        Downbeat (bar-start) times in seconds. With librosa these are derived
        by assuming 4/4 and picking the strongest bar phase; with allin1 they
        come from the model directly.
    sections : list[dict]
        Structure as ``{"start", "end", "label"}`` dicts (e.g. intro/verse/
        chorus). May be a single "track" span if structure is unavailable.
    backend : str
        Which analyser produced this ("allin1" or "librosa").
    """

    path: str
    sr: int
    duration: float
    bpm: float
    beats: np.ndarray
    downbeats: np.ndarray
    sections: list = field(default_factory=list)
    backend: str = "librosa"

    def __post_init__(self) -> None:
        self.beats = _arr(self.beats)
        self.downbeats = _arr(self.downbeats)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "sr": self.sr,
            "duration": round(float(self.duration), 4),
            "bpm": round(float(self.bpm), 3),
            "backend": self.backend,
            "beats": [round(float(t), 4) for t in self.beats],
            "downbeats": [round(float(t), 4) for t in self.downbeats],
            "sections": self.sections,
        }


@dataclass
class TransitionPlan:
    """Where and how to transition from A into B.

    All times are in seconds. ``region_a``/``region_b`` are the (start, end)
    spans of each track that overlap during the transition.
    """

    tier: int
    overlap_sec: float
    region_a: tuple = (0.0, 0.0)
    region_b: tuple = (0.0, 0.0)
    anchor_downbeats_a: np.ndarray = field(default_factory=lambda: np.array([]))
    anchor_downbeats_b: np.ndarray = field(default_factory=lambda: np.array([]))
    stretch_ratio: float = 1.0
    fade_shape: str = "equal_power"
    eq_params: dict = field(default_factory=dict)
    bridge: Optional[dict] = None  # {"kind", "start", "end"} or None

    def __post_init__(self) -> None:
        self.anchor_downbeats_a = _arr(self.anchor_downbeats_a)
        self.anchor_downbeats_b = _arr(self.anchor_downbeats_b)

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "overlap_sec": round(float(self.overlap_sec), 4),
            "region_a": [round(float(x), 4) for x in self.region_a],
            "region_b": [round(float(x), 4) for x in self.region_b],
            "anchor_downbeats_a": [round(float(t), 4) for t in self.anchor_downbeats_a],
            "anchor_downbeats_b": [round(float(t), 4) for t in self.anchor_downbeats_b],
            "stretch_ratio": round(float(self.stretch_ratio), 5),
            "fade_shape": self.fade_shape,
            "eq_params": self.eq_params,
            "bridge": self.bridge,
        }
