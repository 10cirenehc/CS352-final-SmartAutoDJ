"""Tempo / beat / downbeat / structure analysis.

Two backends behind one interface (``analyze`` -> ``AnalysisResult``):

  * **allin1** (preferred): a model that returns BPM, beats, downbeats and
    *labeled* sections (intro/verse/chorus/...). Carries install risk
    (needs madmom + ffmpeg), so it is optional.
  * **librosa** (fallback, always available): tempo + beats from
    ``librosa.beat.beat_track``; downbeats *derived* by assuming 4/4 and
    choosing the strongest bar phase; sections from a self-similarity
    segmentation.

``backend="auto"`` tries allin1 and silently falls back to librosa on any
failure — this is the de-risking guarantee: the pipeline always runs.

Course topics: Self-Similarity (structure segmentation), Deep Learning
(allin1 / neural beat tracking), Pitch/Chroma (features for segmentation).
"""

from __future__ import annotations

import warnings

import librosa
import numpy as np

from . import DEFAULT_SR
from .io import load_audio
from .types import AnalysisResult


def analyze(path: str, backend: str = "auto", sr: int = DEFAULT_SR) -> AnalysisResult:
    """Analyse a track and return an :class:`AnalysisResult`.

    Parameters
    ----------
    backend : {"auto", "allin1", "librosa"}
        "auto" prefers allin1 and falls back to librosa on any error.
    """
    if backend in ("auto", "allin1"):
        try:
            return _analyze_allin1(path, sr=sr)
        except Exception as exc:  # import error, runtime error, slow/flaky tool
            if backend == "allin1":
                raise
            warnings.warn(f"allin1 unavailable ({exc}); falling back to librosa.")
    return _analyze_librosa(path, sr=sr)


# --------------------------------------------------------------------------- #
# Preferred backend: all-in-one
# --------------------------------------------------------------------------- #
def _ensure_natten_compat() -> None:
    """Shim a stale symbol so allin1's pinned NATTEN imports on modern torch.

    allin1 1.1.0 pins NATTEN 0.15.x, whose Python layer does
    ``from torch.cuda import _device_t`` — a symbol newer torch removed. The
    NATTEN C++ backend itself compiles and runs fine against current torch, so we
    just re-provide the missing type alias before NATTEN/allin1 import. This is
    the documented fragile-pin friction in CLAUDE.md §5; the librosa fallback
    covers machines where even this doesn't hold.
    """
    import torch.cuda

    if not hasattr(torch.cuda, "_device_t"):
        from typing import Union

        torch.cuda._device_t = Union[torch.device, str, int, None]


def _analyze_allin1(path: str, sr: int = DEFAULT_SR) -> AnalysisResult:
    import tempfile

    _ensure_natten_compat()
    import allin1  # imported lazily so the package isn't a hard dependency

    # multiprocess=False is REQUIRED here: on macOS, allin1's worker processes
    # are spawned fresh and re-import the pinned NATTEN *without* the runtime
    # _device_t shim, so they die on import and the parent deadlocks. Running
    # in-process keeps the shim in effect. Byproducts (demixed stems, specs) go
    # to temp dirs so the repo stays clean.
    cache = tempfile.gettempdir()
    result = allin1.analyze(
        path,
        device="cpu",
        demix_dir=f"{cache}/allin1_demix",
        spec_dir=f"{cache}/allin1_spec",
        multiprocess=False,
    )
    if isinstance(result, (list, tuple)):  # allin1 returns a list for list input
        result = result[0]
    y, _ = load_audio(path, sr=sr)
    sections = [
        {"start": float(s.start), "end": float(s.end), "label": str(s.label)}
        for s in getattr(result, "segments", [])
    ]
    return AnalysisResult(
        path=path,
        sr=sr,
        duration=len(y) / sr,
        bpm=float(result.bpm),
        beats=np.asarray(result.beats, dtype=float),
        downbeats=np.asarray(result.downbeats, dtype=float),
        sections=sections,
        backend="allin1",
    )


# --------------------------------------------------------------------------- #
# Fallback backend: librosa
# --------------------------------------------------------------------------- #
def _analyze_librosa(path: str, sr: int = DEFAULT_SR) -> AnalysisResult:
    y, sr = load_audio(path, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    bpm = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(beat_frames, sr=sr)

    downbeats = _derive_downbeats(y, sr, beat_frames, beats)
    sections = _segment_structure(y, sr)

    return AnalysisResult(
        path=path,
        sr=sr,
        duration=len(y) / sr,
        bpm=bpm,
        beats=beats,
        downbeats=downbeats,
        sections=sections,
        backend="librosa",
    )


def _derive_downbeats(
    y: np.ndarray, sr: int, beat_frames: np.ndarray, beats: np.ndarray, meter: int = 4
) -> np.ndarray:
    """Pick downbeats by assuming a fixed ``meter`` (default 4/4).

    For each candidate bar phase (which beat in the bar is the "1"), sum the
    onset strength across the beats that fall on that phase. The phase with the
    greatest accumulated onset energy is taken as the downbeat phase — bar
    starts tend to carry the strongest accent.
    """
    if len(beats) < meter:
        return beats[:1].copy()

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    beat_frames = np.asarray(beat_frames)
    beat_frames = np.clip(beat_frames, 0, len(onset_env) - 1)
    beat_strength = onset_env[beat_frames]

    phase_scores = [beat_strength[phase::meter].sum() for phase in range(meter)]
    best_phase = int(np.argmax(phase_scores))
    return beats[best_phase::meter].copy()


def _segment_structure(y: np.ndarray, sr: int, n_segments: int = 4) -> list[dict]:
    """Cut the track into contiguous sections via self-similarity.

    Uses beat-synchronous chroma and ``librosa.segment.agglomerative`` to find
    boundaries (Self-Similarity course topic). Labels are generic ("seg0"..)
    since librosa does not name sections; allin1 provides real labels.
    """
    duration = len(y) / sr
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        bounds = librosa.segment.agglomerative(chroma, n_segments)
        bound_times = librosa.frames_to_time(bounds, sr=sr)
        edges = np.unique(np.concatenate([[0.0], bound_times, [duration]]))
        return [
            {"start": float(a), "end": float(b), "label": f"seg{i}"}
            for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]))
            if b - a > 1e-3
        ]
    except Exception:
        # Structure is a nicety, not required for alignment — degrade gracefully.
        return [{"start": 0.0, "end": float(duration), "label": "track"}]
