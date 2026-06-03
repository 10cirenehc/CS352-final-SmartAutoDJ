"""SmartAutoDJ: Structure-Aware Automatic DJ Transitions.

A research prototype that, given two songs A and B, generates a short
beat-matched transition that is musically smarter than a plain crossfade.

The system is organized as a pipeline of composable stages (see ``pipeline.py``)
and is evaluated across three tiers:

    Tier 1  baseline      naive linear crossfade
    Tier 2  beat-aligned  tempo match + downbeat alignment + fade/EQ curves
    Tier 3  AI-enhanced   tier 2 + an additive generated "bridge" layer

This is a Music Perception course project; modules favour clear, commented
signal-processing code over cleverness.
"""

from .types import AnalysisResult, TransitionPlan

__all__ = ["AnalysisResult", "TransitionPlan"]

__version__ = "0.1.0"

# Canonical working sample rate for the whole pipeline (mono, float32).
DEFAULT_SR = 44100
