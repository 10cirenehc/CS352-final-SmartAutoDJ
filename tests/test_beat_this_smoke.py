"""De-risk smoke test for the preferred beat_this backend.

Skips cleanly when beat_this isn't installed (the librosa fallback covers that
case). When it *is* installed, this confirms beat_this actually runs on a short
clip and returns the fields the pipeline expects — the Colab-friendly upgrade
that replaces the fragile allin1 path (CLAUDE.md §5/§11).
"""

from __future__ import annotations

import pytest

from smartautodj import analysis

pytest.importorskip(
    "beat_this.inference", reason="beat_this optional; librosa fallback used otherwise"
)


def test_beat_this_runs_on_click_track(click_tracks):
    a_path, _, sr = click_tracks
    res = analysis.analyze(a_path, backend="beat_this", sr=sr)
    assert res.backend == "beat_this"
    assert res.bpm > 0
    assert res.beats.size > 0
    assert res.downbeats.size > 0
    # Structure features are attached for every backend.
    assert res.key.get("mode") in ("major", "minor")
