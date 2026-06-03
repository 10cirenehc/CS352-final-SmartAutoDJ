"""De-risk smoke test for the optional allin1 backend.

Skips cleanly when allin1 isn't installed (the librosa fallback covers that
case). When it *is* installed, this confirms allin1 actually runs on a short
clip and returns the fields the pipeline expects — finding out early per the
CLAUDE.md de-risk rule.
"""

from __future__ import annotations

import pytest

from smartautodj import analysis

# Apply the NATTEN/_device_t shim BEFORE importing allin1, otherwise the import
# raises and the test skips even when allin1 is installed and working.
analysis._ensure_natten_compat()
allin1 = pytest.importorskip("allin1", reason="allin1 optional; librosa fallback used otherwise")


def test_allin1_runs_on_click_track(click_tracks):
    a_path, _, sr = click_tracks
    res = analysis.analyze(a_path, backend="allin1", sr=sr)
    assert res.backend == "allin1"
    assert res.bpm > 0
    assert res.beats.size > 0
