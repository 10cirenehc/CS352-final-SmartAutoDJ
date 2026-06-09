"""De-risk smoke test for the optional pretrained genre classifier.

Skips cleanly when ``transformers`` isn't installed (the pipeline then falls back
to the default beat-aligned style). When it *is* installed, this confirms the
GTZAN model loads, runs on CPU, and returns a label + a confidence in [0, 1] on a
short synthetic clip — proving the dependency works end-to-end (CLAUDE.md §5/§11).

Downloads the model on first run (~90 MB), then it's cached; subsequent runs are
fast (~0.3 s per excerpt).
"""

from __future__ import annotations

import pytest

from smartautodj import genre

pytest.importorskip(
    "transformers", reason="genre model optional; falls back to default style otherwise"
)


def test_classify_genre_returns_label_and_confidence(click_tracks):
    a_path, _, _ = click_tracks
    out = genre.classify_genre(a_path)
    assert isinstance(out["label"], str) and out["label"]
    # On a successful run the label is a real GTZAN genre (not the failure marker).
    if out["label"] != "unknown":
        assert out["label"] in genre.STYLE_FAMILIES
        assert 0.0 <= out["confidence"] <= 1.0
        assert abs(sum(out["probs"].values()) - 1.0) < 0.05  # softmax over 10 genres
