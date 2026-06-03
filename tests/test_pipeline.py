"""End-to-end and unit tests on synthetic click tracks."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from smartautodj import align, analysis, transition
from smartautodj.io import load_audio
from smartautodj.pipeline import run


# --------------------------------------------------------------------------- #
# Unit-level
# --------------------------------------------------------------------------- #
def test_analyze_recovers_tempo(click_tracks):
    a_path, _, sr = click_tracks
    res = analysis.analyze(a_path, backend="librosa", sr=sr)
    # Tempo trackers commonly land on the true BPM or an octave (half/double).
    candidates = [120.0, 60.0, 240.0]
    assert min(abs(res.bpm - c) for c in candidates) < 6.0
    assert res.beats.size > 4
    assert res.downbeats.size >= 1
    assert res.backend == "librosa"


def test_decide_stretch_within_and_outside_tolerance():
    # 124 vs 120 -> within 10% -> stretch toward A's tempo.
    assert align.decide_stretch(120.0, 124.0, tol=0.10) == pytest.approx(120 / 124, rel=1e-6)
    # 120 vs 160 -> too far -> no stretch.
    assert align.decide_stretch(120.0, 160.0, tol=0.10) == 1.0


def test_equal_power_fades_preserve_power():
    fo, fi = transition.fade_curves(1000, "equal_power")
    assert np.allclose(fo ** 2 + fi ** 2, 1.0, atol=1e-6)


def test_bands_recombine(click_tracks):
    a_path, _, sr = click_tracks
    y, sr = load_audio(a_path, sr=sr)
    low, high = transition.split_bands(y[:sr], sr, cutoff=200.0, order=4)
    assert np.allclose(low + high, y[:sr], atol=1e-3)


def test_alignment_reduces_beat_error(click_tracks):
    a_path, b_path, sr = click_tracks
    a = analysis.analyze(a_path, backend="librosa", sr=sr)
    b = analysis.analyze(b_path, backend="librosa", sr=sr)
    plan = align.select_transition_region(a, b, bars=4)
    a_t, b_t = align.matched_beats_in_overlap(a, b, plan)
    assert a_t.size > 0
    # Downbeat-anchored + tempo-matched: beats should line up tightly.
    assert float(np.mean(np.abs(a_t - b_t))) < 0.06


# --------------------------------------------------------------------------- #
# End-to-end across tiers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tier", [1, 2, 3])
def test_run_tier_produces_outputs(click_tracks, tmp_path, tier):
    a_path, b_path, sr = click_tracks
    out = tmp_path / "outputs"
    result = run(
        a_path, b_path, tier=tier, out_dir=str(out), bars=4,
        backend="librosa", make_plots=True, sr=sr,
    )
    assert os.path.exists(result["wav"])
    assert os.path.exists(result["sidecar"])
    assert len(result["plots"]) == 4
    for p in result["plots"]:
        assert os.path.exists(p)

    sidecar = json.loads(open(result["sidecar"]).read())
    assert sidecar["tier"] == tier
    assert "beat_alignment" in sidecar["metrics"]
    assert "tempo_match" in sidecar["metrics"]
    assert "loudness_continuity" in sidecar["metrics"]


def test_tier3_bridge_changes_audio(click_tracks, tmp_path):
    """Tier 3 must differ audibly from tier 2 (the additive bridge)."""
    a_path, b_path, sr = click_tracks
    r2 = run(a_path, b_path, tier=2, out_dir=str(tmp_path / "t2"),
             bars=4, backend="librosa", make_plots=False, sr=sr)
    r3 = run(a_path, b_path, tier=3, out_dir=str(tmp_path / "t3"),
             bars=4, backend="librosa", make_plots=False, sr=sr)
    y2, _ = load_audio(r2["wav"], sr=sr)
    y3, _ = load_audio(r3["wav"], sr=sr)
    n = min(len(y2), len(y3))
    assert not np.allclose(y2[:n], y3[:n], atol=1e-4)


def test_tier2_beats_tier1_on_alignment(click_tracks, tmp_path):
    """Beat-aligned tier 2 should not be worse than the naive tier-1 baseline."""
    a_path, b_path, sr = click_tracks
    r1 = run(a_path, b_path, tier=1, out_dir=str(tmp_path / "t1"),
             bars=4, backend="librosa", make_plots=False, sr=sr)
    r2 = run(a_path, b_path, tier=2, out_dir=str(tmp_path / "t2"),
             bars=4, backend="librosa", make_plots=False, sr=sr)
    e1 = r1["metrics"]["beat_alignment"]["mean_abs_error_sec"]
    e2 = r2["metrics"]["beat_alignment"]["mean_abs_error_sec"]
    assert e2 is not None
    if e1 is not None:
        assert e2 <= e1 + 1e-6
