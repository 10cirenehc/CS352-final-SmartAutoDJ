"""End-to-end and unit tests on synthetic click tracks."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from smartautodj import align, analysis, generative, structure, transition
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


def test_beat_period_robust_to_missed_beats():
    """A dropped beat (2x gap) must not skew the period estimate.

    Regression: a least-squares slope inflated the period when beat_this missed
    beats on a real song, which inverted the stretch direction. The robust
    median-of-inliers estimate must ignore the gap.
    """
    beats = np.array([0.0, 0.5, 1.0, 1.5, 2.5, 3.0, 3.5, 4.0])  # missing beat at 2.0
    assert abs(align._beat_period(beats) - 0.5) < 0.02


def test_refine_stretch_direction(click_tracks):
    """If A is faster than B, B must be sped UP (rate > 1), not slowed down."""
    a_path, b_path, sr = click_tracks
    a = analysis.analyze(a_path, backend="librosa", sr=sr)  # 120 BPM
    b = analysis.analyze(b_path, backend="librosa", sr=sr)  # 124 BPM
    # Make a synthetic "A faster than B" case by swapping roles: stretch A toward
    # B should be >1 since b(124) is faster than a(120) -> speed A up.
    rate_speed_up = align._refine_stretch(b, a, 0.10)  # match A(120) up to B(124)
    assert rate_speed_up > 1.0
    rate_slow_down = align._refine_stretch(a, b, 0.10)  # match B(124) down to A(120)
    assert rate_slow_down < 1.0


def test_decide_stretch_within_and_outside_tolerance():
    # 124 vs 120 -> within 10% -> stretch toward A's tempo.
    assert align.decide_stretch(120.0, 124.0, tol=0.10) == pytest.approx(120 / 124, rel=1e-6)
    # 120 vs 160 -> too far -> no stretch.
    assert align.decide_stretch(120.0, 160.0, tol=0.10) == 1.0


def test_equal_power_fades_preserve_power():
    fo, fi = transition.fade_curves(1000, "equal_power")
    assert np.allclose(fo ** 2 + fi ** 2, 1.0, atol=1e-6)


def test_bass_swap_has_no_midpoint_dropout():
    """Equal-power bass handoff keeps total low-band energy ~constant.

    Regression: the old linear schedule sent both bass gains to zero at t=0.5,
    causing an audible bass dropout in the middle of every tier-2/3 transition.
    """
    bass_out, bass_in = transition.bass_swap_curves(2000, center=0.5, width=0.2)
    power_sum = bass_out ** 2 + bass_in ** 2
    assert np.allclose(power_sum, 1.0, atol=1e-6)  # never dips toward silence
    # Endpoints: A's bass fully on at the start, B's fully on at the end.
    assert bass_out[0] == pytest.approx(1.0)
    assert bass_in[-1] == pytest.approx(1.0)


def test_bands_recombine(click_tracks):
    a_path, _, sr = click_tracks
    y, sr = load_audio(a_path, sr=sr)
    low, high = transition.split_bands(y[:sr], sr, cutoff=200.0, order=4)
    assert np.allclose(low + high, y[:sr], atol=1e-3)


def test_cue_skips_quiet_intro(intro_drop_track):
    """The incoming switch point must land in the loud body, not the quiet intro."""
    path, sr, intro_sec = intro_drop_track
    b = analysis.analyze(path, backend="librosa", sr=sr)
    # Energy in the body should clearly exceed the intro (sanity on the fixture).
    t, info = structure.choose_incoming_switch(b, overlap=4.0, phrase_bars=2)
    assert t >= intro_sec - 0.5  # chosen point is at/after the drop, not t~0
    assert info["reason"] in ("first_high_energy_phrase", "max_energy_phrase")


def test_estimate_key_returns_valid_key(click_tracks):
    a_path, _, sr = click_tracks
    y, sr = load_audio(a_path, sr=sr)
    key = structure.estimate_key(y, sr)
    assert key["mode"] in ("major", "minor")
    assert 0 <= key["tonic"] < 12
    # Same track is perfectly compatible with itself.
    assert structure.key_compatibility(key, key) == 1.0


def test_resolve_bridge_prefers_clip_else_procedural(click_tracks, tmp_path):
    a_path, _, sr = click_tracks
    # No clip -> procedural fallback.
    bridge, meta = generative.resolve_bridge(4.0, sr, 120.0, clip_path=None, seed=1)
    assert meta["source"] == "procedural-placeholder"
    assert bridge.size == int(round(4.0 * sr))
    # Existing clip -> loaded as ai-clip and length-matched.
    import soundfile as sf
    clip_path = tmp_path / "clip.wav"
    sf.write(clip_path, np.zeros(int(8 * sr), dtype="float32"), sr)
    bridge2, meta2 = generative.resolve_bridge(4.0, sr, 120.0, clip_path=str(clip_path))
    assert meta2["source"] == "ai-clip"
    assert bridge2.size == int(round(4.0 * sr))


def test_loudness_match_sets_rms_relative_to_reference():
    """A transient layer (low RMS, high peak) must be scaled by RMS, not peak, so
    it sits at a known level under the program — else it's masked/inaudible."""
    from smartautodj import mix
    rng = np.random.default_rng(0)
    program = rng.standard_normal(20000).astype(np.float32) * 0.3
    # Stab-like layer: near-silent with two spikes -> very low RMS, full peak.
    layer = np.zeros(20000, dtype=np.float32)
    layer[1000] = 1.0
    layer[9000] = 1.0
    out = mix.loudness_match(layer, program, rel_db=-3.0)
    pr = float(np.sqrt(np.mean(program ** 2)))
    lr = float(np.sqrt(np.mean(out ** 2)))
    assert abs(20 * np.log10(lr / pr) - (-3.0)) < 0.3  # within 0.3 dB of target


@pytest.mark.parametrize("kind", ["riser", "drum_fill", "siren", "pad", "cymbal"])
def test_make_placeholder_bridge_kinds(kind):
    """Every genre procedural element renders finite audio of the right length."""
    y = generative.make_placeholder_bridge(kind, 1.0, sr=22050, bpm=120.0, seed=2)
    assert y.shape[0] == 22050
    assert np.all(np.isfinite(y))


def test_beat_align_clip_locks_to_target_tempo(click_tracks):
    """A clip is time-stretched toward the mix tempo; silence passes through."""
    a_path, _, sr = click_tracks  # 120 BPM click track
    y, sr = load_audio(a_path, sr=sr)
    out, est = generative.beat_align_clip(y, sr, target_bpm=140.0)
    assert est > 0
    assert len(out) <= len(y)  # sped up toward 140 -> shorter
    # Silence guard: no tempo, returned unchanged.
    z = np.zeros(sr, dtype=np.float32)
    out2, est2 = generative.beat_align_clip(z, sr, 140.0)
    assert est2 == 0.0 and len(out2) == len(z)


def test_build_bridge_prompt_includes_tempo_and_key(click_tracks):
    a_path, b_path, sr = click_tracks
    a = analysis.analyze(a_path, backend="librosa", sr=sr)
    b = analysis.analyze(b_path, backend="librosa", sr=sr)
    prompt = generative.build_bridge_prompt(a, b, 8.0)
    assert "BPM" in prompt
    assert "second" in prompt


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
        backend="librosa", make_plots=True, sr=sr, style="default",
    )
    assert os.path.exists(result["wav"])
    assert os.path.exists(result["sidecar"])
    # 9 track-level/continuity base plots; tier >=2 adds 3 structure/EQ visuals;
    # tier 3 adds the mel-spectrogram bridge plot (see viz.render_all).
    assert len(result["plots"]) == (9 if tier == 1 else 12 if tier == 2 else 13)
    for p in result["plots"]:
        assert os.path.exists(p)

    sidecar = json.loads(open(result["sidecar"]).read())
    assert sidecar["tier"] == tier
    # The chosen style is recorded for inspection/evaluation.
    assert sidecar["plan"]["selection"]["style"] == "default"
    assert sidecar["params"]["style"] == "default"
    assert "beat_alignment" in sidecar["metrics"]
    assert "tempo_match" in sidecar["metrics"]
    assert "loudness_continuity" in sidecar["metrics"]


def test_tier3_bridge_changes_audio(click_tracks, tmp_path):
    """Tier 3 must differ audibly from tier 2 (the additive bridge)."""
    a_path, b_path, sr = click_tracks
    r2 = run(a_path, b_path, tier=2, out_dir=str(tmp_path / "t2"),
             bars=4, backend="librosa", make_plots=False, sr=sr, style="default")
    r3 = run(a_path, b_path, tier=3, out_dir=str(tmp_path / "t3"),
             bars=4, backend="librosa", make_plots=False, sr=sr, style="default")
    y2, _ = load_audio(r2["wav"], sr=sr)
    y3, _ = load_audio(r3["wav"], sr=sr)
    n = min(len(y2), len(y3))
    assert not np.allclose(y2[:n], y3[:n], atol=1e-4)


def test_tier2_beats_tier1_on_alignment(click_tracks, tmp_path):
    """Beat-aligned tier 2 should not be worse than the naive tier-1 baseline."""
    a_path, b_path, sr = click_tracks
    r1 = run(a_path, b_path, tier=1, out_dir=str(tmp_path / "t1"),
             bars=4, backend="librosa", make_plots=False, sr=sr, style="default")
    r2 = run(a_path, b_path, tier=2, out_dir=str(tmp_path / "t2"),
             bars=4, backend="librosa", make_plots=False, sr=sr, style="default")
    e1 = r1["metrics"]["beat_alignment"]["mean_abs_error_sec"]
    e2 = r2["metrics"]["beat_alignment"]["mean_abs_error_sec"]
    assert e2 is not None
    if e1 is not None:
        assert e2 <= e1 + 1e-6
