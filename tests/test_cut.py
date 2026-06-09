"""Quick-cut transition shape: fade A out -> effect over the seam -> drop B in.

Unlike the blend (A and B summed over the whole overlap), the cut is sequential:
A's tail and B's head are concatenated, not crossfaded, with B's bass swelling in
via a high-pass filter-in. These tests cover the asymmetric curve helpers and the
sequential render structure, plus a regression guard that the blend is unchanged.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from smartautodj import generative, genre, transition
from smartautodj.io import to_samples
from smartautodj.mix import mix_transition
from smartautodj.pipeline import run
from smartautodj.types import TransitionPlan


def _centroid(x: np.ndarray, sr: int) -> float:
    """Spectral centroid (Hz) — a brightness proxy."""
    X = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64)))
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    return float((f * X).sum() / (X.sum() + 1e-12))


def test_cut_fade_out_dies_before_drop():
    g = transition.cut_fade_out(1000, fade_frac=0.6)
    assert g[0] == pytest.approx(1.0, abs=1e-3)
    assert g[-1] == pytest.approx(0.0, abs=1e-6)
    assert np.all(g[700:] == 0.0)  # silent through the run-up to the drop
    assert np.all(np.diff(g) <= 1e-6)  # monotonic non-increasing


def test_cut_bridge_envelope_rises_into_drop():
    e = transition.cut_bridge_envelope(1000)
    assert e[0] == pytest.approx(0.0, abs=1e-6)
    assert e[-1] == pytest.approx(1.0)
    assert np.all(np.diff(e) >= -1e-9)  # peaks at the drop


def test_sweep_ramp_monotonic_0_to_1():
    r = transition.sweep_ramp(1000)
    assert r[0] == pytest.approx(0.0, abs=1e-6)
    assert r[-1] == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.diff(r) >= -1e-9)


def _tone(freq: float, n: int, sr: int, amp: float = 0.5) -> np.ndarray:
    return (amp * np.sin(2 * np.pi * freq * np.arange(n) / sr)).astype(np.float32)


def test_cut_render_is_sequential():
    """Output = A up to the seam end, then B from its drop onward (concatenated)."""
    sr = 22050
    na, nb = 5 * sr, 4 * sr
    a, b = _tone(220, na, sr), _tone(330, nb, sr)
    plan = TransitionPlan(tier=2, overlap_sec=1.0, region_a=(4.0, 5.0),
                          region_b=(1.0, 2.0), shape="cut")
    y, info = mix_transition(a, b, plan, sr, bridge=None, cut_sweep_sec=0.0, cut_xfade_sec=0.0)

    a1 = to_samples(5.0, sr)
    b0 = to_samples(1.0, sr)
    assert info["shape"] == "cut"
    assert len(y) == a1 + (nb - b0)  # A's kept tail + all of B from the drop
    assert abs(y[a1 - 1]) < 1e-3  # A faded to silence by the drop
    # B drops in at full level just past the declick.
    seg = y[a1 + int(0.02 * sr): a1 + int(0.5 * sr)]
    assert np.max(np.abs(seg)) > 0.3


def test_cut_highpass_woosh_swells_bass_in():
    """High-pass woosh: a sub-bass B enters quiet (low band ramped in), then full."""
    sr = 22050
    na, nb = 3 * sr, 3 * sr
    a = _tone(220, na, sr, amp=0.3)
    b = _tone(60, nb, sr, amp=0.5)  # pure sub-bass -> entirely in the low band
    plan = TransitionPlan(tier=2, overlap_sec=1.0, region_a=(2.0, 3.0),
                          region_b=(0.5, 1.5), shape="cut")
    # Hard cut (no crossfade) so the seam end == drop index is stable.
    y, info = mix_transition(a, b, plan, sr, bridge=None, cut_sweep_sec=0.5,
                             cut_woosh="highpass", cut_xfade_sec=0.0)

    drop = to_samples(3.0, sr)  # a1 == len(a_intro) + seam = where B drops
    sweep_n = info["sweep_sample"]
    assert sweep_n > 0
    early = y[drop: drop + sweep_n // 4]
    late = y[drop + sweep_n: drop + sweep_n + sweep_n // 4]
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    assert rms(early) < rms(late)  # bass swelled in over the sweep


def test_cut_lowpass_woosh_brightens_in():
    """Low-pass woosh: a broadband B enters dark and brightens (centroid rises)."""
    sr = 22050
    na, nb = 3 * sr, 3 * sr
    rng = np.random.default_rng(0)
    a = _tone(220, na, sr, amp=0.3)
    b = (0.5 * rng.standard_normal(nb)).astype(np.float32)  # broadband
    plan = TransitionPlan(tier=2, overlap_sec=1.0, region_a=(2.0, 3.0),
                          region_b=(0.5, 1.5), shape="cut")
    y, info = mix_transition(a, b, plan, sr, bridge=None, cut_sweep_sec=0.5,
                             cut_woosh="lowpass", cut_xfade_sec=0.0)
    drop = to_samples(3.0, sr)
    sweep_n = info["sweep_sample"]
    early = y[drop: drop + sweep_n // 4]
    late = y[drop + sweep_n: drop + sweep_n + sweep_n // 4]
    assert _centroid(early, sr) < _centroid(late, sr)  # filter opens dark -> bright


def _peak_freq_path(x, sr, n_fft=2048, hop=512):
    """Per-frame peak-magnitude frequency (Hz) — traces a sweeping resonance."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    out = []
    for i in range(0, len(x) - n_fft, hop):
        mag = np.abs(np.fft.rfft(x[i:i + n_fft] * np.hanning(n_fft)))
        out.append(freqs[int(np.argmax(mag))])
    return np.array(out)


def test_swept_bandpass_is_stable_and_rings_high():
    """The TPT SVF stays bounded even sweeping past fs/6 (the naive SVF would blow up)."""
    sr = 44100
    x = (0.3 * np.random.default_rng(0).standard_normal(sr)).astype(np.float32)  # 1 s noise
    y = transition.swept_bandpass_svf(x, sr, f_lo=200.0, f_hi=16000.0, q=10.0)
    assert np.all(np.isfinite(y))
    assert float(np.max(np.abs(y))) < 50.0  # bounded (resonant but not exploding)


def test_make_whoosh_sweeps_up_high():
    """The noise whoosh shows a resonant ridge climbing low -> high (a real sweep)."""
    sr = 44100
    w = transition.make_whoosh(2 * sr, sr, f_lo=200.0, f_hi=12000.0, q=10.0)
    path = _peak_freq_path(w, sr)
    early = float(np.median(path[: len(path) // 4]))
    late = float(np.median(path[-len(path) // 4:]))
    assert late > early * 3  # the resonant peak climbs a lot
    assert late > 5000.0  # ...and reaches genuinely high frequencies


def test_cut_noise_woosh_adds_high_sweep_to_seam():
    """The 'noise' woosh injects a rising high-frequency sweep into the seam."""
    sr = 44100
    na, nb = 4 * sr, 3 * sr
    a, b = _tone(220, na, sr), _tone(330, nb, sr)
    plan = TransitionPlan(tier=2, overlap_sec=2.0, region_a=(2.0, 4.0),
                          region_b=(0.5, 2.5), shape="cut")
    y, info = mix_transition(a, b, plan, sr, bridge=None, cut_sweep_sec=2.0,
                             cut_woosh="noise", cut_xfade_sec=0.0)
    assert info["woosh"] == "noise"
    s0, L = info["overlap_start_sample"], info["overlap_len_sample"]
    seam = y[s0:s0 + L]
    # The seam's late half has far more high-frequency (>5 kHz) energy than a plain
    # 220 Hz tone would — the swept whoosh put it there.
    late = seam[L // 2:]
    spec = np.abs(np.fft.rfft(late)); f = np.fft.rfftfreq(len(late), 1.0 / sr)
    hi = float(spec[f > 5000].sum()); lo = float(spec[f < 5000].sum())
    assert hi > 0.05 * lo  # meaningful high-frequency content present


def test_cut_crossfade_overlaps_riser_into_b():
    """A crossfade overlaps the riser tail with B, shortening the output vs a hard cut."""
    sr = 22050
    na, nb = 4 * sr, 3 * sr
    a, b = _tone(220, na, sr), _tone(330, nb, sr)
    plan = TransitionPlan(tier=2, overlap_sec=1.0, region_a=(3.0, 4.0),
                          region_b=(0.5, 1.5), shape="cut")
    y_hard, _ = mix_transition(a, b, plan, sr, bridge=None, cut_sweep_sec=0.0, cut_xfade_sec=0.0)
    y_x, info = mix_transition(a, b, plan, sr, bridge=None, cut_sweep_sec=0.0, cut_xfade_sec=0.25)
    xf = info["xfade_sample"]
    assert xf > 0
    assert len(y_x) == len(y_hard) - xf  # the overlap shortens the output
    # No silent sample at the seam->B boundary (the crossfade fills it).
    boundary = info["overlap_start_sample"] + info["overlap_len_sample"] - xf
    assert np.max(np.abs(y_x[boundary: boundary + xf])) > 1e-3


def test_cut_effect_leads_the_seam():
    """The added effect rises into the drop (isolated by differencing vs no-bridge).

    A's faded tail dominates the *early* seam, so total seam energy falls; the
    meaningful check is the effect's own contribution, which we recover as
    (seam_with_bridge - seam_without_bridge) since A's tail is identical in both.
    """
    sr = 22050
    na, nb = 4 * sr, 3 * sr
    a, b = _tone(220, na, sr), _tone(330, nb, sr)
    bridge = _tone(1000, sr, sr, amp=0.8)  # 1 s of "effect"
    plan = TransitionPlan(tier=3, overlap_sec=1.0, region_a=(3.0, 4.0),
                          region_b=(0.5, 1.5), shape="cut")
    a0, a1 = to_samples(3.0, sr), to_samples(4.0, sr)
    y_eff, info = mix_transition(a, b, plan, sr, bridge=bridge, cut_sweep_sec=0.0, cut_xfade_sec=0.0)
    y_none, _ = mix_transition(a, b, plan, sr, bridge=None, cut_sweep_sec=0.0, cut_xfade_sec=0.0)

    assert info.get("bridge_envelope") == "rising_into_drop"
    effect = y_eff[a0:a1] - y_none[a0:a1]  # the effect's contribution alone
    q = len(effect) // 4
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    assert rms(effect[-q:]) > rms(effect[:q])  # swells into the drop
    # Where A is dead (late seam), the effect clearly fills the seam.
    assert rms(y_eff[a1 - q:a1]) > rms(y_none[a1 - q:a1]) * 3


def test_run_cut_style_is_short_and_sequential(click_tracks, tmp_path):
    """End-to-end: --style cut gives a short overlap and a cut-shaped sidecar."""
    a_path, b_path, _ = click_tracks
    res = run(a_path, b_path, tier=2, out_dir=str(tmp_path), style="cut", make_plots=False)
    sc = json.load(open(res["sidecar"]))
    assert sc["plan"]["shape"] == "cut"
    assert sc["mix_info"]["shape"] == "cut"
    assert sc["plan"]["overlap_sec"] < 13.0  # ~5 bars (~10s), vs the 8-bar blend (~16s)


def test_riserize_brightens_into_the_drop():
    """riserize turns flat white noise into a clip that brightens toward the end."""
    sr = 22050
    rng = np.random.default_rng(0)
    noise = (0.5 * rng.standard_normal(2 * sr)).astype(np.float32)
    out = generative.riserize(noise, sr)
    q = len(out) // 4
    early, late = out[:q], out[-q:]
    assert _centroid(late, sr) > _centroid(early, sr)  # filter opens up (brighter late)
    # The synthetic uplifter swells late, so late RMS exceeds early RMS too.
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    assert rms(late) > rms(early)


def test_resolve_bridge_riserizes_long_ai_clip(tmp_path):
    """A clip LONGER than the seam is trimmed THEN riserized, so the peak isn't chopped.

    Exercises the ordering bug: if riserize ran before the trim, the bright tail would
    be cut off and the late half would NOT be brighter than the early half.
    """
    import soundfile as sf
    sr = 22050
    rng = np.random.default_rng(1)
    clip = (0.5 * rng.standard_normal(6 * sr)).astype(np.float32)  # 6 s >> 2 s seam
    p = tmp_path / "ai_clip.wav"
    sf.write(p, clip, sr)
    bridge, meta = generative.resolve_bridge(2.0, sr, 120.0, clip_path=str(p))
    assert meta["source"] == "ai-clip" and meta["riserized"] is True
    assert abs(len(bridge) - int(round(2.0 * sr))) <= 1  # fit to the seam
    q = len(bridge) // 4
    assert _centroid(bridge[-q:], sr) > _centroid(bridge[:q], sr)  # peak at the end, kept


def test_resolve_bridge_respects_no_riserize(tmp_path):
    import soundfile as sf
    sr = 22050
    rng = np.random.default_rng(2)
    clip = (0.5 * rng.standard_normal(3 * sr)).astype(np.float32)
    p = tmp_path / "ai_clip.wav"
    sf.write(p, clip, sr)
    _, meta = generative.resolve_bridge(2.0, sr, 120.0, clip_path=str(p), riserize_clip=False)
    assert meta["riserized"] is False


def test_riff_is_melodic_option():
    """The 'riff' element exists, is melodic, and builds a melody-allowing prompt."""
    spec = genre.BRIDGE_SPECS["riff"]
    assert spec.melodic is True

    class _A:  # minimal analysis stub
        bpm = 124.0
        key = {"name": "C major"}
    a = b = _A()
    fx = generative.build_bridge_prompt(a, b, 6.0, template=spec.prompt, melodic=True)
    assert "melodic riff" in fx.lower()
    assert "no melody" not in fx.lower()  # melodic wrapper, NOT the isolated-FX one
    iso = generative.build_bridge_prompt(a, b, 6.0, template=genre.BRIDGE_SPECS["riser"].prompt)
    assert "no melody" in iso.lower()  # the abstract-FX wrapper still says no melody


def test_make_placeholder_riff_is_tonal():
    """The procedural riff produces audible, tonal (not noise) audio."""
    sr = 22050
    y = generative.make_placeholder_bridge("riff", 2.0, sr=sr, bpm=120.0)
    assert float(np.max(np.abs(y))) > 0.1
    # Tonal -> energy concentrates in a few FFT bins (low spectral flatness), unlike noise.
    mag = np.abs(np.fft.rfft(y)) + 1e-9
    flatness = float(np.exp(np.mean(np.log(mag))) / np.mean(mag))  # 0=tonal, ~1=noise
    assert flatness < 0.3


def test_cut_always_riser():
    """The cut always uses its riser, even when energy/tempo would swap the effect."""
    preset = genre.STYLE_PRESETS["cut"]
    assert preset.bridge_element == "riser"
    # Big energy jump would normally force "buildup"; tempo-match would force "hits".
    assert genre.select_effect("cut", preset, 120.0, 120.0, energy_a=0.1, energy_b=0.9) == "riser"


def test_blend_default_unchanged(click_tracks, tmp_path):
    """Regression: the default (blend) path still renders the symmetric overlap."""
    a_path, b_path, _ = click_tracks
    res = run(a_path, b_path, tier=2, out_dir=str(tmp_path), style="default", make_plots=False)
    sc = json.load(open(res["sidecar"]))
    assert sc["plan"]["shape"] == "blend"
    assert "shape" not in sc["mix_info"]  # blend info dict has no cut marker
