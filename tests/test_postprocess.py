"""Unit tests for the bridge post-processing DSP (loudness / limit / duck / carve)."""

from __future__ import annotations

import numpy as np

from smartautodj import postprocess as pp


def test_loudness_match_sets_relative_rms():
    rng = np.random.default_rng(0)
    program = rng.standard_normal(20000).astype(np.float32) * 0.3
    layer = np.zeros(20000, dtype=np.float32)
    layer[1000] = layer[9000] = 1.0  # transient: low RMS, full peak
    out = pp.loudness_match(layer, program, rel_db=-3.0)
    ratio_db = 20 * np.log10(_rms(out) / _rms(program))
    assert abs(ratio_db - (-3.0)) < 0.3


def test_soft_limit_caps_peaks_below_ceiling():
    x = np.linspace(-3, 3, 1000).astype(np.float32)  # well past the ceiling
    y = pp.soft_limit(x, ceiling=0.5)
    assert np.max(np.abs(y)) <= 0.5 + 1e-6
    # Small signals pass roughly linearly (within the soft knee).
    small = np.array([0.01, -0.02], dtype=np.float32)
    assert np.allclose(pp.soft_limit(small, 0.5), small, atol=2e-3)


def test_sidechain_duck_attenuates_under_loud_control():
    sr = 22050
    program = np.ones(sr, dtype=np.float32)
    control = np.zeros(sr, dtype=np.float32)
    control[: sr // 2] = 1.0  # loud in the first half, silent in the second
    out = pp.sidechain_duck(program, control, sr, depth_db=6.0, smooth_ms=10.0)
    # First half ducked (well below 1), second half ~untouched.
    assert out[sr // 4] < 0.7
    assert out[-1] > 0.95


def test_spectral_carve_reduces_energy_where_bridge_lives():
    sr = 22050
    t = np.arange(sr) / sr
    program = (0.5 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)
    control = (0.5 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)  # same band -> carve it
    out = pp.spectral_carve(program, control, sr, depth_db=12.0)
    assert _rms(out) < _rms(program)  # program energy reduced where the bridge sits


def test_integrate_bridge_blends_and_reports():
    sr = 22050
    rng = np.random.default_rng(1)
    program = (rng.standard_normal(sr) * 0.3).astype(np.float32)
    bridge = np.zeros(sr, dtype=np.float32)
    bridge[100] = bridge[5000] = 1.0
    overlap, info = pp.integrate_bridge(program, bridge, sr)
    assert overlap.shape == program.shape
    assert info["bridge_rel_db"] == -3.0 and info["bridge_spectral_carve"] is True
    assert not np.allclose(overlap, program)  # the bridge actually changed the mix


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))
