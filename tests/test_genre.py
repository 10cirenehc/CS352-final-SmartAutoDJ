"""Unit tests for genre -> transition-style mapping (no model required).

These exercise the pure-logic part of the genre feature: the family mapping, the
A/B combine rule, and the fact that presets toggle the existing pipeline knobs
(stretch off, bass-swap off). The model-dependent classifier is covered by the
skippable ``test_genre_smoke.py``.
"""

from __future__ import annotations

from smartautodj import align, analysis, genre


# --------------------------------------------------------------------------- #
# Style families + combine rule
# --------------------------------------------------------------------------- #
def test_each_gtzan_label_maps_to_a_known_preset():
    """Every GTZAN family must resolve to a defined preset."""
    for label, family in genre.STYLE_FAMILIES.items():
        assert family in genre.STYLE_PRESETS, f"{label} -> {family} has no preset"


def test_same_family_uses_that_style():
    # hiphop + hiphop -> urban (short, no stretch; hip-hop horn+stab+fill element).
    name, preset = genre.transition_style("hiphop", "hiphop")
    assert name == "urban"
    assert preset.bars == 2
    assert preset.tempo_tol == 0.0
    assert preset.bass_swap is False
    assert preset.bridge_element == "buildup"
    assert preset.bridge_auto is True  # hip-hop gets its element by default at tier 3


def test_reggae_maps_to_dub_siren():
    name, preset = genre.transition_style("reggae", "reggae")
    assert name == "dub"
    assert preset.bridge_element == "siren"


def test_every_preset_element_has_a_spec():
    for name, preset in genre.STYLE_PRESETS.items():
        assert preset.bridge_element in genre.BRIDGE_SPECS, name


def test_rock_and_pop_are_on_demand_only():
    # band (rock) / smooth (pop) don't add a bridge by default — only on --generate.
    assert genre.STYLE_PRESETS["band"].bridge_auto is False
    assert genre.STYLE_PRESETS["smooth"].bridge_auto is False


def test_b_sets_the_feel_for_equal_length_styles():
    # jazz(smooth,8) -> rock(band,8): equal length, so incoming B (rock) wins.
    assert genre.transition_style("jazz", "rock")[0] == "band"
    # disco(dance,16) -> classical(ambient,16): equal length -> B (ambient).
    assert genre.transition_style("disco", "classical")[0] == "ambient"


def test_clash_downgrades_to_the_longer_safer_style():
    # disco(dance,16) -> hiphop(urban,2): A wants a long blend; don't hard-cut.
    assert genre.transition_style("disco", "hiphop")[0] == "dance"
    # urban(2) -> dance(16): B is the longer one, so B sets the feel.
    assert genre.transition_style("hiphop", "disco")[0] == "dance"


def test_unknown_genre_falls_back_to_default():
    name, preset = genre.transition_style("unknown", "pop")
    assert name == "default"
    assert preset == genre.STYLE_PRESETS["default"]
    # A label outside the GTZAN taxonomy also falls back.
    assert genre.transition_style("techno", "house")[0] == "default"


def test_classify_genre_degrades_gracefully_on_bad_input():
    """Passing an array without sr must not raise — returns 'unknown'."""
    import numpy as np

    out = genre.classify_genre(np.zeros(1000, dtype="float32"), sr=None)
    assert out["label"] == "unknown"
    assert out["confidence"] == 0.0


# --------------------------------------------------------------------------- #
# Presets toggle the existing pipeline knobs
# --------------------------------------------------------------------------- #
def test_preset_tempo_tol_zero_disables_stretch(click_tracks):
    """A preset with tempo_tol=0 (urban/ambient) must yield no time-stretch."""
    a_path, b_path, sr = click_tracks
    a = analysis.analyze(a_path, backend="librosa", sr=sr)  # 120 BPM
    b = analysis.analyze(b_path, backend="librosa", sr=sr)  # 124 BPM
    preset = genre.STYLE_PRESETS["urban"]
    plan = align.select_transition_region(a, b, bars=2, tempo_tol=preset.tempo_tol)
    assert plan.stretch_ratio == 1.0  # 124/120 is outside tol=0 -> no stretch


def test_preset_bass_swap_off_means_plain_crossfade():
    """bass_swap=False presets feed an empty eq_params (plain volume crossfade)."""
    assert genre.STYLE_PRESETS["urban"].bass_swap is False
    assert genre.STYLE_PRESETS["ambient"].bass_swap is False
    assert genre.STYLE_PRESETS["dance"].bass_swap is True
