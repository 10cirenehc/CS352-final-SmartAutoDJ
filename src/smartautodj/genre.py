"""Genre classification + genre-driven transition-style presets.

Optional upgrade (mirrors the ``allin1``/``beat_this`` pattern): a small
pretrained genre classifier decides *how* the transition should sound, not just
where it happens. Without ``transformers`` installed — or if the model fails to
load — :func:`classify_genre` returns ``"unknown"`` and the pipeline falls back
to today's default beat-aligned style. Nothing breaks.

Pipeline:  classify A and B -> map each label to a *style family* ->
:func:`transition_style` combines the two into one :class:`StylePreset` that
overrides knobs the pipeline already supports (overlap ``bars``, ``tempo_tol``
for the time-stretch gate, the bass-swap EQ toggle, the cue method) and selects a
genre-authentic **bridge element** (:class:`BridgeSpec` in :data:`BRIDGE_SPECS`) —
an EDM riser, a hip-hop air-horn + fill, a dub siren, an ambient pad wash, etc. —
that the tier-3 layer generates (MusicGen) or synthesises procedurally.

Course-topic tie-in (Deep Learning / Embeddings): the classifier is a small
distilHuBERT transformer finetuned on GTZAN — a learned audio embedding feeding
a genre head — the same family of model as the neural beat tracker and MusicGen.

GTZAN taxonomy is only 10 genres (blues, classical, country, disco, hiphop,
jazz, metal, pop, reggae, rock) — *no* house/techno/EDM/ambient. We map ``disco``
-> ``dance`` and ``classical`` -> ``ambient`` as proxies; real EDM often lands on
disco/pop. The ``--style``/``--genre`` overrides let a user force the intended
style when the model mislabels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Small distilHuBERT finetuned on GTZAN — the canonical HF audio-course genre
# model. CPU-friendly; ~0.3 s per excerpt once the model is cached.
MODEL_NAME = "sanchit-gandhi/distilhubert-finetuned-gtzan"
TARGET_SR = 16000  # the model's expected input rate
EXCERPT_SEC = 30.0  # classify a mid-track excerpt (skip intro/outro)

_PIPELINE = None  # cached transformers pipeline (loaded once, lazily)


# --------------------------------------------------------------------------- #
# Style families + presets (the genre -> transition-mechanics table)
# --------------------------------------------------------------------------- #

# GTZAN label -> transition-style family. (reggae is split out as its own `dub`
# family — the dub siren is its signature transition element.)
STYLE_FAMILIES: dict[str, str] = {
    "disco": "dance",
    "hiphop": "urban",
    "reggae": "dub",
    "rock": "band",
    "metal": "band",
    "country": "band",
    "blues": "band",
    "jazz": "smooth",
    "pop": "smooth",
    "classical": "ambient",
}


@dataclass(frozen=True)
class BridgeSpec:
    """A genre-authentic transition element (the tier-3 additive layer).

    Real DJs add genre-specific material during a transition — an EDM riser, a
    hip-hop air-horn + fill, a dub siren, an ambient pad wash. ``prompt`` is the
    MusicGen text template (``{bpm}``/``{key}``/``{dur}`` placeholders; leans on
    sustained texture and says "no melody", which is what the model does well);
    ``proc_kind`` is the offline procedural fallback synthesised by
    :func:`generative.make_placeholder_bridge`.
    """

    element: str
    prompt: str
    proc_kind: str


# element -> spec. Prompts describe a *sensible transition effect* in the genre's
# style (not a literal recreation/blend of the tracks). The MusicGen-Style audio
# conditioning carries the actual timbre of the songs; the prompt just names the
# kind of effect that makes sense. Keep them short and effect-focused.
BRIDGE_SPECS: dict[str, BridgeSpec] = {
    "riser": BridgeSpec(
        "riser",
        "{bpm} BPM rising filtered-noise uplifter sweeping up into the drop, "
        "building tension, about {dur} seconds",
        "riser",
    ),
    "buildup": BridgeSpec(
        "buildup",
        "{bpm} BPM transition buildup, rising white noise with an accelerating snare "
        "and hi-hat roll building tension into the drop, punchy, about {dur} seconds",
        "drum_fill",
    ),
    "hits": BridgeSpec(
        "hits",
        "{bpm} BPM a few big hard-hitting impact booms on the beat leading into the "
        "drop, punchy and powerful, about {dur} seconds",
        "drum_fill",
    ),
    "impact": BridgeSpec(
        "impact",
        "{bpm} BPM one huge impact boom and reverse-cymbal crash hitting the drop, "
        "about {dur} seconds",
        "cymbal",
    ),
    "sweep": BridgeSpec(
        "sweep",
        "{bpm} BPM white-noise sweep transition effect, atmospheric, about {dur} seconds",
        "riser",
    ),
    "siren": BridgeSpec(
        "siren",
        "{bpm} BPM dub siren transition effect with tape echo, about {dur} seconds",
        "siren",
    ),
    "pad": BridgeSpec(
        "pad",
        "an ambient swell and long reverb wash{key}, slow evolving, about {dur} seconds",
        "pad",
    ),
}


@dataclass(frozen=True)
class StylePreset:
    """Genre-driven overrides for the existing transition knobs.

    Every field maps onto a parameter the pipeline already threads through, so
    applying a preset is parameter-passing, not new DSP:

    * ``bars`` -> overlap length (``align.select_transition_region``)
    * ``tempo_tol`` -> time-stretch gate (``align.decide_stretch``; ``0`` disables)
    * ``bass_swap`` -> equal-power bass-swap EQ (``transition.process_overlap``)
    * ``cue_method`` -> incoming cue strategy ("energy" / "novelty")
    * ``bridge_element`` -> which :data:`BRIDGE_SPECS` element the tier-3 layer uses
    * ``bridge_auto`` -> add that element by default at tier 3 (True), or only when
      the user explicitly opts in via ``--generate`` / an existing clip (False)
    """

    bars: int
    tempo_tol: float
    bass_swap: bool
    cue_method: str
    bridge_element: str
    bridge_auto: bool


# Numbers are starting points to tune by ear. ``default`` reproduces today's
# behaviour and is used whenever genre is unknown. ``techno`` and ``dub`` are
# manual-only ``--style`` choices: GTZAN can't detect techno, and reggae auto-maps
# to ``dub`` via STYLE_FAMILIES. Coverage policy (genre-appropriate): dance/urban/
# techno/dub/ambient add their element by default; rock & pop only on --generate.
STYLE_PRESETS: dict[str, StylePreset] = {
    # house/EDM: long beatmatched blend, bass-swap, build into the drop.
    "dance": StylePreset(16, 0.10, True, "match", "buildup", True),
    # hip-hop: short, punchy, no stretch/EQ — a buildup (or hard hits when matched).
    "urban": StylePreset(2, 0.0, False, "match", "buildup", True),
    # techno (manual): long blend, atmospheric noise sweep.
    "techno": StylePreset(16, 0.10, True, "match", "sweep", True),
    # reggae/dancehall: punchy, signature dub siren.
    "dub": StylePreset(2, 0.0, False, "match", "siren", True),
    # rock/metal/country/blues: a single impact accent, on-demand only.
    "band": StylePreset(8, 0.06, True, "match", "impact", False),
    # jazz/pop: subtle sweep "glue", on-demand only.
    "smooth": StylePreset(8, 0.10, True, "match", "sweep", False),
    # classical: long crossfade, no stretch/EQ; ambient pad wash.
    "ambient": StylePreset(16, 0.0, False, "match", "pad", True),
    # today's default behaviour (also the unknown-genre fallback).
    "default": StylePreset(8, 0.10, True, "match", "riser", True),
}


def select_effect(
    style_name: str, preset: StylePreset,
    a_bpm: float, b_bpm: float, energy_a, energy_b,
) -> str:
    """Pick the appropriate transition effect from the genre default + context.

    Smart default (overridable via ``--effect``): a big energy jump *into* B reads
    as a drop -> a **buildup**; when the two tracks are tempo-matched and the style
    is punchy (dance/hip-hop), **hard hits** land on the beat; otherwise the genre's
    default effect.
    """
    base = preset.bridge_element
    if energy_a is not None and energy_b is not None and (energy_b - energy_a) > 0.12:
        return "buildup"  # B drops in noticeably louder -> build into it
    tempo_matched = (
        a_bpm > 0 and b_bpm > 0 and abs(a_bpm - b_bpm) / max(a_bpm, b_bpm) < 0.06
    )
    if tempo_matched and style_name in ("dance", "urban"):
        return "hits"  # locked tempo -> on-beat hits hit hard
    return base


def transition_style(genre_a: str, genre_b: str) -> tuple[str, StylePreset]:
    """Combine two tracks' genres into one transition style + preset.

    Rule: the **incoming track B sets the feel** (we drop *into* B). But if A's
    style calls for a longer/safer blend than B's, we don't hard-cut out of it —
    we take the longer of the two styles. Returns ``("default", ...)`` whenever
    either genre is unknown / outside the GTZAN taxonomy.
    """
    fam_a = STYLE_FAMILIES.get(genre_a)
    fam_b = STYLE_FAMILIES.get(genre_b)
    if fam_a is None or fam_b is None:
        return "default", STYLE_PRESETS["default"]
    if fam_a == fam_b:
        chosen = fam_b
    elif STYLE_PRESETS[fam_a].bars > STYLE_PRESETS[fam_b].bars:
        # A wants a longer blend than B -> downgrade to the safer (longer) style
        # rather than hard-cutting across a stylistic gap (e.g. dance -> urban).
        chosen = fam_a
    else:
        chosen = fam_b  # B sets the feel (equal length or B is the longer one)
    return chosen, STYLE_PRESETS[chosen]


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

def _get_pipeline():
    """Load + cache the transformers audio-classification pipeline once."""
    global _PIPELINE
    if _PIPELINE is None:
        from transformers import pipeline  # lazy: optional dependency

        _PIPELINE = pipeline("audio-classification", model=MODEL_NAME, device="cpu")
    return _PIPELINE


def _mid_excerpt(y: np.ndarray, sr: int, secs: float = EXCERPT_SEC) -> np.ndarray:
    """Center ``secs`` of audio (skip intro/outro, where genre cues are weakest)."""
    n = int(secs * sr)
    if y.size > n:
        start = (y.size - n) // 2
        y = y[start : start + n]
    return y


def classify_genre(path_or_y, sr: int | None = None) -> dict:
    """Classify one track's genre with the pretrained GTZAN model.

    ``path_or_y`` may be an audio file path (loaded + resampled to 16 kHz mono)
    or an already-loaded 1-D array (``sr`` then required). Classifies a ~30 s
    mid-track excerpt. Returns ``{"label", "confidence", "probs"}``; on any
    failure (model/transformers absent, bad audio) returns
    ``{"label": "unknown", "confidence": 0.0, "probs": {}}`` so callers never
    crash — the pipeline then uses the default style.
    """
    try:
        import librosa

        if isinstance(path_or_y, str):
            y, _ = librosa.load(path_or_y, sr=TARGET_SR, mono=True)
        else:
            y = np.asarray(path_or_y, dtype=np.float32).reshape(-1)
            if sr is None:
                raise ValueError("sr required when passing an audio array")
            if sr != TARGET_SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

        y = _mid_excerpt(y.astype(np.float32), TARGET_SR)
        clf = _get_pipeline()
        out = clf({"array": y, "sampling_rate": TARGET_SR}, top_k=None)
        probs = {o["label"]: round(float(o["score"]), 4) for o in out}
        top = max(out, key=lambda o: o["score"])
        return {
            "label": str(top["label"]),
            "confidence": round(float(top["score"]), 4),
            "probs": probs,
        }
    except Exception as exc:  # optional model: degrade gracefully
        return {"label": "unknown", "confidence": 0.0, "probs": {}, "error": str(exc)}
