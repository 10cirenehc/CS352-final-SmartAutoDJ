"""Pipeline orchestrator + CLI.

Composes the stages and selects a tier by toggling stages on/off:

    Tier 1  load -> analyze -> [baseline region] -------------------> mix
    Tier 2  load -> analyze -> select+stretch+align -> transition --> mix
    Tier 3  Tier 2 + generative bridge ----------------------------> mix

Both tracks are always analysed (so beats/structure feed the metrics and plots
in every tier), but tier 1 ignores that analysis for *region selection* — it
just crossfades A's tail into B's head. Every run writes a WAV, a JSON sidecar
(analysis + plan + metrics), and plots.

    python -m smartautodj.pipeline --song-a A.wav --song-b B.wav --tier 2 --out outputs/
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import DEFAULT_SR
from . import align as align_mod
from . import analysis as analysis_mod
from . import evaluate as evaluate_mod
from . import generative as gen_mod
from . import genre as genre_mod
from . import io as io_mod
from . import remote as remote_mod
from . import viz as viz_mod
from .mix import mix_transition


def run(
    song_a: str,
    song_b: str,
    tier: int = 2,
    out_dir: str = "outputs",
    bars: int | None = None,
    overlap_sec: float | None = None,
    bridge_kind: str | None = None,
    bridge_clip: str | None = None,
    generate: bool = False,
    bridge_eval_q: int = 2,
    bridge_gain_db: float = 0.0,
    bridge_duck_db: float = 6.0,
    bridge_limit: bool = True,
    bridge_spectral: bool = True,
    backend: str = "auto",
    tempo_tol: float | None = None,
    cue_method: str | None = None,
    style: str = "auto",
    genre: str | None = None,
    effect: str | None = None,
    bpm_a: float | None = None,
    bpm_b: float | None = None,
    cue_a: float | None = None,
    cue_b: float | None = None,
    fade_sharpness: float = 2.5,
    cut_sweep_bars: float = 2.0,
    cut_woosh: str = "noise",
    cut_xfade_bars: float = 2.0,
    cut_fade_frac: float = 0.65,
    riserize: bool = True,
    make_plots: bool = True,
    seed: int = 0,
    sr: int = DEFAULT_SR,
) -> dict:
    """Run one transition and write WAV + sidecar (+ plots). Returns a summary.

    Genre-driven style: when ``style="auto"`` the two tracks are classified and a
    :class:`genre.StylePreset` sets the overlap ``bars``, the time-stretch gate
    ``tempo_tol``, the ``cue_method``, the bass-swap EQ, and the tier-3 riser.
    Pass ``style`` as a named preset (e.g. ``"dance"``) to skip detection, or
    ``genre="hiphop"`` to force the label. Any explicitly passed ``bars`` /
    ``tempo_tol`` / ``cue_method`` overrides the preset (the library/CLI default
    for those is ``None`` = "let the style decide").
    """
    if tier not in (1, 2, 3):
        raise ValueError(f"tier must be 1, 2 or 3 (got {tier})")
    os.makedirs(out_dir, exist_ok=True)

    # --- load + analyze (every tier) ---
    y_a, sr = io_mod.load_audio(song_a, sr=sr)
    y_b, sr = io_mod.load_audio(song_b, sr=sr)
    a = analysis_mod.analyze(song_a, backend=backend, sr=sr)
    b = analysis_mod.analyze(song_b, backend=backend, sr=sr)
    # Manual tempo override (beat trackers mis-detect tempo octave/half-time on some
    # clips). Sets the BPM used for the stretch decision and the generation prompt;
    # the beat GRID stays as detected (phase alignment is a separate concern).
    if bpm_a:
        a.bpm = float(bpm_a)
    if bpm_b:
        b.bpm = float(bpm_b)

    # --- pick the transition style (genre-driven) ---
    style_name, preset = _resolve_style(style, genre, a, b, y_a, y_b, sr)
    # Explicit args win; otherwise the preset fills the knobs.
    eff_bars = bars if bars is not None else preset.bars
    eff_tol = tempo_tol if tempo_tol is not None else preset.tempo_tol
    eff_cue = cue_method if cue_method is not None else preset.cue_method

    # --- select region (tier-dependent) ---
    if tier == 1:
        # Default the baseline overlap to the same bars-based length tier 2/3 use
        # (so the tiers are compared over an equal region); allow an override.
        ov = overlap_sec if overlap_sec else eff_bars * align_mod.METER * 60.0 / max(a.bpm, 1.0)
        plan = align_mod.baseline_plan(a.duration, b.duration, overlap_sec=ov)
    else:
        plan = align_mod.select_transition_region(
            a, b, bars=eff_bars, tempo_tol=eff_tol, tier=tier, cue_method=eff_cue,
            y_a=y_a, y_b=y_b, sr=sr, cue_a=cue_a, cue_b=cue_b, fade_sharpness=fade_sharpness,
            shape=preset.shape,
        )
        # Preset toggles the bass-swap EQ off (e.g. urban/ambient) by clearing
        # eq_params, which transition.process_overlap reads as "plain crossfade".
        if not preset.bass_swap:
            plan.eq_params = {}

    # Record the chosen style + both genres in the plan selection -> sidecar.
    plan.selection = {
        **plan.selection,
        "style": style_name,
        "genre_a": a.genre.get("label", "unknown"),
        "genre_b": b.genre.get("label", "unknown"),
    }

    # --- stretch B onto A's tempo (no-op when ratio ~= 1, e.g. tier 1) ---
    y_b_proc = align_mod.apply_stretch(y_b, plan.stretch_ratio, sr=sr)

    stem = f"{_stem(song_a)}__{_stem(song_b)}__tier{tier}"
    pair = f"{_stem(song_a)}__{_stem(song_b)}"

    # --- generative bridge (tier 3 only; genre picks the element) ---
    bridge = None
    bridge_kind_used = None
    # The genre's style preset selects a genre-authentic element (EDM riser, hip-hop
    # air-horn+fill, dub siren, ambient pad, ...). Default-on styles add it at tier 3;
    # on-demand styles (rock/pop) only when the user opts in via --generate / a clip.
    # Effect: manual --effect wins; else smart default (genre + transition context).
    if effect and effect in genre_mod.BRIDGE_SPECS:
        effect_name = effect
    else:
        inc = (plan.selection or {}).get("incoming", {})
        effect_name = genre_mod.select_effect(
            style_name, preset, a.bpm, b.bpm, inc.get("energy_a"), inc.get("energy_b")
        )
    spec = genre_mod.BRIDGE_SPECS[effect_name]
    # Cache key includes the element so different effects don't cross-serve each
    # other's clip for the same pair (and the sidecar can't mislabel the audio).
    default_clip = os.path.join("assets", "generated", f"{pair}__{spec.element}_bridge.wav")
    clip_path = bridge_clip or (default_clip if os.path.exists(default_clip) else None)
    if tier == 3 and (preset.bridge_auto or generate or clip_path):
        prompt = gen_mod.build_bridge_prompt(a, b, plan.overlap_sec, template=spec.prompt,
                                             melodic=spec.melodic)
        # --generate: synthesize the clip now on Modal (CUDA, musicgen-style),
        # conditioned on the A->B boundary reference, and drop it where resolve_bridge
        # expects it — so one command produces the real AI bridge (no Colab round-trip).
        if clip_path is None and generate:
            ref = gen_mod.extract_boundary_reference(y_a, y_b_proc, plan, sr)
            # eval_q (1..6): how strongly MusicGen-Style adheres to the reference
            # timbre. The boundary reference is a muddy A+B crossfade, so conditioning
            # HARD on it produces muffled nonsense — keep it LOOSE (eval_q=1) by default
            # and let the text prompt (genre/key/tempo/effect) carry the style.
            # --bridge-eval-q overrides for the rare case you want the songs' timbre.
            eval_q = 1 if bridge_eval_q == 2 else bridge_eval_q
            clip_path = remote_mod.generate_bridge_to_file(
                ref, sr, prompt, plan.overlap_sec, default_clip,
                params={"eval_q": eval_q},
            )
        # Procedural fallback kind comes from the genre spec unless --bridge overrides.
        bridge_kind_used = bridge_kind or spec.proc_kind
        # Only abstract upward sweeps get riserized (an EQ filter-sweep into the drop);
        # drum fills / hits / sirens / pads / riffs must NOT be filtered into a sweep
        # (that's what made the hip-hop fill sound muffled).
        riserize_this = riserize and effect_name in ("riser", "sweep")
        bridge, src = gen_mod.resolve_bridge(
            plan.overlap_sec, sr, a.bpm, clip_path=clip_path, kind=bridge_kind_used, seed=seed,
            riserize_clip=riserize_this,
        )
        plan.bridge = {
            "start": round(float(plan.region_a[0]), 4),
            "end": round(float(plan.region_a[1]), 4),
            "element": spec.element,  # the actually-selected effect (manual or context)
            "prompt": prompt,
            **src,
        }
        # On the procedural fallback, export the conditioning artifacts (boundary
        # reference WAV + where to drop the generated clip) for the manual/Modal path.
        if src.get("source") != "ai-clip":
            ref = gen_mod.extract_boundary_reference(y_a, y_b_proc, plan, sr)
            ref_path = io_mod.save_wav(
                os.path.join(out_dir, f"{stem}_boundary_ref.wav"), ref, sr
            )
            plan.bridge["boundary_ref"] = ref_path
            plan.bridge["expected_clip_path"] = default_clip

    # --- mix + evaluate ---
    # Cut-shape knobs (bars -> seconds at A's tempo; ignored by the blend path):
    # the woosh length (B's filter-in) and the riser->B crossfade length. 0 disables.
    bar_dur = align_mod.METER * 60.0 / max(a.bpm, 1.0)
    # Pass seconds directly (0 bars -> 0.0 -> disabled; never None, which means
    # "use the mix default" for direct callers).
    cut_sweep_sec = max(0.0, cut_sweep_bars) * bar_dur
    cut_xfade_sec = max(0.0, cut_xfade_bars) * bar_dur
    y_out, info = mix_transition(y_a, y_b_proc, plan, sr, bridge=bridge,
                                 bridge_gain_db=bridge_gain_db, bridge_duck_db=bridge_duck_db,
                                 bridge_limit=bridge_limit, bridge_spectral=bridge_spectral,
                                 cut_sweep_sec=cut_sweep_sec, cut_woosh=cut_woosh,
                                 cut_xfade_sec=cut_xfade_sec, cut_fade_frac=cut_fade_frac)
    metrics = evaluate_mod.evaluate_all(a, b, plan, y_out, info, sr)

    # --- export WAV + sidecar (+ plots) ---
    wav_path = io_mod.save_wav(os.path.join(out_dir, f"{stem}.wav"), y_out, sr)
    plot_paths = []
    if make_plots:
        plot_paths = viz_mod.render_all(
            y_a, y_b_proc, y_out, a, b, plan, info, sr, out_dir, stem
        )

    sidecar = {
        "tier": tier,
        "song_a": song_a,
        "song_b": song_b,
        "sr": sr,
        "params": {
            "bars": eff_bars,
            "overlap_sec": overlap_sec,
            "bridge_kind": bridge_kind_used,  # the effective procedural kind (or None)
            "generate": generate if tier == 3 else None,
            "backend": backend,
            "tempo_tol": eff_tol,
            "cue_method": eff_cue,
            "style": style_name,
            "genre_override": genre,
            "seed": seed,
        },
        "analysis_a": a.to_dict(),
        "analysis_b": b.to_dict(),
        "plan": plan.to_dict(),
        "mix_info": info,
        "metrics": metrics,
        "outputs": {"wav": wav_path, "plots": plot_paths},
    }
    sidecar_path = os.path.join(out_dir, f"{stem}.json")
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    return {
        "wav": wav_path,
        "sidecar": sidecar_path,
        "plots": plot_paths,
        "metrics": metrics,
        "tier": tier,
        "style": style_name,
        "genre_a": a.genre.get("label", "unknown"),
        "genre_b": b.genre.get("label", "unknown"),
    }


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _resolve_style(style, genre, a, b, y_a, y_b, sr):
    """Resolve the transition style + preset, classifying genre when needed.

    Sets ``a.genre`` / ``b.genre`` (for the sidecar) as a side effect.

      * ``genre="LABEL"``  -> force that GTZAN label for both tracks (no model).
      * ``style="auto"``   -> classify A and B, combine via genre.transition_style.
      * ``style="<name>"`` -> use that named preset directly (no model).

    Anything unknown / failing falls back to the ``default`` preset (today's
    behaviour), per the project's de-risking rule.
    """
    if style == "cut":
        # `cut` is always a riser/riff (effect not genre-picked — GTZAN can't tell
        # EDM from hip-hop), but we still classify (or honor --genre) so the detected
        # genre can be CONVEYED in the generation prompt (the style cue, since the
        # audio conditioning is kept loose). Prompt-only; does not change the effect.
        if genre is not None:
            a.genre = {"label": genre, "confidence": None, "probs": {}, "source": "override"}
            b.genre = dict(a.genre)
        else:
            a.genre = genre_mod.classify_genre(y_a, sr=sr)
            b.genre = genre_mod.classify_genre(y_b, sr=sr)
        return "cut", genre_mod.STYLE_PRESETS["cut"]

    if genre is not None:
        a.genre = {"label": genre, "confidence": None, "probs": {}, "source": "override"}
        b.genre = dict(a.genre)
        return genre_mod.transition_style(genre, genre)

    if style == "auto":
        # Reuse the already-loaded audio; classify_genre resamples to 16 kHz.
        a.genre = genre_mod.classify_genre(y_a, sr=sr)
        b.genre = genre_mod.classify_genre(y_b, sr=sr)
        return genre_mod.transition_style(a.genre["label"], b.genre["label"])

    preset = genre_mod.STYLE_PRESETS.get(style)
    if preset is None:
        return "default", genre_mod.STYLE_PRESETS["default"]
    return style, preset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="smartautodj",
        description="Generate a structure-aware DJ transition from A into B.",
    )
    p.add_argument("--song-a", required=True, help="path to outgoing track A")
    p.add_argument("--song-b", required=True, help="path to incoming track B")
    p.add_argument("--tier", type=int, default=2, choices=(1, 2, 3),
                   help="1=baseline crossfade, 2=beat-aligned, 3=AI-enhanced")
    p.add_argument("--out", default="outputs", help="output directory")
    p.add_argument("--style", default="auto",
                   choices=("auto", "dance", "urban", "techno", "dub", "band",
                            "smooth", "ambient", "default", "cut"),
                   help="transition style: auto (detect genre) or a named preset "
                        "('cut' = punchy quick-cut: fade A out, short buildup, drop B in)")
    p.add_argument("--genre", default=None,
                   help="force a GTZAN genre label for both tracks (skip detection), "
                        "e.g. hiphop, disco, classical")
    p.add_argument("--effect", default=None,
                   choices=("riser", "buildup", "hits", "impact", "sweep", "siren", "pad", "riff"),
                   help="override the tier-3 transition effect (default: chosen from "
                        "genre + transition context). 'riff' = a melodic run in the tracks' "
                        "key/timbre instead of an abstract FX")
    p.add_argument("--bpm-a", type=float, default=None,
                   help="override the detected tempo of A (when the beat tracker mis-detects)")
    p.add_argument("--bpm-b", type=float, default=None,
                   help="override the detected tempo of B")
    p.add_argument("--cue-a", type=float, default=None,
                   help="manually place A's exit at this time (s); snapped to the nearest "
                        "downbeat. Use when auto-detection misses the chorus.")
    p.add_argument("--cue-b", type=float, default=None,
                   help="manually place B's entry at this time (s); snapped to a downbeat")
    p.add_argument("--fade-sharpness", type=float, default=2.5,
                   help="crossfade quickness: 1=gradual equal-power, higher=quicker/punchier swap")
    p.add_argument("--cut-sweep-bars", type=float, default=2.0,
                   help="quick-cut only (--style cut): bars over which the woosh/swoosh runs; "
                        "0 = no woosh")
    p.add_argument("--cut-woosh", default="noise",
                   choices=("noise", "bandpass", "highpass", "lowpass", "none"),
                   help="quick-cut only: the swoosh into the drop — noise (swept-resonant-noise "
                        "whoosh layer, the aggressive default), bandpass (swept resonant "
                        "band-pass on B's entrance), highpass/lowpass (gentle 2-band B filter-in), "
                        "or none")
    p.add_argument("--cut-xfade-bars", type=float, default=2.0,
                   help="quick-cut only: bars over which B crossfades in over the swoosh "
                        "(the 'smoosh'; B enters on a downbeat); 0 = hard cut")
    p.add_argument("--cut-fade-frac", type=float, default=0.65,
                   help="quick-cut only: fraction of the seam over which A fades to silence "
                        "(lower = A fades out sooner/more; 0.65 default)")
    p.add_argument("--bars", type=int, default=None,
                   help="overlap length in bars (tier 2/3); overrides the style preset")
    p.add_argument("--overlap-sec", type=float, default=None,
                   help="tier-1 overlap length in seconds (default: match tier 2/3 bars)")
    p.add_argument("--bridge", default=None,
                   choices=("riser", "drum_fill", "siren", "pad", "cymbal"),
                   help="override the procedural bridge kind (tier 3 fallback); "
                        "default: the genre style's element")
    p.add_argument("--bridge-clip", default=None,
                   help="path to a generated AI bridge clip (tier 3); "
                        "defaults to assets/generated/<pair>_bridge.wav if present")
    p.add_argument("--generate", action="store_true",
                   help="tier 3: generate the AI bridge on Modal (CUDA musicgen-style) "
                        "if no clip exists — needs `.[gen]` + a deployed infra/modal_bridge.py")
    p.add_argument("--bridge-eval-q", type=int, default=2,
                   help="MusicGen-Style conditioning strength 1..6 (higher = bridge "
                        "reproduces the songs more / blends; lower = cleaner isolated "
                        "effect from the prompt); used when --generate")
    p.add_argument("--bridge-gain-db", type=float, default=0.0,
                   help="bridge loudness (RMS) relative to the program in the overlap; "
                        "higher = more prominent/punchy (e.g. +3), lower = subtler (e.g. -6)")
    p.add_argument("--bridge-duck-db", type=float, default=6.0,
                   help="how much to duck the two songs UNDER the bridge (0 = no ducking)")
    p.add_argument("--no-bridge-limit", action="store_true",
                   help="disable soft-limiting the bridge transients")
    p.add_argument("--no-spectral-carve", action="store_true",
                   help="disable the per-frequency spectral carve of the program under the bridge")
    p.add_argument("--no-riserize", action="store_true",
                   help="tier 3: don't impose a rising filter/pitch sweep on the generated "
                        "clip (by default the AI clip is 'riserized' so it sweeps up into the drop)")
    p.add_argument("--backend", default="auto",
                   choices=("auto", "beat_this", "librosa", "allin1"),
                   help="analysis backend (auto: beat_this -> librosa)")
    p.add_argument("--tempo-tol", type=float, default=None,
                   help="max fractional tempo gap to time-stretch B; overrides the preset")
    p.add_argument("--cue", default=None, choices=("match", "energy", "novelty"),
                   help="cue selection: match (cross-similarity A-exit+B-entry, default), "
                        "energy, or novelty; overrides the style preset")
    p.add_argument("--no-plots", action="store_true", help="skip rendering plots")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the bridge")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run(
        song_a=args.song_a,
        song_b=args.song_b,
        tier=args.tier,
        out_dir=args.out,
        bars=args.bars,
        overlap_sec=args.overlap_sec,
        bridge_kind=args.bridge,
        bridge_clip=args.bridge_clip,
        generate=args.generate,
        bridge_eval_q=args.bridge_eval_q,
        bridge_gain_db=args.bridge_gain_db,
        bridge_duck_db=args.bridge_duck_db,
        bridge_limit=not args.no_bridge_limit,
        bridge_spectral=not args.no_spectral_carve,
        backend=args.backend,
        tempo_tol=args.tempo_tol,
        cue_method=args.cue,
        style=args.style,
        genre=args.genre,
        effect=args.effect,
        bpm_a=args.bpm_a,
        bpm_b=args.bpm_b,
        cue_a=args.cue_a,
        cue_b=args.cue_b,
        fade_sharpness=args.fade_sharpness,
        cut_sweep_bars=args.cut_sweep_bars,
        cut_woosh=args.cut_woosh,
        cut_xfade_bars=args.cut_xfade_bars,
        cut_fade_frac=args.cut_fade_frac,
        riserize=not args.no_riserize,
        make_plots=not args.no_plots,
        seed=args.seed,
    )
    print(f"tier {result['tier']} -> {result['wav']}")
    print(f"  style: {result['style']} "
          f"(genre A={result['genre_a']}, B={result['genre_b']})")
    print(f"  sidecar: {result['sidecar']}")
    print(f"  plots:   {len(result['plots'])} file(s)")
    m = result["metrics"]
    print(f"  beat-align err: {m['beat_alignment']['mean_abs_error_sec']} s | "
          f"bpm gap {m['tempo_match']['bpm_gap_before']} -> {m['tempo_match']['bpm_gap_after']} | "
          f"max dB jump {m['loudness_continuity']['max_db_jump']}")


if __name__ == "__main__":
    main()
