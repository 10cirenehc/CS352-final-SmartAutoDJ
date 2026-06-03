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
from . import io as io_mod
from . import viz as viz_mod
from .mix import mix_transition


def run(
    song_a: str,
    song_b: str,
    tier: int = 2,
    out_dir: str = "outputs",
    bars: int = 8,
    overlap_sec: float | None = None,
    bridge_kind: str = "riser",
    backend: str = "auto",
    tempo_tol: float = 0.10,
    make_plots: bool = True,
    seed: int = 0,
    sr: int = DEFAULT_SR,
) -> dict:
    """Run one transition and write WAV + sidecar (+ plots). Returns a summary."""
    if tier not in (1, 2, 3):
        raise ValueError(f"tier must be 1, 2 or 3 (got {tier})")
    os.makedirs(out_dir, exist_ok=True)

    # --- load + analyze (every tier) ---
    y_a, sr = io_mod.load_audio(song_a, sr=sr)
    y_b, sr = io_mod.load_audio(song_b, sr=sr)
    a = analysis_mod.analyze(song_a, backend=backend, sr=sr)
    b = analysis_mod.analyze(song_b, backend=backend, sr=sr)

    # --- select region (tier-dependent) ---
    if tier == 1:
        # Default the baseline overlap to the same bars-based length tier 2/3 use
        # (so the tiers are compared over an equal region); allow an override.
        ov = overlap_sec if overlap_sec else bars * align_mod.METER * 60.0 / max(a.bpm, 1.0)
        plan = align_mod.baseline_plan(a.duration, b.duration, overlap_sec=ov)
    else:
        plan = align_mod.select_transition_region(
            a, b, bars=bars, tempo_tol=tempo_tol, tier=tier
        )

    # --- stretch B onto A's tempo (no-op when ratio ~= 1, e.g. tier 1) ---
    y_b_proc = align_mod.apply_stretch(y_b, plan.stretch_ratio)

    # --- generative bridge (tier 3 only) ---
    bridge = None
    if tier == 3:
        raw = gen_mod.make_placeholder_bridge(
            bridge_kind, plan.overlap_sec, sr=sr, bpm=a.bpm, seed=seed
        )
        bridge = gen_mod.prepare_clip(raw, sr, plan.overlap_sec)
        plan.bridge = {
            "kind": bridge_kind,
            "start": round(float(plan.region_a[0]), 4),
            "end": round(float(plan.region_a[1]), 4),
            "source": "procedural-placeholder",
        }

    # --- mix + evaluate ---
    y_out, info = mix_transition(y_a, y_b_proc, plan, sr, bridge=bridge)
    metrics = evaluate_mod.evaluate_all(a, b, plan, y_out, info, sr)

    # --- export WAV + sidecar (+ plots) ---
    stem = f"{_stem(song_a)}__{_stem(song_b)}__tier{tier}"
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
            "bars": bars,
            "overlap_sec": overlap_sec,
            "bridge_kind": bridge_kind if tier == 3 else None,
            "backend": backend,
            "tempo_tol": tempo_tol,
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
    }


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


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
    p.add_argument("--bars", type=int, default=8, help="overlap length in bars (tier 2/3)")
    p.add_argument("--overlap-sec", type=float, default=None,
                   help="tier-1 overlap length in seconds (default: match tier 2/3 bars)")
    p.add_argument("--bridge", default="riser", choices=("riser", "drum_fill"),
                   help="procedural bridge kind (tier 3)")
    p.add_argument("--backend", default="auto", choices=("auto", "allin1", "librosa"),
                   help="analysis backend")
    p.add_argument("--tempo-tol", type=float, default=0.10,
                   help="max fractional tempo gap to time-stretch B")
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
        backend=args.backend,
        tempo_tol=args.tempo_tol,
        make_plots=not args.no_plots,
        seed=args.seed,
    )
    print(f"tier {result['tier']} -> {result['wav']}")
    print(f"  sidecar: {result['sidecar']}")
    print(f"  plots:   {len(result['plots'])} file(s)")
    m = result["metrics"]
    print(f"  beat-align err: {m['beat_alignment']['mean_abs_error_sec']} s | "
          f"bpm gap {m['tempo_match']['bpm_gap_before']} -> {m['tempo_match']['bpm_gap_after']} | "
          f"max dB jump {m['loudness_continuity']['max_db_jump']}")


if __name__ == "__main__":
    main()
