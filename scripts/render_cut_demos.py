#!/usr/bin/env python3
"""Render the curated quick-cut (`--style cut`) demos with metadata + plots.

The four showcase clips are deterministic `--style cut` transitions (bars=7, hence
the "cut7" tag) with different transition effects. They were saved by hand as
`outputs/<id>.wav` without sidecars; this script regenerates each through the real
pipeline (the audio is byte-identical) so every clip gets its JSON sidecar and the
full plot suite, named to match the WAV (`outputs/<id>.json`, `outputs/<id>_*.png`).

`scripts/build_site_assets.py` then packages these into the demo gallery.

    python scripts/render_cut_demos.py
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

from smartautodj import pipeline

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "outputs")

# Pairs rendered as a 1/2/3 tier ladder (blend) for the objective evaluation
# section — baseline crossfade -> beat-aligned -> AI-bridge. Metrics only.
EVAL_PAIRS = [
    {"id": "victory_lap__ultimate", "a": "data/victory_lap.mp3", "b": "data/ultimate.mp3"},
    {"id": "skrillex__core", "a": "data/skrillex.mp3", "b": "data/core.mp3"},
]

# Pairs that need only tier-1 & tier-2 blends rendered (no tier-3 — it would clobber
# the curated cut demo at outputs/<id>__tier3.*). These feed the small tier-1/tier-2
# "what each tier sounds like" players next to each cut demo. The EVAL_PAIRS above
# already produce tier-1/2 for victory_lap__ultimate and skrillex__core; this list is
# just the pairs the eval ladder doesn't cover (the São Paulo AI demo).
LADDER_PAIRS = [
    {"id": "sao_paulo__put4", "a": "data/sao_paulo.mp3", "b": "data/put4.mp3"},
]

# id -> render config. `id` matches the existing hand-saved WAV stem.
DEMOS = [
    {"id": "victory_lap__ultimate__cut7_hits", "a": "data/victory_lap.mp3",
     "b": "data/ultimate.mp3", "effect": "hits",
     "title_a": "Victory Lap", "title_b": "Ultimate"},
    {"id": "victory_lap__ultimate__cut7_riser", "a": "data/victory_lap.mp3",
     "b": "data/ultimate.mp3", "effect": "riser",
     "title_a": "Victory Lap", "title_b": "Ultimate"},
    {"id": "skrillex__core__cut7_riff", "a": "data/skrillex.mp3",
     "b": "data/core.mp3", "effect": "riff",
     "title_a": "Skrillex", "title_b": "Core"},
    {"id": "skrillex__core__cut7_riser", "a": "data/skrillex.mp3",
     "b": "data/core.mp3", "effect": "riser",
     "title_a": "Skrillex", "title_b": "Core"},
]


def render_one(cfg: dict) -> str:
    """Render one cut demo and relocate its artifacts to outputs/<id>.*."""
    cid = cfg["id"]
    with tempfile.TemporaryDirectory() as tmp:
        res = pipeline.run(
            song_a=os.path.join(REPO_ROOT, cfg["a"]),
            song_b=os.path.join(REPO_ROOT, cfg["b"]),
            tier=3, style="cut", effect=cfg["effect"], out_dir=tmp,
        )
        stem = os.path.splitext(os.path.basename(res["wav"]))[0]  # <a>__<b>__tier3

        # WAV
        wav_dst = os.path.join(OUT_DIR, f"{cid}.wav")
        shutil.move(res["wav"], wav_dst)

        # Plots: {stem}_{key}.png -> {id}_{key}.png
        new_plots = []
        for p in res["plots"]:
            key = os.path.basename(p)[len(stem) + 1:]  # strip "{stem}_"
            dst = os.path.join(OUT_DIR, f"{cid}_{key}")
            shutil.move(p, dst)
            new_plots.append(os.path.relpath(dst, REPO_ROOT))

        # Sidecar: fix internal paths + tag the demo identity, write to outputs/<id>.json
        with open(res["sidecar"]) as fh:
            sc = json.load(fh)
        sc["outputs"]["wav"] = os.path.relpath(wav_dst, REPO_ROOT)
        sc["outputs"]["plots"] = new_plots
        sc["demo"] = {
            "id": cid, "effect": cfg["effect"],
            "title_a": cfg["title_a"], "title_b": cfg["title_b"],
            "shape": "cut",
        }
        with open(os.path.join(OUT_DIR, f"{cid}.json"), "w") as fh:
            json.dump(sc, fh, indent=2)
    return cid


def render_eval_tiers(cfg: dict, tiers: tuple[int, ...] = (1, 2, 3)) -> None:
    """Render a pair at the given tiers (blend, metrics only) into outputs/<id>__tier{N}.*."""
    for tier in tiers:
        pipeline.run(
            song_a=os.path.join(REPO_ROOT, cfg["a"]),
            song_b=os.path.join(REPO_ROOT, cfg["b"]),
            tier=tier, style="auto", out_dir=OUT_DIR, make_plots=False,
        )


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    for cfg in DEMOS:
        print(f"rendering {cfg['id']} (cut / {cfg['effect']}) ...", flush=True)
        render_one(cfg)
        print(f"  -> outputs/{cfg['id']}.json (+ plots)")
    for cfg in EVAL_PAIRS:
        print(f"rendering eval tiers 1/2/3 for {cfg['id']} ...", flush=True)
        render_eval_tiers(cfg)
    for cfg in LADDER_PAIRS:
        print(f"rendering tiers 1/2 for {cfg['id']} ...", flush=True)
        render_eval_tiers(cfg, tiers=(1, 2))
    print(f"\nDone: {len(DEMOS)} cut demos + {len(EVAL_PAIRS)} eval ladders "
          f"+ {len(LADDER_PAIRS)} tier-1/2 ladders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
