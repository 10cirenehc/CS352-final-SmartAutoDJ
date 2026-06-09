#!/usr/bin/env python3
"""Build the static GitHub Pages showcase assets for SmartAutoDJ.

Reads the (gitignored, large) pipeline outputs in ``outputs/`` and produces a small,
committed set of web assets under ``docs/``:

  - ``docs/assets/audio/<id>__tier{N}.mp3`` : the transition WAV, trimmed to a short
    window around the actual transition and encoded as a mono MP3 (so the repo stays
    small and a listener hears the transition immediately, not 2 min of song A first).
  - ``docs/assets/plots/<id>__tier{N}_<key>.png`` : the curated plots per tier.
  - ``docs/manifest.json`` : a slim, relative-path projection of the sidecars that the
    page (``docs/js/app.js``) renders the gallery from.

Generation is NOT live -- the AI bridge clips are pre-rendered (Colab/MusicGen) and the
local pipeline treats them as static assets, so the website only ever shows pre-rendered
audio + images. This script just packages what the pipeline already wrote.

Run from the repo root:

    python scripts/build_site_assets.py

Requires ``ffmpeg`` on PATH (already a project dependency). ``pngquant`` is used to shrink
PNGs if present, but is optional.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from typing import Optional

# Optional: the cross-tier comparison chart needs the (heavy) analysis stack.
# Guarded so the script still packages audio/plots for users without it installed.
try:
    from smartautodj import viz as _viz
except Exception:  # pragma: no cover - environment without the package
    _viz = None

# --- Paths ------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
DOCS = os.path.join(REPO_ROOT, "docs")
AUDIO_DIR = os.path.join(DOCS, "assets", "audio")
PLOTS_DIR = os.path.join(DOCS, "assets", "plots")

# --- Encoding / trimming config --------------------------------------------
PAD_SEC = 7.0          # seconds of context kept on each side of the overlap region
MP3_BITRATE = "96k"    # mono @ 96k -> ~360 KB for a ~28 s clip; plenty for a web demo

# Google Form embed URL for the listening survey. Create the form by hand
# (Send -> <> embed) and paste its URL here, then re-run this script.
SURVEY_URL = ""  # e.g. "https://docs.google.com/forms/d/e/XXXX/viewform?embedded=true"

# Curated demos to publish. Edit this list to choose which song pairs appear.
# `id` must match the output stem `<id>__tier{N}.wav`. Titles are display-only;
# they fall back to the id halves if omitted.
# NOTE (copyright): publishing clips on a public site reintroduces the concern that
# made `data/`/`outputs/` gitignored. Prefer royalty-free pairs (FMA / YT Audio Library)
# for anything you push publicly; the transition-only window also helps.
CURATED = [
    {
        "id": "victory_lap__ultimate",
        "title_a": "Victory Lap",
        "title_b": "Ultimate",
    },
    # Synthetic fixture -- short and not musically convincing; useful for testing only.
    # {"id": "demo_a__demo_b", "title_a": "Demo A (synthetic)", "title_b": "Demo B (synthetic)"},
]

# Which plot keys to copy per tier, in display order. Tier 1 (baseline) lacks the
# structure/EQ plots; tier 3 adds the mel-spectrogram with the AI bridge region.
# The "story" plots (beat scatter, loudness, structure) lead; signal plots trail.
PLOT_POLICY = {
    1: ["beatmatch", "loudness", "keywheel", "ssm", "genre", "spectrogram", "fades"],
    2: ["beatmatch", "loudness", "structure", "ribbon", "keywheel", "ssm",
        "genre", "bassswap", "spectrogram", "fades"],
    3: ["beatmatch", "loudness", "structure", "ribbon", "keywheel", "ssm",
        "genre", "bassswap", "melbridge", "spectrogram", "fades"],
}

TIER_LABELS = {
    1: "Baseline crossfade",
    2: "Beat-aligned",
    3: "AI-enhanced (bridge)",
}


# --- Helpers ----------------------------------------------------------------
def sidecar_path(stem_id: str, tier: int) -> str:
    return os.path.join(OUT_DIR, f"{stem_id}__tier{tier}.json")


def load_sidecar(stem_id: str, tier: int) -> Optional[dict]:
    path = sidecar_path(stem_id, tier)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def transition_window(sidecar: dict) -> tuple[float, float]:
    """Return (start_sec, dur_sec) for a window around the transition overlap."""
    sr = sidecar["sr"]
    mix = sidecar["mix_info"]
    ov0 = mix["overlap_start_sample"] / sr
    ov_len = mix["overlap_len_sample"] / sr
    out_len = mix["out_len_sample"] / sr
    start = max(0.0, ov0 - PAD_SEC)
    dur = ov_len + 2 * PAD_SEC
    # don't run past the end of the rendered output
    dur = min(dur, out_len - start)
    return round(start, 3), round(dur, 3)


def _output_grid_clip(sidecar: dict, key: str, win_start: float, win_dur: float) -> list:
    """Beat/downbeat times on the trimmed-MP3 clock (``key`` = "beats"/"downbeats").

    The published MP3 is trimmed to a window of the OUTPUT timeline, but the
    analysis beats live in three different clocks. We map them onto the output and
    then onto the clip:
      * A is laid down 1:1, so A's beats up to the overlap end are already output
        times (verified: ``region_a[0] == overlap_start``).
      * B enters at the overlap; its (stretched) beats map to
        ``overlap_start + (t/stretch_ratio - region_b[0])``.
    Both are then shifted by ``-win_start`` and clipped to ``[0, win_dur]``."""
    sr = sidecar["sr"]
    mi = sidecar["mix_info"]
    pl = sidecar["plan"]
    ov_start = mi["overlap_start_sample"] / sr
    ov_end = (mi["overlap_start_sample"] + mi["overlap_len_sample"]) / sr
    rate = pl.get("stretch_ratio") or 1.0
    rb0 = pl["region_b"][0]
    out = []
    for t in sidecar["analysis_a"].get(key, []):
        if t <= ov_end + 1e-6:  # A is 1:1 with the output up to the overlap end
            out.append(t)
    for t in sidecar["analysis_b"].get(key, []):
        ot = ov_start + (t / rate - rb0)  # B's entry maps onto the output clock
        if ot >= ov_start - 1e-6:
            out.append(ot)
    w1 = win_start + win_dur
    clip = sorted({round(t - win_start, 3) for t in out if win_start - 1e-6 <= t <= w1 + 1e-6})
    return clip


def viz_block(sidecar: dict, win_start: float, win_dur: float) -> dict:
    """Per-tier data the interactive player needs: clip-relative grids + tempo."""
    sr = sidecar["sr"]
    mi = sidecar["mix_info"]
    pl = sidecar["plan"]
    ov_start = mi["overlap_start_sample"] / sr
    ov_len = mi["overlap_len_sample"] / sr
    a, b = sidecar["analysis_a"], sidecar["analysis_b"]
    return {
        "beat_times": _output_grid_clip(sidecar, "beats", win_start, win_dur),
        "downbeat_times": _output_grid_clip(sidecar, "downbeats", win_start, win_dur),
        "overlap": {"start": round(ov_start - win_start, 3), "dur": round(ov_len, 3)},
        "bpm_a": round(a["bpm"], 2) if a.get("bpm") else None,
        "bpm_b": round(b["bpm"], 2) if b.get("bpm") else None,
        "stretch_ratio": round(pl.get("stretch_ratio") or 1.0, 5),
    }


def encode_mp3(src_wav: str, dst_mp3: str, start: float, dur: float) -> None:
    """Trim [start, start+dur] from src_wav and encode a mono MP3 to dst_mp3."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start}", "-t", f"{dur}",  # -ss before -i = fast seek
        "-i", src_wav,
        "-ac", "1", "-b:a", MP3_BITRATE,
        dst_mp3,
    ]
    subprocess.run(cmd, check=True)


def encode_mp3_full(src_wav: str, dst_mp3: str) -> None:
    """Encode the whole WAV (untrimmed) to a mono MP3 — the full-transition playbar."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src_wav,
         "-ac", "1", "-b:a", MP3_BITRATE, dst_mp3],
        check=True,
    )


def copy_png(src: str, dst: str) -> None:
    shutil.copy2(src, dst)
    if shutil.which("pngquant"):
        # Lossy shrink in place; --force overwrites, --skip-if-larger is a safety net.
        subprocess.run(
            ["pngquant", "--quality=65-85", "--force", "--skip-if-larger",
             "--output", dst, dst],
            check=False,
        )


def _r(v, n):
    """Round, tolerating None (some metrics are undefined for quick-cut transitions)."""
    return round(v, n) if isinstance(v, (int, float)) else None


def slim_metrics(sidecar: dict) -> dict:
    m = sidecar["metrics"]
    return {
        "beat_align_err_sec": _r(m["beat_alignment"]["mean_abs_error_sec"], 4),
        "bpm_gap_before": _r(m["tempo_match"]["bpm_gap_before"], 2),
        "bpm_gap_after": _r(m["tempo_match"]["bpm_gap_after"], 2),
        "max_db_jump": _r(m["loudness_continuity"]["max_db_jump"], 2),
    }


def key_name(analysis: dict) -> Optional[str]:
    key = analysis.get("key")
    if isinstance(key, dict):
        return key.get("name")
    return None


def build_tier(stem_id: str, tier: int) -> Optional[dict]:
    """Build one tier entry: trim audio, copy plots, return the manifest dict."""
    sidecar = load_sidecar(stem_id, tier)
    if sidecar is None:
        print(f"  ! tier{tier}: no sidecar, skipping")
        return None

    wav_rel = sidecar["outputs"]["wav"]  # e.g. "outputs/<id>__tier{N}.wav"
    src_wav = os.path.join(REPO_ROOT, wav_rel)
    if not os.path.exists(src_wav):
        print(f"  ! tier{tier}: WAV missing ({wav_rel}), skipping")
        return None

    # --- audio ---
    start, dur = transition_window(sidecar)
    mp3_name = f"{stem_id}__tier{tier}.mp3"
    encode_mp3(src_wav, os.path.join(AUDIO_DIR, mp3_name), start, dur)

    # --- plots (only those the policy asks for AND the sidecar actually wrote) ---
    available = {os.path.basename(p) for p in sidecar["outputs"]["plots"]}
    plots: dict[str, str] = {}
    for key in PLOT_POLICY[tier]:
        png_name = f"{stem_id}__tier{tier}_{key}.png"
        if png_name not in available:
            print(f"  - tier{tier}: plot '{key}' not produced, skipping")
            continue
        src_png = os.path.join(OUT_DIR, png_name)
        if not os.path.exists(src_png):
            print(f"  ! tier{tier}: plot file missing ({png_name}), skipping")
            continue
        copy_png(src_png, os.path.join(PLOTS_DIR, png_name))
        plots[key] = f"assets/plots/{png_name}"

    entry = {
        "label": TIER_LABELS[tier],
        "audio": f"assets/audio/{mp3_name}",
        "window_sec": {"start": start, "dur": dur},
        "plots": plots,
        "metrics": slim_metrics(sidecar),
        "viz": viz_block(sidecar, start, dur),
    }

    bridge = sidecar.get("plan", {}).get("bridge")
    if tier == 3 and isinstance(bridge, dict):
        entry["bridge_prompt"] = bridge.get("prompt")
        entry["bridge_source"] = bridge.get("source")

    return entry


def build_pair(cfg: dict) -> Optional[dict]:
    stem_id = cfg["id"]
    print(f"Building pair: {stem_id}")
    tiers: dict[str, dict] = {}
    # pull display metadata from any tier that has it (tier 1 is fine)
    meta_sidecar = (
        load_sidecar(stem_id, 1)
        or load_sidecar(stem_id, 2)
        or load_sidecar(stem_id, 3)
    )
    if meta_sidecar is None:
        print(f"  ! no sidecars found for '{stem_id}', skipping pair")
        return None

    for tier in (1, 2, 3):
        entry = build_tier(stem_id, tier)
        if entry is not None:
            tiers[str(tier)] = entry

    if not tiers:
        print(f"  ! no tiers built for '{stem_id}', skipping pair")
        return None

    a, b = meta_sidecar["analysis_a"], meta_sidecar["analysis_b"]
    parts = stem_id.split("__")
    pair = {
        "id": stem_id,
        "title_a": cfg.get("title_a") or (parts[0] if parts else stem_id),
        "title_b": cfg.get("title_b") or (parts[1] if len(parts) > 1 else ""),
        "bpm_a": round(a.get("bpm"), 1) if a.get("bpm") else None,
        "bpm_b": round(b.get("bpm"), 1) if b.get("bpm") else None,
        "key_a": key_name(a),
        "key_b": key_name(b),
        "tiers": tiers,
    }

    # Cross-tier "what each tier buys" chart (needs all tiers' metrics at once).
    comparison = build_comparison_plot(stem_id)
    if comparison:
        pair["comparison_plot"] = comparison
    return pair


def build_comparison_plot(stem_id: str) -> Optional[str]:
    """Render the tier-comparison bar chart into PLOTS_DIR; return its rel path."""
    if _viz is None:
        print("  - comparison chart skipped (smartautodj not importable)")
        return None
    metrics_by_tier = {
        tier: sc.get("metrics", {})
        for tier in (1, 2, 3)
        if (sc := load_sidecar(stem_id, tier)) is not None
    }
    if len(metrics_by_tier) < 2:
        return None
    png_name = f"{stem_id}__comparison.png"
    dst = os.path.join(PLOTS_DIR, png_name)
    _viz.plot_tier_comparison(metrics_by_tier, dst)
    if shutil.which("pngquant"):
        subprocess.run(["pngquant", "--quality=65-85", "--force", "--skip-if-larger",
                        "--output", dst, dst], check=False)
    return f"assets/plots/{png_name}"


# --- quick-cut demos (single-render showcase clips) -------------------------
# Each is one `--style cut` transition (rendered by scripts/render_cut_demos.py
# into outputs/<id>.json + <id>_*.png). The gallery features these four.
# Each entry: {id, title_a?, title_b?}. The cut demos (rendered by
# render_cut_demos.py) carry a `demo` tag with titles; entries generated directly
# by the CLI (no tag) supply titles here and derive effect/shape from the plan.
CUT_DEMOS = [
    {"id": "victory_lap__ultimate__cut7_hits"},
    {"id": "victory_lap__ultimate__cut7_riser"},
    {"id": "skrillex__core__cut7_riff"},
    {"id": "skrillex__core__cut7_riser"},
    {"id": "sao_paulo__put4__tier3", "title_a": "São Paulo", "title_b": "PUT4"},
]

# Plots published per demo, in display order (the "story" plots lead). The
# loudness-continuity plot is omitted here — it duplicates the "max loudness jump"
# metric card already shown above the gallery.
DEMO_PLOT_ORDER = [
    "keywheel", "ssm", "structure", "ribbon", "genre",
    "melbridge", "beatmatch", "bassswap", "spectrogram", "fades", "waveforms",
]

EFFECT_LABELS = {
    "hits": "Air-horn + vocal stab", "riser": "Riser sweep",
    "riff": "Melodic riff", "buildup": "Buildup", "siren": "Dub siren",
    "pad": "Pad wash", "cymbal": "Cymbal swell", "sweep": "Noise sweep",
}


def build_ladder(pair_id: str) -> list:
    """Tier-1 & tier-2 audio snippets of a demo's pair, for the small side-by-side
    players. Each rung is the same pair rendered one tier "dumber" than the featured
    cut: tier 1 = naive crossfade, tier 2 = beat-aligned blend. Reuses the eval-ladder
    blend renders (outputs/<pair>__tier{N}.wav). Returns [] if they aren't present."""
    rungs = []
    for tier in (1, 2):
        sc = load_sidecar(pair_id, tier)
        if sc is None:
            continue
        src = os.path.join(REPO_ROOT, sc["outputs"]["wav"])
        if not os.path.exists(src):
            continue
        start, dur = transition_window(sc)
        mp3 = f"{pair_id}__tier{tier}.mp3"
        encode_mp3(src, os.path.join(AUDIO_DIR, mp3), start, dur)
        rungs.append({
            "tier": tier,
            "label": TIER_LABELS[tier],
            "audio": f"assets/audio/{mp3}",
            "max_db_jump": _r(sc["metrics"]["loudness_continuity"]["max_db_jump"], 1),
        })
    return rungs


def load_demo_sidecar(cid: str) -> Optional[dict]:
    path = os.path.join(OUT_DIR, f"{cid}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def build_demo(cfg: dict) -> Optional[dict]:
    """Package one demo (audio + plots + metrics + viz) for the manifest.

    ``cfg`` is ``{id, title_a?, title_b?}``. Title/effect/shape come from the
    sidecar's ``demo`` tag when present (render_cut_demos.py), else from ``cfg``
    and the plan (``plan.bridge.element`` / ``plan.shape``) for clips rendered
    directly by the CLI."""
    cid = cfg["id"]
    sc = load_demo_sidecar(cid)
    if sc is None:
        print(f"  ! demo {cid}: no sidecar, skipping")
        return None
    src_wav = os.path.join(REPO_ROOT, sc["outputs"]["wav"])
    if not os.path.exists(src_wav):
        print(f"  ! demo {cid}: WAV missing, skipping")
        return None

    start, dur = transition_window(sc)
    encode_mp3(src_wav, os.path.join(AUDIO_DIR, f"{cid}.mp3"), start, dur)
    # Full untrimmed transition (plain playbar at the bottom of the card).
    encode_mp3_full(src_wav, os.path.join(AUDIO_DIR, f"{cid}__full.mp3"))

    # Map produced plot files ({cid}_{key}.png) to keys, then publish in order.
    avail = {}
    for rel in sc["outputs"]["plots"]:
        base = os.path.basename(rel)
        if base.startswith(f"{cid}_") and base.endswith(".png"):
            avail[base[len(cid) + 1:-4]] = os.path.join(REPO_ROOT, rel)
    plots: dict[str, str] = {}
    for key in DEMO_PLOT_ORDER:
        src = avail.get(key)
        if src and os.path.exists(src):
            png_name = f"{cid}_{key}.png"
            copy_png(src, os.path.join(PLOTS_DIR, png_name))
            plots[key] = f"assets/plots/{png_name}"

    demo = sc.get("demo", {})
    a, b = sc["analysis_a"], sc["analysis_b"]
    plan = sc.get("plan", {})
    bridge = plan.get("bridge") or {}
    parts = cid.split("__")
    effect = demo.get("effect") or bridge.get("element")
    return {
        "id": cid,
        "title_a": cfg.get("title_a") or demo.get("title_a") or (parts[0] if parts else cid),
        "title_b": cfg.get("title_b") or demo.get("title_b") or (parts[1] if len(parts) > 1 else ""),
        "effect": effect,
        "effect_label": EFFECT_LABELS.get(effect, effect),
        "shape": demo.get("shape") or plan.get("shape", "blend"),
        "audio": f"assets/audio/{cid}.mp3",
        "audio_full": f"assets/audio/{cid}__full.mp3",
        "ladder": build_ladder("__".join(parts[:2])),
        "window_sec": {"start": start, "dur": dur},
        "plots": plots,
        "metrics": slim_metrics(sc),
        "viz": viz_block(sc, start, dur),
        "bpm_a": round(a["bpm"], 1) if a.get("bpm") else None,
        "bpm_b": round(b["bpm"], 1) if b.get("bpm") else None,
        "key_a": key_name(a),
        "key_b": key_name(b),
        "genre_a": (a.get("genre") or {}).get("label"),
        "genre_b": (b.get("genre") or {}).get("label"),
        "bridge_prompt": bridge.get("prompt"),
        "bridge_source": bridge.get("source"),
    }


# --- objective evaluation: tier 1/2/3 ladders compared per pair --------------
EVAL_PAIRS = [
    {"id": "victory_lap__ultimate", "title": "Victory Lap → Ultimate"},
    {"id": "skrillex__core", "title": "Skrillex → Core"},
]


def build_comparison(cfg: dict) -> Optional[dict]:
    """Build one pair's tier-1/2/3 comparison (chart + per-tier metric numbers)."""
    pid = cfg["id"]
    by_tier = {t: sc for t in (1, 2, 3) if (sc := load_sidecar(pid, t)) is not None}
    if len(by_tier) < 2:
        print(f"  ! eval {pid}: need >=2 tier sidecars, skipping")
        return None
    chart = build_comparison_plot(pid)
    tiers = {str(t): slim_metrics(sc) for t, sc in by_tier.items()}
    return {"id": pid, "title": cfg["title"], "chart": chart, "tiers": tiers}


def dir_size_human(path: str) -> str:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024 or unit == "GB":
            return f"{total:.1f} {unit}"
        total /= 1024


def main() -> int:
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found on PATH (brew install ffmpeg).", file=sys.stderr)
        return 1

    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Featured gallery = the quick-cut demos (rendered by scripts/render_cut_demos.py).
    print("Building cut demos...")
    demos = [d for cfg in CUT_DEMOS if (d := build_demo(cfg)) is not None]
    if not demos:
        print("ERROR: no demos built. Run `python scripts/render_cut_demos.py` first.",
              file=sys.stderr)
        return 1

    # Objective evaluation: tier 1/2/3 comparison per pair.
    print("Building tier comparisons...")
    comparisons = [c for cfg in EVAL_PAIRS if (c := build_comparison(cfg)) is not None]

    manifest = {
        # Date is intentionally left for the committer to note; avoids nondeterminism.
        "generated_at": None,
        "demos": demos,
        "comparisons": comparisons,
    }
    manifest_path = os.path.join(DOCS, "manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print()
    print(f"Wrote {manifest_path} ({len(demos)} demo(s), {len(comparisons)} comparison(s))")
    print(f"docs/assets/audio : {dir_size_human(AUDIO_DIR)}")
    print(f"docs/assets/plots : {dir_size_human(PLOTS_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
