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
        "id": "song_A__song_B",
        "title_a": "Song A",
        "title_b": "Song B",
    },
    # Synthetic fixture -- short and not musically convincing; useful for testing only.
    # {"id": "demo_a__demo_b", "title_a": "Demo A (synthetic)", "title_b": "Demo B (synthetic)"},
]

# Which plot keys to copy per tier (tier 1 never has a structure plot).
PLOT_POLICY = {
    1: ["spectrogram", "fades"],
    2: ["spectrogram", "structure", "fades"],
    3: ["spectrogram", "structure", "fades"],
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


def copy_png(src: str, dst: str) -> None:
    shutil.copy2(src, dst)
    if shutil.which("pngquant"):
        # Lossy shrink in place; --force overwrites, --skip-if-larger is a safety net.
        subprocess.run(
            ["pngquant", "--quality=65-85", "--force", "--skip-if-larger",
             "--output", dst, dst],
            check=False,
        )


def slim_metrics(sidecar: dict) -> dict:
    m = sidecar["metrics"]
    return {
        "beat_align_err_sec": round(m["beat_alignment"]["mean_abs_error_sec"], 4),
        "bpm_gap_before": round(m["tempo_match"]["bpm_gap_before"], 2),
        "bpm_gap_after": round(m["tempo_match"]["bpm_gap_after"], 2),
        "max_db_jump": round(m["loudness_continuity"]["max_db_jump"], 2),
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
    return {
        "id": stem_id,
        "title_a": cfg.get("title_a") or (parts[0] if parts else stem_id),
        "title_b": cfg.get("title_b") or (parts[1] if len(parts) > 1 else ""),
        "bpm_a": round(a.get("bpm"), 1) if a.get("bpm") else None,
        "bpm_b": round(b.get("bpm"), 1) if b.get("bpm") else None,
        "key_a": key_name(a),
        "key_b": key_name(b),
        "tiers": tiers,
    }


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

    pairs = [p for cfg in CURATED if (p := build_pair(cfg)) is not None]
    if not pairs:
        print("ERROR: no demo pairs built. Run the pipeline into outputs/ first, "
              "and check the `id`s in CURATED.", file=sys.stderr)
        return 1

    manifest = {
        # Date is intentionally left for the committer to note; avoids nondeterminism.
        "generated_at": None,
        "survey_url": SURVEY_URL,
        "pairs": pairs,
    }
    manifest_path = os.path.join(DOCS, "manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print()
    print(f"Wrote {manifest_path} ({len(pairs)} pair(s))")
    print(f"docs/assets/audio : {dir_size_human(AUDIO_DIR)}")
    print(f"docs/assets/plots : {dir_size_human(PLOTS_DIR)}")
    if not SURVEY_URL:
        print("NOTE: SURVEY_URL is empty -- the survey section will show a placeholder. "
              "Create a Google Form and paste its embed URL into SURVEY_URL, then re-run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
