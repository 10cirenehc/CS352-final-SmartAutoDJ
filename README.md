# SmartAutoDJ — Structure-Aware Automatic DJ Transitions

Given two songs **A** and **B**, generate a short, beat-matched transition that is
musically smarter than a plain crossfade. The system analyzes tempo, beats, downbeats
and structure, aligns the tracks at a musically meaningful boundary, and synthesizes the
transition with volume fades and EQ-style filtering. An optional AI "bridge" layer
(riser / drum fill) is mixed into the transition region.

CS_352 Music Perception final project. Research prototype — optimized for clarity,
reproducibility, and demonstrating course concepts. See `CLAUDE.md` for the full design.

## Tiers

The system is evaluated in three tiers, each a strict superset of the last:

| Tier | Name         | What it does                                                |
|------|--------------|-------------------------------------------------------------|
| 1    | baseline     | naive linear crossfade (A's tail over B's head, no alignment) |
| 2    | beat-aligned | tempo-match + downbeat-aligned region + equal-power fades + bass-swap EQ |
| 3    | AI-enhanced  | tier 2 + an additive procedural bridge layer (riser / drum fill) |

## Setup

```bash
brew install ffmpeg
conda create -n smartdj python=3.10
conda activate smartdj
pip install -e .            # installs librosa/numpy/scipy/matplotlib/soundfile
```

The analysis stage prefers **allin1** (BPM, beats, downbeats, labeled structure) and
silently falls back to **librosa** if it isn't installed or fails — so the pipeline always
runs. `--backend librosa` forces the fallback; `--backend allin1` requires the install
below.

### Optional: allin1 (higher-quality downbeats + labeled sections)

allin1's dependency pins are fragile (CLAUDE.md §5). The following matrix is **verified
working** on macOS arm64 / Python 3.10 with a modern torch:

```bash
pip install "git+https://github.com/CPJKU/madmom.git"          # madmom from source
pip install allin1                                              # pulls torch + demucs + NATTEN
pip install "natten==0.15.1" --no-build-isolation --force-reinstall --no-deps  # legacy NATTEN API allin1 needs, built against installed torch
pip install torchcodec                                          # demucs stem saving uses it
```

Two compatibility shims are handled automatically in `analysis.py`:
- a `torch.cuda._device_t` alias (NATTEN 0.15 imports a symbol newer torch removed), and
- `allin1.analyze(..., multiprocess=False)` — on macOS the spawned workers re-import NATTEN
  without the shim and deadlock, so analysis runs in-process.

First run downloads model weights and demuxes each track on CPU (~30 s/track).

## Usage

```bash
python -m smartautodj.pipeline \
    --song-a data/a.wav --song-b data/b.wav \
    --tier 2 --out outputs/
```

Useful flags: `--tier {1,2,3}`, `--bars N` (overlap length in bars, tier 2/3),
`--overlap-sec S` (tier-1 overlap), `--bridge {riser,drum_fill}` (tier 3),
`--backend {auto,allin1,librosa}`, `--no-plots`, `--seed N`.

Each run writes, into `--out`:
- `<stem>.wav` — the rendered transition
- `<stem>.json` — sidecar: per-track analysis (BPM, beats, downbeats, sections), the
  transition plan (region, anchors, stretch ratio, fade/EQ params, bridge placement),
  and objective metrics
- `<stem>_{waveforms,fades,region,spectrogram}.png` — visualizations

## Objective metrics (written to the sidecar)

- **beat-alignment error** — mean |Δt| between matched A/B beats in the overlap
- **tempo-match quality** — BPM gap before vs after time-stretching B
- **loudness continuity** — largest sudden level (dB) jump across the overlap

Example (synthetic 120 vs 125 BPM clips, same 8-bar overlap across tiers), showing what
each tier buys:

| Tier | beat-align err (s) | BPM gap (before → after) | max dB jump |
|------|--------------------|--------------------------|-------------|
| 1    | 0.110              | 5.86 → 5.86              | 3.66        |
| 2    | 0.005              | 5.86 → 0.81              | 3.61        |
| 3    | 0.005              | 5.86 → 0.81 (+ bridge)   | 3.59        |

Tiers are compared over an equal overlap region by default. (On these sparse click tracks
the loudness gap is small; on continuous music a naive crossfade's mid-point power dip and
a hard cut show up as larger dB jumps.)

## Layout

```
src/smartautodj/   io · analysis · align · transition · generative · mix · viz · evaluate · pipeline
tests/             synthetic click-track unit + end-to-end tests (+ allin1 smoke test)
data/              input clips (gitignored)
outputs/           WAVs + plots + JSON (gitignored)
assets/generated/  small AI clips from Colab (committed)
```

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

The tests synthesize click tracks at known tempos, so `analyze → align → mix` is tested
deterministically without shipping any copyrighted audio.
