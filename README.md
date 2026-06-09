# SmartAutoDJ — Structure-Aware Automatic DJ Transitions

Given two songs **A** and **B**, generate a short, beat-matched transition that is
musically smarter than a plain crossfade. SmartAutoDJ analyzes tempo, beats, downbeats,
structure, and key; aligns the tracks at a musically meaningful boundary; reshapes the
overlap with equal-power fades and a bass-swap EQ; and can mix in an optional
AI-generated **bridge** (riser, drum fill, sweep, …). A genre classifier picks the
transition *style* per song pair.

CS 352 Music Perception final project · Northwestern University · Prof. Jason Smith ·
Spring 2026.

**Live demo + writeup:** https://10cirenehc.github.io/CS352-final-SmartAutoDJ/ — the site
explains the methods (cue-point selection, equal-power / bass-swap fades, the
swept-resonant swoosh, AI-bridge mixing, genre detection) in detail. This README is the
build/run reference; see `CLAUDE.md` for the full design notes.

## Tiers

The system is evaluated in three tiers, each a strict superset of the last:

| Tier | Name         | What it does |
|------|--------------|--------------|
| 1    | baseline     | naive linear crossfade (A's tail over B's head, no alignment) |
| 2    | beat-aligned | structure-aware cue points + tempo-match + downbeat alignment + equal-power fades + bass-swap EQ |
| 3    | AI-enhanced  | tier 2 + an additive bridge layer (procedural, or MusicGen-Style generated) |

Any tier also has a `--style cut` **quick-cut** variant: fade A out fast, build a swoosh
over the seam, and drop B in on the downbeat.

## Setup

```bash
brew install ffmpeg rubberband          # ffmpeg: audio I/O; rubberband: HQ time-stretch
conda create -n smartdj python=3.10
conda activate smartdj
pip install -e .                         # core: librosa, numpy, scipy, matplotlib, soundfile, pyrubberband
```

The pipeline runs on this core alone. The optional installs below upgrade individual
stages — each is independent, and the pipeline degrades gracefully if one is missing.

**Beat / downbeat tracking — `beat_this` (default, recommended).** The analysis backend
is `auto`, which **prefers the `beat_this` neural tracker** and falls back to librosa. It
is *not* pulled in by `pip install -e .`, so install it to get the better downbeats
(otherwise `auto` silently uses librosa):

```bash
pip install https://github.com/CPJKU/beat_this/archive/main.zip
```

**Genre-driven style — `.[genre]`.** A pretrained distilHuBERT / GTZAN classifier picks
the transition style per pair (CPU, on the torch stack `beat_this` already brings in):

```bash
pip install -e ".[genre]"
```

**AI bridge generation — `.[gen]` (+ Modal).** Tier-3 `--generate` runs MusicGen-Style on
a serverless CUDA GPU via Modal. The local side is only the lightweight client;
audiocraft / torch-2.1 live exclusively in the Modal image (`infra/modal_bridge.py`):

```bash
pip install -e ".[gen]"
modal setup                              # one-time browser auth
modal deploy infra/modal_bridge.py       # build image + cache weights once
```

Without `--generate`, tier 3 uses a pre-generated clip from `assets/generated/` if present,
else a procedural fallback synth — so the default run is fast and free.

**`allin1` (optional, local-only).** Labeled structure segments via all-in-one. Its
NATTEN / madmom pins are fragile and broken in Colab (see `CLAUDE.md` §5 for the verified
matrix). It is no longer on the auto path — request it explicitly with `--backend allin1`.

## Usage

```bash
# beat-aligned (tier 2), auto genre style:
python -m smartautodj.pipeline --song-a data/a.mp3 --song-b data/b.mp3 --tier 2 --out outputs/

# tier-3 quick cut (procedural bridge):
python -m smartautodj.pipeline --song-a data/a.mp3 --song-b data/b.mp3 --tier 3 --style cut

# tier-3 with an AI bridge generated on Modal:
python -m smartautodj.pipeline --song-a data/a.mp3 --song-b data/b.mp3 --tier 3 --generate
```

### Flags

**Core**

| flag | default | meaning |
|------|---------|---------|
| `--song-a`, `--song-b` | — | input tracks (required) |
| `--tier {1,2,3}` | `2` | 1 baseline · 2 beat-aligned · 3 AI bridge |
| `--out DIR` | `outputs` | output directory |
| `--no-plots` | off | skip plot rendering |
| `--seed N` | `0` | RNG seed for the procedural bridge |

**Style & genre**

| flag | default | meaning |
|------|---------|---------|
| `--style {auto,dance,urban,techno,dub,band,smooth,ambient,default,cut}` | `auto` | genre-driven preset, or `cut` for the quick-cut shape |
| `--genre LABEL` | auto-detect | force a GTZAN label for both tracks (e.g. `hiphop`, `disco`, `classical`) |
| `--effect {riser,buildup,hits,impact,sweep,siren,pad,riff}` | from genre | override the tier-3 effect (`riff` = a melodic run in the tracks' key/timbre) |

**Region & alignment**

| flag | default | meaning |
|------|---------|---------|
| `--bars N` | preset | overlap length in bars (tier 2/3); overrides the style preset |
| `--overlap-sec S` | match bars | tier-1 overlap length in seconds |
| `--tempo-tol F` | preset | max fractional tempo gap allowed before time-stretching B |
| `--cue {match,energy,novelty}` | preset | cue-point selection method |
| `--bpm-a`, `--bpm-b` | detected | override a track's tempo (when the beat tracker mis-detects) |
| `--cue-a`, `--cue-b` | auto | manually place A's exit / B's entry (s); snapped to the nearest downbeat |
| `--fade-sharpness F` | `2.5` | crossfade quickness (1 = gradual equal-power, higher = punchier swap) |

**Quick-cut** (`--style cut`)

| flag | default | meaning |
|------|---------|---------|
| `--cut-woosh {noise,bandpass,highpass,lowpass,none}` | `noise` | the swoosh into the drop (`noise` = swept-resonant whoosh layer; `bandpass` sweeps a resonant band-pass on B's entrance) |
| `--cut-sweep-bars F` | `2.0` | bars the swoosh runs over (0 = no swoosh) |
| `--cut-xfade-bars F` | `2.0` | bars over which B crossfades in over the swoosh (the "smoosh"; 0 = hard cut) |
| `--cut-fade-frac F` | `0.65` | fraction of the seam over which A fades to silence (lower = A fades sooner) |

**AI bridge** (tier 3)

| flag | default | meaning |
|------|---------|---------|
| `--generate` | off | generate the bridge on Modal (needs `.[gen]` + a deployed `infra/modal_bridge.py`) |
| `--bridge {riser,drum_fill,siren,pad,cymbal}` | genre element | override the procedural bridge kind (fallback synth) |
| `--bridge-clip PATH` | auto | use a specific pre-generated AI clip |
| `--bridge-eval-q N` | `2` | MusicGen-Style conditioning strength 1..6 (lower = cleaner isolated FX, higher = blends the songs in) |
| `--bridge-gain-db F` | `0` | bridge loudness vs the program (+ = more prominent, − = subtler) |
| `--bridge-duck-db F` | `6` | how much to duck the songs under the bridge (0 = no ducking) |
| `--no-riserize` | off | don't impose the rising filter sweep on the AI clip |
| `--no-bridge-limit`, `--no-spectral-carve` | off | disable the bridge soft-limit / spectral carve |

**Backend**

| flag | default | meaning |
|------|---------|---------|
| `--backend {auto,beat_this,librosa,allin1}` | `auto` | analysis backend (`auto` = beat_this → librosa) |

### Outputs

Each run writes into `--out`:

- `<stem>.wav` — the rendered transition
- `<stem>.json` — sidecar: per-track analysis (BPM, beats, downbeats, sections, key,
  genre), the transition plan (region, anchors, stretch ratio, fade/EQ params, bridge
  placement + prompt), and the objective metrics
- `<stem>_*.png` — the visualization suite (waveforms, fades, structure, key wheel,
  self-similarity, loudness continuity, beat-alignment, bass-swap, spectrogram,
  mel + bridge, genre probabilities)

## Objective metrics (written to the sidecar)

- **beat-alignment error** — mean |Δt| between matched A/B beats in the overlap
- **tempo-match quality** — BPM gap before vs after time-stretching B
- **loudness continuity** — largest sudden level (dB) jump across the overlap

The clearest tier-over-tier win is loudness continuity (e.g. Skrillex → Core: a 12.6 dB
jump at tier 1 drops to ~1.0 dB at tier 3). The **Evaluation** section on the site shows
the per-pair tier-1/2/3 comparison. Beat-alignment error is meaningful for blends but
near-meaningless for quick cuts (the songs barely overlap).

## Website / demo assets

The GitHub Pages site lives in `docs/` and is data-driven from `docs/manifest.json`.
To regenerate it after new renders:

```bash
python scripts/render_cut_demos.py      # render the showcase quick-cut demos + tier ladders
python scripts/build_site_assets.py     # trim/encode MP3s, copy plots, write manifest.json
```

Publish via the repo **Settings → Pages → Deploy from a branch → `main` / `/docs`**.

## Layout

```
src/smartautodj/   io · analysis · structure · align · transition · generative · genre · remote · mix · postprocess · viz · evaluate · pipeline
infra/             modal_bridge.py — serverless CUDA MusicGen-Style bridge generation
scripts/           render_cut_demos.py · build_site_assets.py — render demos + package the web assets
docs/              GitHub Pages site (index.html, css, js, manifest.json, assets)
tests/             synthetic click-track unit + end-to-end tests
data/              input clips (gitignored)
outputs/           WAVs + plots + JSON (gitignored)
assets/generated/  small AI bridge clips (committed)
```

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

The tests synthesize click tracks at known tempos, so `analyze → align → mix` is tested
deterministically without shipping any copyrighted audio.

## References & credits

The methods are explained on the
[project site](https://10cirenehc.github.io/CS352-final-SmartAutoDJ/); this is the source
material behind the models, tools, and techniques the pipeline uses.

**Models we run**

- **beat_this** — neural beat/downbeat tracker (default backend). Foscarin, Schlüter,
  Widmer, *Beat This! Accurate Beat Tracking Without DBN Postprocessing*, ISMIR 2024 —
  [paper](https://arxiv.org/abs/2407.21658) · [code](https://github.com/CPJKU/beat_this)
- **all-in-one** (`allin1`) — beats/downbeats + functional structure (optional backend).
  Kim & Nam, *All-In-One Metrical And Functional Structure Analysis With Neighborhood
  Attentions on Demixed Audio*, WASPAA 2023 — [paper](https://arxiv.org/abs/2307.16425) ·
  [code](https://github.com/mir-aidj/all-in-one); builds on **madmom** (Böck et al., ACM
  MM 2016 — [code](https://github.com/CPJKU/madmom))
- **distilHuBERT** — genre-classifier backbone, checkpoint
  [`sanchit-gandhi/distilhubert-finetuned-gtzan`](https://huggingface.co/sanchit-gandhi/distilhubert-finetuned-gtzan).
  Chang, Yang, Lee, *DistilHuBERT*, ICASSP 2022
  ([paper](https://arxiv.org/abs/2110.01900)); distilled from **HuBERT** (Hsu et al., 2021
  — [paper](https://arxiv.org/abs/2106.07447))
- **MusicGen / MusicGen-Style** — AI bridge generation. Copet et al., *Simple and
  Controllable Music Generation*, NeurIPS 2023 ([paper](https://arxiv.org/abs/2306.05284));
  audio conditioning via Rouard et al., *Audio Conditioning for Music Generation via
  Discrete Bottleneck Features*, ISMIR 2024 ([paper](https://arxiv.org/abs/2407.12563)) —
  [audiocraft](https://github.com/facebookresearch/audiocraft) ·
  [model](https://huggingface.co/facebook/musicgen-style). Alternative text-to-audio:
  **Stable Audio Open** (Evans et al. 2024 — [paper](https://arxiv.org/abs/2407.14358))

**Core libraries & tools**

- [librosa](https://librosa.org/) (McFee et al., SciPy 2015) — audio analysis & features
- [NumPy](https://numpy.org/) · [SciPy](https://scipy.org/) · [Matplotlib](https://matplotlib.org/)
  — numerics, DSP filters, plotting
- [PySoundFile](https://python-soundfile.readthedocs.io/) / libsndfile ·
  [FFmpeg](https://ffmpeg.org/) — audio I/O & decoding
- [pyrubberband](https://github.com/bmcfee/pyrubberband) +
  [Rubber Band Library](https://breakfastquay.com/rubberband/) — transient-aware time-stretch
- [Hugging Face Transformers](https://github.com/huggingface/transformers) — runs the genre classifier
- [Modal](https://modal.com/) — serverless CUDA for bridge generation
- [WaveSurfer.js](https://wavesurfer.xyz/) — interactive waveform player ·
  [GitHub Pages](https://pages.github.com/) — site hosting

**Datasets**

- **GTZAN** — genre-classifier training set. Tzanetakis & Cook, *Musical Genre
  Classification of Audio Signals*, IEEE TSAP 2002 —
  [doi](https://doi.org/10.1109/TSA.2002.800560)
- Audio clips: [Free Music Archive](https://freemusicarchive.org/) and the
  [YouTube Audio Library](https://www.youtube.com/audiolibrary) (royalty-free) + personal picks

**Algorithms & methods**

- **Krumhansl–Schmuckler** key-finding (key estimation) — Krumhansl, *Cognitive
  Foundations of Musical Pitch*, Oxford University Press, 1990
- **Camelot wheel** harmonic mixing (key compatibility) —
  [Mixed In Key](https://mixedinkey.com/camelot-wheel/)
- **TPT / Zavalishin state-variable filter** (the swept swoosh) — Zavalishin, *The Art of
  VA Filter Design* —
  [pdf](https://archive.org/details/the-art-of-va-filter-design-rev.-2.1.2)
- **Self-similarity matrices** (structure visualization) — Foote, *Visualizing Music and
  Audio using Self-Similarity*, ACM Multimedia 1999

**Related work**

- Chen, Hsu, Liao, Martínez Ramírez, Mitsufuji, Yang, *Automatic DJ Transitions with
  Differentiable Audio Effects and GANs*, 2021 — closest prior work
  ([paper](https://arxiv.org/abs/2110.06525))
- Zehren, Alunno, Bientinesi, *Automatic Detection of Cue Points for the Emulation of DJ
  Mixing*, Computer Music Journal 2022 ([doi](https://doi.org/10.1162/comj_a_00652))
- Heydari, Cwitkowitz, Duan, *BeatNet: CRNN and Particle Filtering for Online
  Joint Beat, Downbeat and Meter Tracking*, ISMIR 2021
  ([paper](https://arxiv.org/abs/2108.03576))
