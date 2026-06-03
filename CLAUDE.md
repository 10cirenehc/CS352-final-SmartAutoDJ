# CLAUDE.md — SmartAutoDJ

Guidance for Claude Code (and humans) working in this repo. This is a **final project for a Music Perception course** (CS_352-style). It is a research prototype + visualizations + listening survey — **not** a shipped product. Optimize for clarity, reproducibility, and demonstrating course concepts, not for production polish.

---

## 1. Project overview

**SmartAutoDJ: Structure-Aware Automatic DJ Transitions.** Given two songs (A and B), generate a short, beat-matched transition that is musically smarter than a plain crossfade. The system analyzes tempo, beats, downbeats, and song structure, aligns the two tracks at a musically meaningful boundary, and synthesizes a transition using volume fades and EQ-style filtering. An optional **AI-generated bridge layer** (riser, drum fill, synth swell, ambient pad) can be mixed into the transition region.

The project is built and evaluated as **three tiers**, each a strict superset of the last:

1. **Baseline** — naive linear crossfade (overlap end of A with start of B, no alignment).
2. **Beat-aligned** — tempo-match + downbeat-aligned transition with fade/EQ curves.
3. **AI-enhanced** — beat-aligned transition plus a generated auxiliary layer mixed in.

Everything is benchmarked tier-1 → tier-3 so we can show *what each piece buys us*.

---

## 2. Team

| Name | NetID | Status |
|------|-------|--------|
| Eric Shang Nan Chen | ESC4970 | For credit |
| Yindi Zhao | YZV5448 | For credit |
| Kefan Yu | — | Not for credit |

---

## 3. Pipeline architecture

The core is a single pipeline; each stage should map to one module (see §4). Data flows:

```
load A, B
  → analyze    (tempo, beats, downbeats, structure/sections for each track)
  → select     (pick transition region: a downbeat / phrase boundary in A and B)
  → stretch    (optional: time-stretch B toward A's tempo if BPMs are close)
  → align      (line up A's outgoing downbeats with B's incoming downbeats)
  → transition (build per-track volume-fade + EQ curves over the region)
  → generative (optional: trim + normalize + beat-align an AI clip into the region)
  → mix        (sum A-tail, B-head, fades/EQ, and AI layer over the overlap)
  → export      (WAV output + plots + JSON/CSV metadata: beat times, region, layer placement)
```

Design rules:
- Each tier is reachable by toggling stages off (baseline = `select`/`align`/`generative` disabled, just `mix` a linear crossfade). Keep stages composable.
- All analysis results (BPM, beat/downbeat times, chosen region) are written to a **JSON/CSV sidecar** next to every output WAV so results are inspectable and the evaluation code can reuse them.
- The generative layer **never replaces** the algorithmic transition — it is an additive bridge.

---

## 4. Proposed repo structure

Nothing but `README.md` and `.gitignore` exists yet. Target layout (create modules as you build them):

```
src/smartautodj/
  io.py          # load/save audio (librosa/soundfile), resampling
  analysis.py    # tempo, beats, downbeats, structure (librosa → allin1/beat_this)
  align.py       # tempo match, time-stretch, downbeat alignment
  transition.py  # fade curves, EQ-style filtering, naive-crossfade baseline
  generative.py  # import + trim + normalize + beat-align AI clips (from assets/)
  mix.py         # sum/overlap tracks + layers over the transition region
  viz.py         # waveforms, spectrograms, beat markers, fade-curve plots
  evaluate.py    # beat-alignment error, tempo match, loudness continuity
  pipeline.py    # orchestrates all stages; tier selection
data/            # input song clips — GITIGNORED (large audio, possible copyright)
assets/generated/# AI clips downloaded from Colab — small, may be committed
outputs/         # WAVs, plots, JSON/CSV — GITIGNORED
notebooks/       # exploration + the Colab generative notebook
web/             # GitHub Pages demo site
```

Add `data/`, `outputs/` to `.gitignore`. Keep `assets/generated/` small if committed.

---

## 5. Environment & setup

**Primary: conda/mamba.** The audio-MIR stack has fragile, conflicting pins — use conda and **isolate the generative model**. Do not try to install everything into one env.

### Env A — analysis & mixing (local, macOS)
Python ~3.10. Core (course toolkit): `librosa`, `numpy`, `scipy`, `matplotlib`, `soundfile`.

```bash
brew install ffmpeg                      # required by audio loaders / allin1
conda create -n smartdj python=3.10
conda activate smartdj
pip install librosa numpy scipy matplotlib soundfile
```

**Better beats/downbeats/structure (optional upgrades over librosa):**
- `madmom` is needed by both tools below and is **broken on PyPI** (only Py<3.10, numpy<1.20). Install from the CPJKU git fork:
  ```bash
  pip install git+https://github.com/CPJKU/madmom.git
  ```
- **all-in-one** (`allin1`) — BPM, beats, downbeats, segment boundaries + labels (intro/verse/chorus/bridge/outro). On macOS NATTEN auto-installs. Needs `ffmpeg` + `madmom` (above):
  ```bash
  pip install allin1
  ```
- **beat_this** (CPJKU) — accurate transformer beat/downbeat tracker; **fallback if allin1 is slow/flaky**. Runs on CPU.
  ```bash
  pip install https://github.com/CPJKU/beat_this/archive/main.zip
  ```
  ```python
  from beat_this.inference import File2Beats
  beats, downbeats = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)("song.wav")
  ```
- `essentia` (optional) — loudness, key, energy, spectral descriptors for pairing/style selection.

### Env B — generative (Google Colab / cloud GPU, NOT local)
MusicGen/AudioCraft effectively need a **CUDA GPU (16 GB rec.)**; on macOS only `musicgen-small` runs (slowly) on CPU. **Decision: generate in Colab, download clips into `assets/generated/`**, and have `generative.py` treat them as static assets the local pipeline trims/aligns/mixes. Alt text-to-audio: **Stable Audio Open**.

```python
# Colab only
pip install -U audiocraft   # needs torch==2.1.0 installed first; ffmpeg present
```

> ⚠️ Never mix Env A and Env B. `allin1`, `madmom`, and `audiocraft` pin incompatible torch/Python/numpy versions. Keep generative work in Colab.

---

## 6. Course-topic mapping

Each module demonstrates a Music Perception course concept — keep these connections explicit in code comments and the writeup:

| Course topic | Where it shows up |
|---|---|
| Loudness & Amplitude | loudness-continuity metric; volume-fade curves |
| Fourier Transforms & Spectrograms | spectrogram visualizations of A/B and the transition |
| Convolution & Filtering | EQ-style transition filters (low/high-pass sweeps) |
| Source Separation (REPET) | stem-aware transitions — fade vocals/drums separately (stretch goal) |
| MFCCs & Chromagrams | key/timbre matching to pick or order song pairs |
| Self-Similarity | structure / phrase-boundary detection for transition points (SSM) |
| Pitch Tracking | key estimation for harmonic compatibility |
| Deep Learning / Autoencoders / Embeddings | neural beat trackers (beat_this) and MusicGen / Stable Audio |

---

## 7. Evaluation & baselines

**Baseline:** naive linear crossfade (tier 1). Compare against tier 2 and tier 3.

Objective metrics (computed over the transition region, written to JSON/CSV):
- **Beat-alignment error** — mean absolute time difference between matched A/B beats in the overlap.
- **Tempo-match quality** — BPM agreement before vs. after time-stretching.
- **Loudness continuity** — check for sudden level jumps across the transition (ties to Loudness topic).

Subjective: **listening survey** with classmates rating *smoothness, rhythmic alignment, creativity, overall preference* across the three tiers — to test whether the AI layer actually helps or just distracts.

---

## 8. Data

- 10–20 **steady-tempo** clips: pop, EDM, hip-hop, dance (clear beats, popular).
- Sources: **Free Music Archive**, **YouTube Audio Library** (royalty-free), plus personal picks.
- Mostly unlabeled; for a small subset, manually mark transition regions / verify downbeats.
- Inputs are audio files; outputs are WAVs + plots + JSON/CSV metadata.
- **Keep audio out of git** (`data/`, `outputs/` gitignored) — size and copyright.

---

## 9. Conventions

- Python + conda; pin tricky installs (madmom from git, beat_this zip) as documented above.
- **librosa is the default path**; only reach for `allin1`/`beat_this` when you need better downbeats or structure than librosa gives.
- Every output WAV gets a sidecar JSON/CSV (beat times, chosen region, layer placement).
- Set deterministic seeds where randomness exists; keep runs reproducible.
- Generative clips are **static assets** from Colab — the local pipeline only trims/normalizes/aligns/mixes them.
- Prefer clear, commented signal-processing code over cleverness — this is a teaching/demo artifact.

---

## 10. References

Papers:
1. Chen, Hsu, Liao, Martínez Ramírez, Mitsufuji, Yang (2021). *Automatic DJ Transitions with Differentiable Audio Effects and GANs.* arXiv:2110.06525 — https://arxiv.org/abs/2110.06525  *(closest prior work)*
2. Copet et al. (2023). *Simple and Controllable Music Generation (MusicGen).* arXiv:2306.05284 — https://doi.org/10.48550/ARXIV.2306.05284
3. Evans et al. (2024). *Stable Audio Open.* arXiv:2407.14358 — https://doi.org/10.48550/arXiv.2407.14358
4. Heydari, Cwitkowitz, Duan (2021). *BeatNet: CRNN + Particle Filtering for Online Beat/Downbeat/Meter Tracking.* arXiv:2108.03576 — https://doi.org/10.48550/ARXIV.2108.03576
5. Zehren, Alunno, Bientinesi (2022). *Automatic Detection of Cue Points for the Emulation of DJ Mixing.* Computer Music Journal 46(3), 67–82 — https://doi.org/10.1162/comj_a_00652

Commercial / related: DJ.Studio Automix/Harmonize (https://dj.studio/automix) · rekordbox (https://rekordbox.com/en/) · Spotify DJ · Apple Music AutoMix.

Libraries: librosa (https://librosa.org/) · all-in-one (https://github.com/mir-aidj/all-in-one) · beat_this (https://github.com/CPJKU/beat_this) · audiocraft (https://github.com/facebookresearch/audiocraft) · essentia (https://essentia.upf.edu/) · matplotlib (https://matplotlib.org/) · numpy (https://numpy.org/) · GitHub Pages (https://pages.github.com/).

---

## 11. Status & roadmap

**Current:** repo scaffolding only (README + .gitignore + this file). No pipeline code yet.

- **Milestone 1 (first meeting):** install/smoke-test each tool; build the full pipeline skeleton; produce ≥1 working transition with visualizations.
- **Milestone 2 (second meeting):** complete prototype + objective evaluation metrics.
- **Final presentation:** demo, survey results, tested audio examples, project website.

**De-risking rule:** test each module on short, consistent clips first. If any tool is too slow/unreliable to install or run, **simplify** — drop to librosa-only beats, pre-generate a small AI-clip library, or cut a feature. Find out early.
