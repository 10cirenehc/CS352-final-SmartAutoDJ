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
  analysis.py    # tempo, beats, downbeats (beat_this → librosa; allin1 opt-in) + structure feats
  structure.py   # RMS/MFCC/chroma/onset novelty, key est, structure-aware cue-point selection
  align.py       # tempo match, RubberBand time-stretch, downbeat alignment, cue selection
  transition.py  # fade curves, equal-power bass-swap EQ, naive-crossfade baseline
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
brew install ffmpeg rubberband           # ffmpeg: audio loaders/allin1; rubberband: HQ time-stretch
conda create -n smartdj python=3.10
conda activate smartdj
pip install librosa numpy scipy matplotlib soundfile pyrubberband
```

> **Time-stretch quality:** `align.apply_stretch` prefers **RubberBand** (via `pyrubberband`,
> needs the `rubberband` brew binary) — transient-aware, avoids the phase-vocoder "phasiness"
> that made early transitions sound warped. Falls back to `librosa` if the binary is absent.

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

**Genre-driven transition style (optional upgrade):**
- **genre classifier** (`transformers`) — picks the **transition style** per song pair (§11).
  Install the extra: `pip install -e ".[genre]"`. Runs the small
  `sanchit-gandhi/distilhubert-finetuned-gtzan` (distilHuBERT) on CPU on the **existing torch
  stack** (torch arrives via beat_this — no new heavy runtime; this is Env A, *not* audiocraft).
  **GTZAN is only 10 genres** (blues, classical, country, disco, hiphop, jazz, metal, pop,
  reggae, rock) — *no* house/techno/ambient — so `disco`→dance and `classical`→ambient are
  proxies. Falls back to the default beat-aligned style when transformers/the model is absent.
  ```bash
  python -m smartautodj.pipeline --song-a A.mp3 --song-b B.mp3 --tier 3            # --style auto
  python -m smartautodj.pipeline --song-a A.mp3 --song-b B.mp3 --style dance       # force a preset
  python -m smartautodj.pipeline --song-a A.mp3 --song-b B.mp3 --genre hiphop      # force the label
  ```

### Env B — generative (Modal / cloud GPU, NOT local)
MusicGen/AudioCraft effectively need a **CUDA GPU**; on macOS `audiocraft` is painful
(pins torch 2.1, no real MPS, xformers). **Decision: generate on [Modal](https://modal.com)**
— a serverless cloud GPU where audiocraft + torch 2.1 + ffmpeg are baked into a container
**image once** (no per-session reinstall, unlike Colab) and the musicgen-style weights are
cached on a Modal Volume. The app is `infra/modal_bridge.py`; the local side is just the
lightweight `modal` client (extra `gen`). `generative.py` still treats the resulting clip as a
static asset it trims/normalizes/mixes. Alt text-to-audio: **Stable Audio Open** (diffusers).

```bash
pip install -e ".[gen]"               # local: lightweight modal client only
modal setup                           # one-time browser auth (free tier ~$30/mo credit)
modal deploy infra/modal_bridge.py    # build image + cache weights once

# standalone one-command generate (upload boundary-ref -> CUDA -> download clip):
modal run infra/modal_bridge.py --ref outputs/A__B__tier3_boundary_ref.wav \
    --prompt "<plan.bridge.prompt from the sidecar>" --out assets/generated/A__B_bridge.wav --duration 14

# or inline — one command does ref + generate + mix:
python -m smartautodj.pipeline --song-a A.mp3 --song-b B.mp3 --tier 3 --generate
```

For unattended/agent runs set `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` instead of `modal setup`.
The old Colab notebook (`notebooks/generative_musicgen.ipynb`) is kept only as a manual fallback.

> ⚠️ Never install Env B's stack locally. `allin1`, `madmom`, and `audiocraft` pin incompatible
> torch/Python/numpy versions — audiocraft lives **only** in the Modal image, never in `smartdj`.

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
| Deep Learning / Autoencoders / Embeddings | neural beat trackers (beat_this), the distilHuBERT **genre classifier** that picks the transition style, and MusicGen / Stable Audio |

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

**Current:** full 3-tier pipeline working, with a Milestone-2 quality/structure pass done.

- **Milestone 1 (first meeting):** ✅ pipeline skeleton + ≥1 working transition with visualizations.
- **Milestone 2 (second meeting):** prototype + objective metrics. Done so far:
  - **Audio quality:** fixed the bass-swap EQ (was a midpoint bass dropout → equal-power
    handoff in `transition.bass_swap_curves`); time-stretch now prefers **RubberBand**
    (`align.apply_stretch`) over the phase vocoder; headroom guard in `mix` (real songs hit
    1.24 full-scale → would have clipped). Robust **median-of-inlier** tempo in
    `align._beat_period` (a polyfit slope inverted the stretch direction when beat_this
    missed beats on a real song — verified fixed: rendered-overlap beat dev ~5 ms).
  - **Structure-aware cue points:** new `structure.py` — RMS energy, MFCC/chroma/onset
    fused **novelty**, Krumhansl **key** + harmonic compatibility. `align.select_transition_region`
    now drops into a high-energy phrase (not B's intro): `cue_method="energy"` (Phase 1) or
    `"novelty"` (Phase 2). Energy/novelty/key go to the sidecar + a `*_structure.png` plot.
  - **Backends:** `backend="auto"` now prefers **beat_this** (Colab-friendly) → librosa;
    **allin1 is optional/local-only** (fragile, broken in Colab), no longer on the auto path.
  - **Generative:** `generative.resolve_bridge` loads a real **MusicGen-Style** clip from
    `assets/generated/<pair>_bridge.wav` (else procedural fallback); tier-3 exports a
    boundary-reference WAV + auto-built prompt.
  - **Generation moved Colab → Modal:** the bridge is now generated on **Modal** (serverless
    CUDA) via `infra/modal_bridge.py` — audiocraft/torch-2.1 baked into an image once, weights
    on a Volume, no per-session reinstall. `src/smartautodj/remote.py` is the thin local client;
    `pipeline.run` gains `--generate` so one command (`--tier 3 --generate`) builds the boundary
    reference, generates on the GPU, and mixes — or run `modal run infra/modal_bridge.py …`
    standalone. `--generate` is opt-in; default runs use the procedural fallback (fast/free).
    Local stays clean (extra `gen = [modal]` is just the client). See §5 (Env B).
  - **Genre-driven transition style:** new `genre.py` — a pretrained distilHuBERT/GTZAN
    classifier (`classify_genre`) picks *how* the transition sounds, not just where. Each
    track's genre maps to a **style family** (`dance`/`urban`/`dub`/`band`/`smooth`/`ambient`),
    and `transition_style` combines A+B into a `StylePreset` that overrides existing knobs —
    overlap `bars`, the time-stretch `tempo_tol` gate, the bass-swap EQ toggle, the cue method.
    `pipeline.run` gains `--style {auto,dance,urban,techno,dub,band,smooth,ambient,default}`
    (techno/dub are manual-only; reggae auto-maps to `dub`) and `--genre LABEL`; explicit
    `--bars/--tempo-tol/--cue` still win. Detected genres + chosen style go to the sidecar +
    structure-plot title. Falls back to `default` when transformers is absent. GTZAN's 10-genre
    taxonomy (no house/techno/ambient) is the known limitation.
  - **Genre-authentic bridge elements:** the style preset also selects a **`BridgeSpec`**
    (`genre.BRIDGE_SPECS`) — the genre's characteristic transition element, grounded in DJ
    practice: dance→`riser`, hip-hop→`stab_fill` (air-horn + vocal stab + boom-bap fill),
    techno→`noise_sweep`, reggae→`siren` (dub siren), ambient→`pad_wash`, pop→`sweep_wash`,
    rock→`cymbal`. Each spec carries a MusicGen prompt template (filled with BPM/key/duration)
    and a procedural fallback synth (`generative.make_placeholder_bridge` now does
    riser/drum_fill/siren/pad/cymbal). **Coverage policy (genre-appropriate):**
    dance/urban/techno/dub/ambient add their element by default at tier 3; **rock & pop add a
    subtle accent only on `--generate`** (a synth layer on those reads as a gimmick — real DJs
    use track-driven moves there). The chosen `element` lands in the sidecar (`plan.bridge`).
    `--bridge {riser,drum_fill,siren,pad,cymbal}` overrides the procedural kind.
  - **Quick-cut transition shape (`--style cut`):** a second render *shape* alongside the
    long beat-matched blend, for transitions that don't muddily overlap. The blend sums A+B
    over the whole overlap; the **cut** is *sequential* — `align.select_transition_region`
    (`shape="cut"`) anchors on B's drop downbeat and builds backward: A's tail fades out fast
    (`transition.cut_fade_out`), a continuous **riser** leads over the ~7-bar seam
    (`cut_bridge_envelope`, summed not ducked), then a swoosh leads the last ~2 bars while
    **B crossfades in** on the downbeat (see next bullet). `TransitionPlan.shape` + a manual-only
    `STYLE_PRESETS["cut"]` carry it (`mix._mix_cut` renders it); the seam is soft-limited
    locally so the effect peak doesn't turn the whole song down. Tier-2 cut = fade + woosh
    (no riser); tier-3 cut = + the riser. Knobs to tune by ear: `cut_fade_out` `fade_frac`,
    `--cut-sweep-bars`, `--cut-xfade-bars`, `--cut-woosh`, `--bridge-gain-db`. Note:
    beat-alignment-error is near-meaningless for the cut (A/B barely overlap) — the meaningful
    checks are "drop lands on a downbeat" (true by construction) + loudness continuity.
  - **Riserize the generated bridge (`generative.riserize`):** MusicGen doesn't reliably make
    a *riser* — the text "rising" isn't honored and MusicGen-Style's audio conditioning pulls
    the clip toward the songs' static timbre. Since a riser is a deterministic DSP effect, we
    force it in post: a model-agnostic **EQ filter-open** on the *real* clip (dull→full via
    `low*(1-t)+clip*t`, reusing `_filt`), peaking on the **last sample** so it resolves on the
    drop. It sounds natural because it's the actual audio brightening, not a synthetic tone (an
    optional synthetic sine uplifter exists but is **off by default** — it read as artificial).
    Applied to the AI clip in `resolve_bridge` **after** length-fitting to the seam (riserizing
    before the `clip[:n]` trim would chop off the climax). Riser generation uses a looser
    `eval_q=1` so MusicGen is freer to sweep. `--no-riserize` disables. Only rising effects are
    riserized (riser/sweep/buildup), never hits/siren/pad. Sidecar records `plan.bridge.riserized`.
  - **Smooth the cut's END — swoosh + smoosh (`mix._mix_cut`):** the drop into B used to be a
    hard concatenation (riser stops dead, B starts) — abrupt. Fixes: (1) a real **swoosh** —
    a swept-resonant-noise whoosh (`transition.make_whoosh` over `transition.swept_bandpass_svf`,
    a **TPT/Zavalishin state-variable filter** with a high-resonance band-pass whose center
    sweeps exp ~200 Hz→12 kHz; TPT stays stable past sr/6 where the naive SVF blows up — that's
    why it can actually sweep *high*) layered over the last `--cut-sweep-bars` of the seam,
    climbing into the drop; a fixed 2-band gain crossfade (the old `highpass`/`lowpass` modes,
    still available) has no moving resonance and can't swoosh. (2) **B crossfades in** over
    `--cut-xfade-bars` (the 2-bar "smoosh"), entering on a downbeat, overlapping the swoosh.
    `--cut-woosh {noise,bandpass,highpass,lowpass,none}`: `noise` (default), `bandpass` (swept
    resonant band-pass on B's own entrance), or the gentle 2-band modes. Defaults: `cut` seam
    **7 bars**, swoosh **2 bars**, smoosh **2 bars**, A fades out over **0.65** of the seam
    (`--cut-fade-frac`). (Reverted an earlier genre-aware effect
    mapping — GTZAN can't tell EDM from hip-hop — so `cut` is always a riser; `--effect`
    overrides.)
  - **Visualization suite + interactive site:** `viz.py` gained DJ/MIR-grade plots beyond the
    original waveform/spectrogram/fade/structure set — a **Camelot key wheel** (`plot_key_wheel`,
    A→B arc colored by `harmonic_compatibility`), **beat-synchronous self-similarity matrices**
    (`plot_ssm_pair`, recomputed from chroma — deliberately not stored in the sidecar), an
    **arrangement ribbon**, a **loudness-continuity curve** (shares the new
    `evaluate.short_term_loudness_db` helper with the metric so they can't drift), a
    **beat-alignment scatter**, a **mel-spectrogram + bridge region**, a **bass-swap low-freq
    handoff**, and a **genre-probability** bar (distilHuBERT). A cross-tier **comparison bar chart**
    (`plot_tier_comparison`) is built by `scripts/build_site_assets.py` (the only place that sees
    all 3 tiers) — the "what each tier buys" money chart. Every plot is wired
    `render_all` → `PLOT_POLICY` → `manifest.json` → `docs/js/app.js` (`PLOT_LABELS` + order); a
    plot is silently dropped from the site unless all three are updated. The site (`docs/`) also
    gained an **interactive WaveSurfer.js v7 player** (playhead, beat/downbeat ticks that light up,
    a glowing overlap region; falls back to `<audio>` if WebAudio is unavailable) and a canvas
    **beat-grid "snap"** animation (`docs/js/beatgrid.js`) driven by `bpm_a`/`bpm_b`/`stretch_ratio`
    — it honestly shows grids locking when B was stretched, or staying offset when the tempo gap
    exceeded tolerance. `build_site_assets.py` emits a per-demo `viz` block mapping the three
    different analysis clocks (A 1:1, B stretched, output) onto the trimmed MP3's clip clock.
  - **Site refresh — cut-demo gallery in the main-branch visual style:** the featured gallery is
    **five `--style cut` quick-cut demos** — four procedural (`victory_lap__ultimate` hits/riser,
    `skrillex__core` riff/riser) that `scripts/render_cut_demos.py` regenerates byte-identically to
    the hand-saved `outputs/<id>.wav` (each gets a sidecar + full plot suite named `outputs/<id>.json`,
    `<id>_*.png`), plus one **AI-generated** (`sao_paulo__put4__tier3`, `--generate` disco riser,
    `source: ai-clip`) packaged from its existing CLI render (NOT re-rendered — the `--generate`/
    riserize path isn't byte-reproducible). `build_site_assets.py` packages them as a manifest
    `demos[]` array; `build_demo(cfg)` reads title/effect/shape from the sidecar `demo` tag when
    present (render_cut_demos.py) else derives them from `cfg` + `plan.bridge.element`/`plan.shape`
    (so tag-less CLI renders like São Paulo work). `slim_metrics` is None-safe (beat-align is
    near-meaningless for cuts). An **Evaluation** section (`manifest.comparisons[]`,
    `build_comparison`) shows a tier-1/2/3 bar chart + metric table per pair (`EVAL_PAIRS`, rendered
    `--style auto` by render_cut_demos.py) — loudness continuity is the clean win; beat-align is
    cue-sensitive on full tracks (stated on the page). The `docs/` frontend adopts the
    **`origin/main:website/` design system** (animated signal canvas, glassmorphic dark theme, hero +
    transition-plan console, interactive 5-stage pipeline board, tier explainer) while keeping the
    data-driven functionality (WaveSurfer player, beat ticks, glowing overlap, per-demo beat-grid snap,
    metric cards, collapsible 12-plot gallery, course topics). The old Google-survey + deploy sections
    were removed. NOTE: the old `song_A__song_B` *source* audio is gone (only `song_B.mp3` in `data/`).
  - **Still TODO:** listening survey; run on a real-song library; tune novelty weights; tune the
    per-genre preset numbers + element prompts by ear; **track-driven moves** (echo/delay throw on
    the outgoing tail, acapella bridge, hard cut) as a follow-up to the additive generated layer.
- **Final presentation:** demo, survey results, tested audio examples, project website.

**De-risking rule:** test each module on short, consistent clips first. If any tool is too slow/unreliable to install or run, **simplify** — drop to librosa-only beats, pre-generate a small AI-clip library, or cut a feature. Find out early.
