# Generated AI clips

Small AI-generated audio clips (risers, drum fills, synth swells, ambient pads) produced
in **Colab** (MusicGen / Stable Audio Open — Env B in `CLAUDE.md`) and committed here as
static assets.

The local pipeline treats these as inputs it only **trims, normalizes, and length-matches**
via `smartautodj.generative.prepare_clip` — the same code path the procedural placeholder
bridge uses. To use a real clip instead of the procedural one, load it and pass it through
`prepare_clip(clip, sr, overlap_sec)` before mixing.

Keep clips short and small so the repo stays lightweight.
