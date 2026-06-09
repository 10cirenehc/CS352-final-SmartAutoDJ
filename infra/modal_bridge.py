"""Modal app: generate the SmartAutoDJ tier-3 bridge on a cloud CUDA GPU.

Replaces the old Google-Colab generative workflow. The heavy stack (audiocraft
MusicGen-Style + torch 2.1 + ffmpeg) is baked into a Modal **image once** and the
1.5 B model weights are cached on a Modal **Volume**, so there is no per-session
reinstall — unlike Colab. The local pipeline never installs any of this; it only
needs the lightweight ``modal`` client (extra ``gen``).

This file lifts the working Colab cells (``notebooks/generative_musicgen.ipynb``)
into a deployable function. MusicGen-Style does true **audio-style conditioning**
on the A->B boundary reference, so the bridge inherits the transition's timbre.

Setup (once):
    pip install modal
    modal setup                          # interactive browser auth
    modal deploy infra/modal_bridge.py   # builds the image, caches weights

Standalone one-command use (upload -> generate -> download):
    modal run infra/modal_bridge.py \
        --ref outputs/A__B__tier3_boundary_ref.wav \
        --prompt "124 BPM electronic transition riser in A minor ..." \
        --out assets/generated/A__B_bridge.wav --duration 14

Inline use: the local pipeline calls the deployed function via
``smartautodj.remote`` when run with ``--generate``.
"""

import modal

app = modal.App("smartautodj-bridge")

# Weights persist here across calls (downloaded once on first invocation).
cache = modal.Volume.from_name("musicgen-style-cache", create_if_missing=True)

# The ONLY place audiocraft / torch 2.1 live — isolated from the local smartdj env
# (torch 2.12). audiocraft pins torch==2.1.0 and its own xformers; install torch
# FIRST (separate layer) so audiocraft/xformers build against it, then audiocraft
# (which pulls a torch-2.1-compatible xformers itself — don't pin it here).
# python 3.10: audiocraft is most proven on 3.9/3.10 (3.11 is less tested).
image = (
    modal.Image.debian_slim(python_version="3.10")
    # ffmpeg (runtime) + the toolchain & ffmpeg dev headers PyAV (av==11.0.0, an
    # audiocraft dep) needs to build from source: pkg-config, a C compiler, and the
    # libav* -dev libraries. Without these the build fails with
    # "pkg-config is required for building PyAV".
    .apt_install(
        "ffmpeg",
        "git",  # pip needs it to install audiocraft from the GitHub repo
        "pkg-config",
        "build-essential",
        "libavformat-dev",
        "libavcodec-dev",
        "libavdevice-dev",
        "libavutil-dev",
        "libavfilter-dev",
        "libswscale-dev",
        "libswresample-dev",
    )
    .pip_install("torch==2.1.0", "torchaudio==2.1.0")
    # Pin a transformers known-good with audiocraft + torch 2.1 (its T5 text
    # conditioner needs torch-aware transformers; a too-new auto-pulled version
    # reports "PyTorch not found"). sentencepiece = the T5 tokenizer backend.
    .pip_install("transformers==4.41.2", "sentencepiece")
    # Install audiocraft from git main, NOT the PyPI 1.3.0 wheel: the MusicGen-Style
    # "style" conditioner was added after 1.3.0, so the released wheel errors with
    # "Unrecognized conditioning model: style" when loading facebook/musicgen-style.
    .pip_install("git+https://github.com/facebookresearch/audiocraft.git")
    .env({"HF_HOME": "/cache", "TORCH_HOME": "/cache"})
)

MODEL_ID = "facebook/musicgen-style"


@app.function(image=image, gpu="A10G", volumes={"/cache": cache}, timeout=900)
def generate_bridge_remote(
    ref_wav: bytes, prompt: str, duration: float, params: dict
) -> bytes:
    """Generate one bridge clip conditioned on ``ref_wav`` + ``prompt``.

    Returns WAV bytes (MusicGen's native 32 kHz). ``params`` overrides the
    generation/style knobs (cfg_coef, cfg_coef_beta, eval_q, excerpt_length, top_k).
    """
    import io

    import torchaudio
    from audiocraft.models import MusicGen

    params = params or {}
    model = MusicGen.get_pretrained(MODEL_ID)
    model.set_generation_params(
        duration=float(duration),
        use_sampling=True,
        top_k=int(params.get("top_k", 250)),
        cfg_coef=float(params.get("cfg_coef", 3.0)),
        cfg_coef_beta=float(params.get("cfg_coef_beta", 5.0)),  # joint text+style
    )
    ref, sr = torchaudio.load(io.BytesIO(ref_wav))
    ref = ref.mean(0, keepdim=True)  # mono
    # eval_q: style fidelity (1..6, lower = looser); excerpt_length: seconds of ref used.
    # The doc requires 1.5 <= excerpt_length <= reference length, so clamp to the
    # actual ref duration (short overlaps would otherwise error).
    ref_secs = ref.shape[-1] / sr
    excerpt = max(1.5, min(float(params.get("excerpt_length", 3.0)), ref_secs - 0.05))
    model.set_style_conditioner_params(eval_q=int(params.get("eval_q", 1)), excerpt_length=excerpt)

    # Positional args (descriptions, melody/style wav, sample_rate) — the kwarg
    # names drift across audiocraft versions (melody / melody_wavs), positional is stable.
    wav = model.generate_with_chroma([prompt], ref[None].expand(1, -1, -1), sr)

    buf = io.BytesIO()
    torchaudio.save(buf, wav[0].cpu(), model.sample_rate, format="wav")
    cache.commit()  # persist any newly-downloaded weights
    return buf.getvalue()


@app.local_entrypoint()
def main(ref: str, prompt: str, out: str, duration: float = 8.0):
    """Standalone: read a local reference WAV, generate, write the clip locally."""
    with open(ref, "rb") as f:
        data = f.read()
    clip = generate_bridge_remote.remote(data, prompt, float(duration), {})
    with open(out, "wb") as f:
        f.write(clip)
    print(f"wrote {out} ({len(clip)} bytes)")
