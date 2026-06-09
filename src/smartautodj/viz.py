"""Plots for inspection and the writeup: waveforms + beats, spectrograms,
fade curves, and the assembled transition region.

Uses the non-interactive Agg backend so it runs headless (CI, servers). Each
function saves a PNG and returns its path; ``render_all`` produces the standard
set for one pipeline run.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import
import librosa
import librosa.display
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from . import structure as structure_mod
from .align import matched_beats_in_overlap, scale_times
from .evaluate import short_term_loudness_db
from .transition import bass_swap_curves, fade_curves, split_bands
from .types import AnalysisResult, TransitionPlan


def plot_waveforms_with_beats(
    y_a: np.ndarray,
    y_b: np.ndarray,
    a: AnalysisResult,
    b: AnalysisResult,
    plan: TransitionPlan,
    sr: int,
    out_path: str,
) -> str:
    """Stacked A/B waveforms with beat (thin) and downbeat (thick) markers and
    the chosen transition region shaded."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    for ax, y, res, region, name in (
        (axes[0], y_a, a, plan.region_a, "Song A (outgoing)"),
        (axes[1], y_b, b, plan.region_b, "Song B (incoming)"),
    ):
        t = np.arange(len(y)) / sr
        ax.plot(t, y, lw=0.4, color="#3a6ea5")
        for bt in res.beats:
            ax.axvline(bt, color="0.7", lw=0.3)
        for db in res.downbeats:
            ax.axvline(db, color="#d1495b", lw=0.8)
        ax.axvspan(region[0], region[1], color="#ffd166", alpha=0.3, label="transition")
        ax.set_title(f"{name} — {res.bpm:.1f} BPM ({res.backend})")
        ax.set_ylabel("amp")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    return _save(fig, out_path)


def plot_spectrogram(y: np.ndarray, sr: int, out_path: str, title: str = "Spectrogram") -> str:
    """Log-frequency power spectrogram (Fourier Transforms & Spectrograms topic)."""
    fig, ax = plt.subplots(figsize=(12, 4))
    s_db = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(s_db, sr=sr, x_axis="time", y_axis="log", ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    return _save(fig, out_path)


def plot_fade_curves(plan: TransitionPlan, sr: int, out_path: str, n: int = 1000) -> str:
    """The fade-out / fade-in gain envelopes used over the overlap."""
    fade_out, fade_in = fade_curves(n, plan.fade_shape)
    x = np.linspace(0, plan.overlap_sec, n)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, fade_out, label="A fade-out", color="#3a6ea5")
    ax.plot(x, fade_in, label="B fade-in", color="#d1495b")
    eq = plan.eq_params or {}
    if eq.get("kind") == "bass_swap":
        bass_out, bass_in = bass_swap_curves(
            n, float(eq.get("swap_center", 0.5)), float(eq.get("swap_width", 0.2))
        )
        ax.plot(x, bass_out, "--", color="#3a6ea5", lw=0.8, label="A bass")
        ax.plot(x, bass_in, "--", color="#d1495b", lw=0.8, label="B bass")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("gain")
    ax.set_title(f"Transition fades ({plan.fade_shape})")
    ax.legend(fontsize=8)
    return _save(fig, out_path)


def plot_transition_region(y_out: np.ndarray, info: dict, sr: int, out_path: str) -> str:
    """The rendered output waveform with the overlap span highlighted."""
    t = np.arange(len(y_out)) / sr
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, y_out, lw=0.4, color="#2a9d8f")
    o0 = info["overlap_start_sample"] / sr
    o1 = (info["overlap_start_sample"] + info["overlap_len_sample"]) / sr
    ax.axvspan(o0, o1, color="#ffd166", alpha=0.35, label="overlap")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amp")
    ax.set_title("Rendered transition")
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, out_path)


def plot_structure_cue(
    a: AnalysisResult, b: AnalysisResult, plan: TransitionPlan, out_path: str
) -> str:
    """Energy + novelty curves for A and B with the chosen cue points marked.

    Shows *why* the transition points were picked: A exits from a sustained part;
    B drops in where energy is high (past the intro). Energy = loudness envelope,
    novelty = where the music structurally changes (Self-Similarity topic)."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    panels = (
        (axes[0], a, plan.region_a[0], "Song A (outgoing) — exit point", "#3a6ea5"),
        (axes[1], b, plan.region_b[0], "Song B (incoming) — drop-in point", "#d1495b"),
    )
    # B's cue is shown in the post-stretch timeline (region_b is already scaled);
    # B's curves live in the original timeline, so scale their time axis to match.
    for ax, res, cue, title, color in panels:
        hop = getattr(res, "rms_hop_sec", 0.0)
        rms = getattr(res, "rms", np.array([]))
        nov = getattr(res, "novelty", np.array([]))
        scale = 1.0
        if res is b and abs(plan.stretch_ratio - 1.0) >= 1e-3:
            scale = 1.0 / plan.stretch_ratio
        if rms.size and hop > 0:
            t = np.arange(rms.size) * hop * scale
            ax.plot(t, rms, color=color, lw=1.0, label="energy (RMS)")
        if nov.size and hop > 0:
            t = np.arange(nov.size) * hop * scale
            ax.plot(t, nov, color="0.5", lw=0.8, label="novelty")
        ax.axvline(cue, color="#06d6a0", lw=2.0, label="chosen cue")
        for db in res.downbeats * scale:
            ax.axvline(db, color="0.85", lw=0.3, zorder=0)
        key = getattr(res, "key", {}) or {}
        ax.set_title(f"{title}   [key: {key.get('name', '?')}]")
        ax.set_ylabel("0..1")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    sel = getattr(plan, "selection", {}) or {}
    if sel.get("style"):
        fig.suptitle(
            f"style: {sel['style']}  (genre A={sel.get('genre_a', '?')}, "
            f"B={sel.get('genre_b', '?')})",
            fontsize=10,
        )
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Harmony — Camelot / circle-of-fifths key wheel (Pitch / Chroma topic)
# --------------------------------------------------------------------------- #
def plot_key_wheel(a: AnalysisResult, b: AnalysisResult, plan: TransitionPlan, out_path: str) -> str:
    """A and B's keys on a circle-of-fifths wheel, joined by a compatibility arc.

    DJs mix harmonically by the Camelot wheel: keys a perfect-fifth apart (adjacent
    on the wheel) or relative major/minor blend smoothly; distant keys clash. We
    place each track at its tonic's circle-of-fifths position (outer ring = major,
    inner = minor) and colour the connecting line by the harmonic-compatibility
    score the pipeline already computed (``structure.key_compatibility``)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.axis("off")
    r_out, r_in = 1.0, 0.66

    def angle_for(tonic: int) -> float:
        # circle of fifths: step by 7 semitones per slot; 12 o'clock = C, clockwise.
        pos = (int(tonic) * 7) % 12
        return np.pi / 2 - pos * (2 * np.pi / 12)

    ax.add_patch(plt.Circle((0, 0), r_out, fill=False, color="0.8", lw=0.8))
    ax.add_patch(plt.Circle((0, 0), r_in, fill=False, color="0.85", lw=0.8))
    for tonic in range(12):
        ang = angle_for(tonic)
        for r, suffix in ((r_out, ""), (r_in, "m")):
            ax.text(r * np.cos(ang), r * np.sin(ang),
                    structure_mod._PITCHES[tonic] + suffix,
                    ha="center", va="center", fontsize=7, color="0.55")

    pts = {}
    for res, color, lab in ((a, "#3a6ea5", "A"), (b, "#d1495b", "B")):
        key = getattr(res, "key", {}) or {}
        tonic, mode = key.get("tonic"), key.get("mode")
        if tonic is None:
            continue
        r = r_out if mode == "major" else r_in
        ang = angle_for(int(tonic))
        x, y = r * np.cos(ang), r * np.sin(ang)
        ax.scatter([x], [y], s=420, color=color, zorder=5, edgecolors="white", linewidths=1.5)
        ax.text(x, y, lab, ha="center", va="center", color="white",
                fontsize=11, fontweight="bold", zorder=6)
        pts[lab] = (x, y, key.get("name", "?"))

    compat = (plan.selection or {}).get("harmonic_compatibility")
    if compat is None:
        compat = structure_mod.key_compatibility(getattr(a, "key", {}), getattr(b, "key", {}))
    if "A" in pts and "B" in pts:
        (xa, ya, _), (xb, yb, _) = pts["A"], pts["B"]
        ax.plot([xa, xb], [ya, yb], color=cm.RdYlGn(float(np.clip(compat, 0, 1))),
                lw=3.5, zorder=4, solid_capstyle="round")
    ka = pts.get("A", (0, 0, "?"))[2]
    kb = pts.get("B", (0, 0, "?"))[2]
    ax.set_title(
        f"Harmonic compatibility (Camelot / circle of fifths)\n"
        f"A: {ka}    B: {kb}    compatibility = {compat:.2f}", fontsize=10)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Self-similarity matrices (Self-Similarity topic)
# --------------------------------------------------------------------------- #
def _self_similarity(y: np.ndarray, sr: int, beats: np.ndarray | None, feature: str) -> np.ndarray:
    """Beat-synchronous self-similarity (affinity) matrix for one track.

    Recomputes chroma (harmony) or MFCC (timbre) from the audio — these are
    discarded after analysis to keep the sidecar small — then beat-synchronises
    the frames so the matrix is O(beats^2) (a full O(frames^2) matrix is far too
    large for a multi-minute track) and gridded to the musical pulse. The classic
    Foote self-similarity matrix: bright off-diagonal blocks = repeated sections."""
    y = np.asarray(y, dtype=np.float32)
    if feature == "mfcc":
        feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=structure_mod.HOP)
    else:
        feat = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=structure_mod.HOP)
    if beats is not None and np.asarray(beats).size > 4:
        bf = librosa.time_to_frames(np.asarray(beats), sr=sr, hop_length=structure_mod.HOP)
        bf = bf[(bf >= 0) & (bf < feat.shape[1])]
        if bf.size > 4:
            feat = librosa.util.sync(feat, bf, aggregate=np.median)
    # Cap size if we couldn't beat-sync (degenerate beats): subsample columns.
    if feat.shape[1] > 500:
        idx = np.linspace(0, feat.shape[1] - 1, 500).astype(int)
        feat = feat[:, idx]
    return librosa.segment.recurrence_matrix(feat, mode="affinity", sym=True)


def plot_ssm_pair(
    y_a: np.ndarray, y_b: np.ndarray, a: AnalysisResult, b: AnalysisResult,
    plan: TransitionPlan, sr: int, out_path: str, feature: str = "chroma",
) -> str:
    """Side-by-side beat-synchronous self-similarity matrices for A and B."""
    # B's audio here is already time-stretched, so map its beats into that clock.
    b_beats = scale_times(b.beats, plan.stretch_ratio)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, y, beats, name in (
        (axes[0], y_a, a.beats, "Song A"),
        (axes[1], y_b, b_beats, "Song B"),
    ):
        r = _self_similarity(y, sr, beats, feature)
        img = ax.imshow(r, origin="lower", aspect="equal", cmap="magma",
                        interpolation="nearest")
        ax.set_title(f"{name} — self-similarity ({feature})")
        ax.set_xlabel("beat")
        ax.set_ylabel("beat")
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Arrangement / structure ribbon
# --------------------------------------------------------------------------- #
def plot_structure_ribbon(
    a: AnalysisResult, b: AnalysisResult, plan: TransitionPlan, out_path: str
) -> str:
    """Horizontal coloured segment bands per track, energy overlaid, cue marked.

    Shows the song *arrangement* as a DJ-software-style lane. Segment labels are
    auto-detected (``seg0..segN``) — without a structure model we don't claim
    intro/verse/chorus names, only the boundaries."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.5), sharex=False)
    scale_b = 1.0 / plan.stretch_ratio if abs(plan.stretch_ratio - 1.0) >= 1e-3 else 1.0
    panels = (
        (axes[0], a, plan.region_a[0], "Song A (outgoing)", 1.0),
        (axes[1], b, plan.region_b[0], "Song B (incoming)", scale_b),
    )
    for ax, res, cue, name, scale in panels:
        sections = getattr(res, "sections", []) or []
        n = max(len(sections), 1)
        for i, s in enumerate(sections):
            x0 = float(s.get("start", 0.0)) * scale
            x1 = float(s.get("end", x0)) * scale
            ax.axvspan(x0, x1, color=cm.viridis(i / max(n - 1, 1)), alpha=0.4)
        rms = getattr(res, "rms", np.array([]))
        hop = getattr(res, "rms_hop_sec", 0.0)
        if rms.size and hop > 0:
            t = np.arange(rms.size) * hop * scale
            ax.plot(t, rms, color="0.12", lw=0.9, label="energy (RMS)")
        ax.axvline(cue, color="#06d6a0", lw=2.0, label="chosen cue")
        ax.set_title(f"{name} — auto-detected segments")
        ax.set_ylabel("0..1")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Loudness continuity (Loudness & Amplitude topic)
# --------------------------------------------------------------------------- #
def plot_loudness_curve(y_out: np.ndarray, info: dict, sr: int, out_path: str) -> str:
    """Short-term loudness across the output, overlap shaded, max jump annotated.

    Uses the same short-term (~1 s) loudness curve as the ``loudness_continuity``
    metric, so a baseline hard cut shows a visible step here while a constant-power
    transition stays flat through the shaded overlap."""
    db, hop = short_term_loudness_db(y_out, sr)
    t = np.arange(db.size) * hop / sr
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(t, db, color="#2a9d8f", lw=1.2)
    o0 = info["overlap_start_sample"] / sr
    o1 = (info["overlap_start_sample"] + info["overlap_len_sample"]) / sr
    ax.axvspan(o0, o1, color="#ffd166", alpha=0.3, label="overlap")
    sf = max(info["overlap_start_sample"] // hop, 0)
    ef = min((info["overlap_start_sample"] + info["overlap_len_sample"]) // hop + 1, db.size)
    window = db[sf:ef]
    if window.size >= 2:
        d = np.abs(np.diff(window))
        j = int(np.argmax(d))
        jt = (sf + j) * hop / sr
        ax.annotate(f"max jump {d[j]:.1f} dB", xy=(jt, window[j]),
                    xytext=(jt, window[j] + 6), fontsize=8, color="#d1495b",
                    arrowprops=dict(arrowstyle="->", color="#d1495b"))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("short-term loudness (dB)")
    ax.set_title("Loudness continuity across the transition")
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Beat-alignment scatter (Beat tracking topic)
# --------------------------------------------------------------------------- #
def plot_beat_alignment(
    a: AnalysisResult, b: AnalysisResult, plan: TransitionPlan, out_path: str
) -> str:
    """Matched A/B beats in the overlap, on a common clock, vs the y=x ideal.

    Each A beat is paired to its nearest B beat (the same pairing ``evaluate``
    uses). Tight to the diagonal = well beat-matched (tier 2/3); scattered off it
    = the unaligned baseline (tier 1)."""
    a_t, b_t = matched_beats_in_overlap(a, b, plan)
    fig, ax = plt.subplots(figsize=(6, 6))
    if a_t.size:
        err = np.abs(a_t - b_t)
        lim = float(max(a_t.max(), b_t.max())) or 1.0
        ax.plot([0, lim], [0, lim], "--", color="0.6", lw=1.0, label="perfect (y=x)")
        sc = ax.scatter(a_t, b_t, c=err * 1000, cmap="RdYlGn_r", s=70, zorder=5,
                        edgecolors="white", linewidths=0.8, vmin=0)
        fig.colorbar(sc, ax=ax, label="|error| (ms)")
        ax.set_title(f"Beat alignment in overlap\nmean |Δ| = {err.mean() * 1000:.0f} ms "
                     f"({a_t.size} beats)")
        ax.legend(loc="upper left", fontsize=8)
    else:
        ax.set_title("Beat alignment — no matched beats in overlap")
    ax.set_xlabel("A beat time (s, overlap clock)")
    ax.set_ylabel("B beat time (s, overlap clock)")
    ax.set_aspect("equal", adjustable="datalim")
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Mel-spectrogram with the AI bridge region (Fourier / Spectrograms topic)
# --------------------------------------------------------------------------- #
def plot_mel_bridge(y_out: np.ndarray, plan: TransitionPlan, sr: int, out_path: str) -> str:
    """Mel-spectrogram of the output with the tier-3 bridge time-range shaded."""
    fig, ax = plt.subplots(figsize=(12, 4))
    s_db = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y_out, sr=sr, n_mels=128), ref=np.max
    )
    img = librosa.display.specshow(s_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    br = plan.bridge or {}
    if br.get("start") is not None and br.get("end") is not None:
        ax.axvspan(float(br["start"]), float(br["end"]), color="#06d6a0",
                   alpha=0.22, label="AI bridge")
        name = br.get("element") or br.get("kind") or "bridge"
        ax.text(float(br["start"]), ax.get_ylim()[1] * 0.9, f" {name}",
                color="#06d6a0", fontsize=9, va="top", fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Mel-spectrogram with AI bridge region")
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Bass-swap low-frequency handoff (Convolution & Filtering topic)
# --------------------------------------------------------------------------- #
def plot_bass_swap(
    y_a: np.ndarray, y_b: np.ndarray, plan: TransitionPlan, sr: int, out_path: str
) -> str:
    """Low-band energy of A handing off to B across the overlap, vs the EQ gains.

    Splits each track's overlap segment at the bass-swap cutoff and plots the
    low-band energy envelopes: A's bass holds, then B's takes over. The dashed
    equal-power swap gains (what ``transition.process_overlap`` actually applies)
    are overlaid — only one bassline dominates at a time, but total low energy
    never drops out (no hollow midpoint)."""
    eq = plan.eq_params or {}
    cutoff = float(eq.get("cutoff_hz", 200.0))
    order = int(eq.get("order", 4))
    a0, a1 = plan.region_a
    b0, b1 = plan.region_b
    a_tail = np.asarray(y_a)[int(a0 * sr):int(a1 * sr)]
    b_head = np.asarray(y_b)[int(b0 * sr):int(b1 * sr)]
    n = int(min(len(a_tail), len(b_head)))
    a_tail, b_head = a_tail[:n], b_head[:n]
    a_low, _ = split_bands(a_tail, sr, cutoff, order)
    b_low, _ = split_bands(b_head, sr, cutoff, order)

    win = max(int(sr * 0.05), 1)
    kern = np.ones(win) / win

    def env(x):
        return np.sqrt(np.convolve(np.asarray(x, dtype=float) ** 2, kern, mode="same"))

    t = np.arange(n) / sr
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, env(a_low), color="#3a6ea5", lw=1.2, label=f"A low band (<{cutoff:.0f} Hz)")
    ax.plot(t, env(b_low), color="#d1495b", lw=1.2, label=f"B low band (<{cutoff:.0f} Hz)")
    ax.set_xlabel("overlap time (s)")
    ax.set_ylabel("low-band energy")
    ax.legend(loc="upper right", fontsize=8)

    if eq.get("kind") == "bass_swap" and n > 1:
        bass_out, bass_in = bass_swap_curves(
            n, float(eq.get("swap_center", 0.5)), float(eq.get("swap_width", 0.2))
        )
        ax2 = ax.twinx()
        ax2.plot(t, bass_out, "--", color="#3a6ea5", lw=0.8, alpha=0.6)
        ax2.plot(t, bass_in, "--", color="#d1495b", lw=0.8, alpha=0.6)
        ax2.set_ylabel("bass-swap gain (dashed)")
        ax2.set_ylim(0, 1.05)
        ax.set_title("Bass-swap low-frequency handoff")
    else:
        ax.set_title("Low-frequency energy across the overlap (no bass swap)")
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Genre classifier probabilities (Deep Learning / Embeddings topic)
# --------------------------------------------------------------------------- #
def plot_genre_probs(a: AnalysisResult, b: AnalysisResult, out_path: str) -> str:
    """Per-track GTZAN class probabilities from the distilHuBERT genre classifier.

    Shows the neural classifier "thinking": the 10-way softmax over GTZAN genres
    for each track, top class highlighted. Degrades to an "unavailable" panel when
    genre wasn't classified (e.g. ``--genre`` override or transformers absent)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, res, color, name in (
        (axes[0], a, "#3a6ea5", "Song A"),
        (axes[1], b, "#d1495b", "Song B"),
    ):
        genre = getattr(res, "genre", {}) or {}
        probs = genre.get("probs") or {}
        if probs:
            items = sorted(probs.items(), key=lambda kv: kv[1])
            labels = [k for k, _ in items]
            vals = [v for _, v in items]
            top = genre.get("label")
            ax.barh(labels, vals, color=[color if k == top else "0.6" for k in labels])
            ax.set_xlim(0, 1)
            ax.set_xlabel("probability")
            conf = genre.get("confidence")
            conf_s = f" ({conf:.0%})" if isinstance(conf, (int, float)) else ""
            ax.set_title(f"{name} — {top}{conf_s}")
        else:
            ax.text(0.5, 0.5, "genre unavailable", ha="center", va="center",
                    transform=ax.transAxes, color="0.5")
            ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Genre classifier (distilHuBERT / GTZAN)", fontsize=11)
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Cross-tier evaluation summary (built by the site script, which sees all tiers)
# --------------------------------------------------------------------------- #
def plot_tier_comparison(metrics_by_tier: dict, out_path: str) -> str:
    """Grouped bars comparing tiers 1/2/3 on the three objective metrics.

    The "what each tier buys" chart: beat-alignment error, max loudness jump, and
    post-stretch tempo gap (lower is better on all three). ``metrics_by_tier`` maps
    a tier int to that run's ``sidecar["metrics"]`` dict. Unlike the per-run plots,
    this needs all three tiers at once, so the site build script calls it."""
    tiers = sorted(metrics_by_tier)
    specs = [
        ("beat_alignment", "mean_abs_error_sec", 1000.0, "Beat-align error (ms)"),
        ("loudness_continuity", "max_db_jump", 1.0, "Max loudness jump (dB)"),
        ("tempo_match", "bpm_gap_after", 1.0, "Tempo gap after stretch (BPM)"),
    ]
    colors = {1: "#bbbbbb", 2: "#3a6ea5", 3: "#7c5cff"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (grp, key, scale, label) in zip(np.atleast_1d(axes), specs):
        vals, labs, cols = [], [], []
        for t in tiers:
            v = (metrics_by_tier[t].get(grp) or {}).get(key)
            vals.append(float(v) * scale if v is not None else 0.0)
            labs.append(f"T{t}")
            cols.append(colors.get(t, "#888888"))
        bars = ax.bar(labs, vals, color=cols, width=0.6)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
        ax.set_title(label, fontsize=10)
        ax.margins(y=0.2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Tier comparison — lower is better", fontsize=12)
    return _save(fig, out_path)


def render_all(
    y_a, y_b, y_out, a, b, plan, info, sr, out_dir, stem: str
) -> list[str]:
    """Produce the standard plot set for one run; returns the saved paths."""
    os.makedirs(out_dir, exist_ok=True)
    p = lambda suffix: os.path.join(out_dir, f"{stem}_{suffix}.png")
    paths = [
        plot_waveforms_with_beats(y_a, y_b, a, b, plan, sr, p("waveforms")),
        plot_fade_curves(plan, sr, p("fades")),
        plot_transition_region(y_out, info, sr, p("region")),
        plot_spectrogram(y_out, sr, p("spectrogram"), title=f"{stem} output"),
        # Track-level / continuity visuals — meaningful in every tier (tier 1's
        # baseline is the contrast: clashing keys, scattered beats, a loudness jump).
        plot_key_wheel(a, b, plan, p("keywheel")),
        plot_ssm_pair(y_a, y_b, a, b, plan, sr, p("ssm")),
        plot_loudness_curve(y_out, info, sr, p("loudness")),
        plot_beat_alignment(a, b, plan, p("beatmatch")),
        plot_genre_probs(a, b, p("genre")),
    ]
    # Structure / EQ visuals (skip for tier-1 baseline, which ignores structure).
    if plan.tier >= 2:
        paths.append(plot_structure_cue(a, b, plan, p("structure")))
        paths.append(plot_structure_ribbon(a, b, plan, p("ribbon")))
        paths.append(plot_bass_swap(y_a, y_b, plan, sr, p("bassswap")))
    # Mel-spectrogram + bridge region only exists at tier 3 (the bridge is the point).
    if plan.tier == 3:
        paths.append(plot_mel_bridge(y_out, plan, sr, p("melbridge")))
    return paths


def _save(fig, out_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
