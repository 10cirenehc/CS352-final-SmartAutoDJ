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
import matplotlib.pyplot as plt
import numpy as np

from .transition import bass_swap_curves, fade_curves
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
    ]
    # Cue-point rationale plot (skip for tier-1 baseline, which ignores structure).
    if plan.tier >= 2:
        paths.append(plot_structure_cue(a, b, plan, p("structure")))
    return paths


def _save(fig, out_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
