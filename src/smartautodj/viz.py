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

from .transition import fade_curves
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
    if (plan.eq_params or {}).get("kind") == "bass_swap":
        t = np.linspace(0, 1, n)
        ax.plot(x, np.clip(1 - 2 * t, 0, 1), "--", color="#3a6ea5", lw=0.8, label="A bass")
        ax.plot(x, np.clip(2 * t - 1, 0, 1), "--", color="#d1495b", lw=0.8, label="B bass")
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
    return paths


def _save(fig, out_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
