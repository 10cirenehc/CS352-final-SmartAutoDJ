"""Tests for the Modal generation glue (no real Modal account / GPU needed).

Real generation runs on Modal (CUDA) and is exercised manually (Step 0 in the
plan / CLAUDE.md §5). Here we mock the `modal` boundary and verify the local
seam: WAV encoding + file writing, the import-error path, and that the pipeline's
`--generate` flow resolves the produced clip as a real `ai-clip`.
"""

from __future__ import annotations

import json
import os
import sys
import types

import numpy as np
import pytest
import soundfile as sf

from smartautodj import remote
from smartautodj.pipeline import run


def test_generate_bridge_to_file_encodes_and_writes(tmp_path, monkeypatch):
    """ref array -> WAV bytes -> Modal -> written clip; args threaded correctly."""
    seen = {}

    def fake_via_modal(ref_wav, prompt, duration, params=None):
        seen["ref_wav"] = ref_wav
        seen["prompt"] = prompt
        seen["duration"] = duration
        # Echo back a tiny valid WAV as the "generated" clip.
        import io
        buf = io.BytesIO()
        sf.write(buf, np.zeros(2048, dtype=np.float32), 32000, format="WAV")
        return buf.getvalue()

    monkeypatch.setattr(remote, "generate_bridge_via_modal", fake_via_modal)
    out = tmp_path / "sub" / "clip.wav"  # nested dir must be created
    ref = np.linspace(-1, 1, 4096).astype(np.float32)
    result = remote.generate_bridge_to_file(ref, 22050, "a riser", 8.0, str(out))

    assert result == str(out)
    assert out.exists()
    assert seen["prompt"] == "a riser" and seen["duration"] == 8.0
    # The reference was encoded to real WAV bytes (RIFF header).
    assert seen["ref_wav"][:4] == b"RIFF"


def test_generate_bridge_via_modal_requires_modal(monkeypatch):
    """A clear error (not an ImportError stacktrace) when modal is unavailable."""
    monkeypatch.setitem(sys.modules, "modal", None)  # force `import modal` to fail
    with pytest.raises(RuntimeError, match="modal not installed"):
        remote.generate_bridge_via_modal(b"RIFF....", "p", 8.0)


def test_generate_bridge_via_modal_invokes_remote(monkeypatch):
    """The deployed function is looked up by name and called with our args."""
    calls = {}

    class _Fn:
        def remote(self, *args):
            calls["args"] = args
            return b"CLIPBYTES"

    fake_modal = types.SimpleNamespace(
        Function=types.SimpleNamespace(from_name=lambda app, fn: (calls.setdefault("name", (app, fn)), _Fn())[1])
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    out = remote.generate_bridge_via_modal(b"REF", "prompt", 12.0, {"eval_q": 2})
    assert out == b"CLIPBYTES"
    assert calls["name"] == (remote.MODAL_APP, remote.MODAL_FN)
    assert calls["args"] == (b"REF", "prompt", 12.0, {"eval_q": 2})


def test_pipeline_generate_resolves_ai_clip(click_tracks, tmp_path, monkeypatch):
    """`--generate` (generate=True) produces a clip that resolves as `ai-clip`.

    The Modal call is mocked to write a clip into a tmp path (not assets/), so no
    repo pollution and no GPU needed.
    """
    a_path, b_path, sr = click_tracks
    clip_dst = tmp_path / "gen_clip.wav"

    def fake_to_file(ref_y, sr_, prompt, duration, out_path, params=None):
        # Ignore the requested assets/ path; write a real clip to tmp instead.
        sf.write(clip_dst, np.sin(np.linspace(0, 50, int(duration * sr_))).astype(np.float32), sr_)
        return str(clip_dst)

    monkeypatch.setattr(remote, "generate_bridge_to_file", fake_to_file)
    result = run(
        a_path, b_path, tier=3, out_dir=str(tmp_path / "out"), bars=4,
        backend="librosa", make_plots=False, sr=sr, style="default", generate=True,
    )
    side = json.loads(open(result["sidecar"]).read())
    assert side["plan"]["bridge"]["source"] == "ai-clip"
    assert side["params"]["generate"] is True


def test_urban_adds_procedural_bridge_by_default(click_tracks, tmp_path):
    """A default-on genre (hip-hop -> urban) adds its element at tier 3, no Modal."""
    a_path, b_path, sr = click_tracks
    result = run(
        a_path, b_path, tier=3, out_dir=str(tmp_path / "out"), bars=4,
        backend="librosa", make_plots=False, sr=sr, style="urban",
    )
    bridge = json.loads(open(result["sidecar"]).read())["plan"]["bridge"]
    from smartautodj import genre
    assert bridge["element"] in genre.BRIDGE_SPECS  # a real effect was chosen
    assert bridge["source"] != "ai-clip"  # procedural stand-in, no --generate


def test_clip_cache_key_includes_element(click_tracks, tmp_path, monkeypatch):
    """Two different effects on the same pair must resolve to DIFFERENT clip paths,
    so one effect's clip can't be served (and mislabeled) for another."""
    a_path, b_path, sr = click_tracks
    captured = []

    def fake_to_file(ref_y, sr_, prompt, duration, out_path, params=None):
        captured.append(out_path)
        dst = tmp_path / (os.path.basename(out_path))
        sf.write(dst, np.zeros(int(duration * sr_), dtype=np.float32), sr_)
        return str(dst)

    monkeypatch.setattr(remote, "generate_bridge_to_file", fake_to_file)
    for eff in ("riser", "impact"):
        run(a_path, b_path, tier=3, out_dir=str(tmp_path / eff), bars=4,
            backend="librosa", make_plots=False, sr=sr, style="urban", effect=eff, generate=True)
    # Distinct effect -> distinct cache filename.
    assert len(captured) == 2 and captured[0] != captured[1]
    assert "riser" in captured[0] and "impact" in captured[1]


def test_rock_bridge_is_on_demand_only(click_tracks, tmp_path, monkeypatch):
    """rock/band adds no bridge by default, but --generate produces its accent."""
    a_path, b_path, sr = click_tracks
    r1 = run(a_path, b_path, tier=3, out_dir=str(tmp_path / "o1"), bars=4,
             backend="librosa", make_plots=False, sr=sr, style="band")
    assert json.loads(open(r1["sidecar"]).read())["plan"]["bridge"] is None

    clip_dst = tmp_path / "clip.wav"

    def fake_to_file(ref_y, sr_, prompt, duration, out_path, params=None):
        sf.write(clip_dst, np.zeros(int(duration * sr_), dtype=np.float32), sr_)
        return str(clip_dst)

    monkeypatch.setattr(remote, "generate_bridge_to_file", fake_to_file)
    r2 = run(a_path, b_path, tier=3, out_dir=str(tmp_path / "o2"), bars=4,
             backend="librosa", make_plots=False, sr=sr, style="band", generate=True)
    bridge = json.loads(open(r2["sidecar"]).read())["plan"]["bridge"]
    assert bridge["element"] == "impact"
    assert bridge["source"] == "ai-clip"
