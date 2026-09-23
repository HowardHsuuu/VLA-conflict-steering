from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "assets/rollouts/spatial_scene10_noise457"


def test_disclosed_pair_is_self_consistent_and_scoped_as_development_only() -> None:
    manifest = json.loads((EXAMPLE / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["artifact_kind"] == "disclosed_outcome_selected_rollout_pair"
    assert "Outcome-selected" in manifest["selection_policy"]
    assert "not a random sample" in manifest["selection_policy"]
    assert "not part of the 660-rollout result table" in manifest["selection_policy"]
    assert manifest["task"] == "spatial-ramekin-vs-cookie-box"
    assert manifest["episode_index"] == 10
    assert manifest["noise_seed"] == 457
    assert manifest["conditions"] == ["conflict", "learned_monitor"]

    observed_files = {path.name for path in EXAMPLE.iterdir() if path.is_file()}
    assert observed_files == {"conflict.mp4", "learned_monitor.mp4", "manifest.json"}
    for condition in manifest["conditions"]:
        outcome = manifest["outcomes"][condition]
        video = manifest["videos"][condition]
        path = EXAMPLE / video["path"]
        assert path.name == f"{condition}.mp4"
        assert path.stat().st_size > 0
        assert b"ftyp" in path.read_bytes()[:32]
        assert video["codec"] == "libx264"
        assert video["pixel_format"] == "yuv420p"
        assert video["frame_shape"] == [256, 256, 3]
        assert video["frame_count"] == math.ceil(outcome["steps"] / manifest["frame_stride"])
