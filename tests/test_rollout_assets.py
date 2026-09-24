from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "assets/rollouts/spatial_scene10_noise457"
GALLERY = ROOT / "assets/rollouts/cross_task_decisive_gallery_v1"
GALLERY_TASKS = {
    "object-ketchup-vs-absent-milk",
    "spatial-cookie-box-vs-cabinet",
    "spatial-ramekin-vs-cookie-box",
}


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


def test_cross_task_gallery_is_hash_bound_and_explicitly_outcome_selected() -> None:
    summary = json.loads((GALLERY / "manifest.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["pair_count"] == 3
    assert summary["video_count"] == 6
    assert set(summary["tasks"]) == GALLERY_TASKS
    assert "not new trials" in summary["interpretation_limit"]

    observed_tasks = {path.name for path in GALLERY.iterdir() if path.is_dir()}
    assert observed_tasks == GALLERY_TASKS
    for task in sorted(GALLERY_TASKS):
        task_dir = GALLERY / task
        manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "verified_exact_outcome_replay"
        assert manifest["task"] == task
        assert manifest["matched_initial_frame"] is True
        assert set(manifest["rendered"]) == {"conflict", "learned_monitor"}
        assert manifest["rendered"]["conflict"]["outcome"]["success"] is False
        assert manifest["rendered"]["learned_monitor"]["outcome"]["success"] is True
        assert {path.name for path in task_dir.iterdir() if path.is_file()} == {
            "conflict.mp4",
            "learned_monitor.mp4",
            "manifest.json",
        }
        assert "/Users/" not in json.dumps(manifest)
        for condition, item in manifest["rendered"].items():
            video = item["video"]
            path = task_dir / video["path"]
            assert path.name == f"{condition}.mp4"
            assert path.stat().st_size == video["mp4_size_bytes"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == video["mp4_sha256"]
            assert video["frame_count"] == item["outcome"]["steps"] + 1
            assert video["decoded_frame_count"] == video["frame_count"]
