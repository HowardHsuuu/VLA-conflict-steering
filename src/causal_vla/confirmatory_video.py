"""Render the disclosed paper example from the frozen confirmatory protocol."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from causal_vla.confirmatory_monitor_runtime import run_confirmatory_monitor_evaluation
from causal_vla.integrations.action_atlas import activate_action_atlas
from causal_vla.public_results import sha256_file
from causal_vla.rollout_video import RolloutVideoEncoding, RolloutVideoEvidence, write_rollout_mp4

TASK_NAME = "spatial-ramekin-vs-cookie-box"
NOISE_SEED = 457
EPISODE_INDEX = 10
CONDITIONS = ("conflict", "learned_monitor")
REPORT_PATH = (
    "results/artifacts/results/monitor_steering_confirmatory_v3/"
    "spatial-ramekin-vs-cookie-box_noise457.json"
)


@dataclass
class _FrameSink:
    stride: int
    steps: int = 0
    frames: list[NDArray[np.uint8]] | None = None

    def __post_init__(self) -> None:
        self.frames = []

    def capture(self, observation: dict[str, Any], *, terminal: bool) -> None:
        self.steps += 1
        if self.steps % self.stride and not terminal:
            return
        pixels = observation.get("pixels")
        value = pixels.get("image") if isinstance(pixels, Mapping) else None
        if not isinstance(value, np.ndarray):
            raise TypeError("LIBERO observation has no pixels.image array")
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 3 or value.shape[2] != 3:
            raise ValueError("LIBERO pixels.image must be HWC RGB")
        if value.dtype != np.dtype(np.uint8):
            scale = 255.0 if float(value.max()) <= 1.0 else 1.0
            value = np.clip(value * scale, 0, 255).astype(np.uint8)
        # Robosuite exposes both axes inverted relative to display convention.
        frame = np.ascontiguousarray(value[::-1, ::-1]).copy()
        assert self.frames is not None
        self.frames.append(cast(NDArray[np.uint8], frame))


class _RecordingEnvironment:
    def __init__(self, environment: Any, sink: _FrameSink) -> None:
        self._environment = environment
        self._sink = sink

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
        result = self._environment.step(action)
        if not isinstance(result, tuple) or len(result) != 5:
            raise TypeError("LIBERO environment step must return a five-tuple")
        observation, reward, terminated, truncated, info = result
        if not isinstance(observation, dict) or not isinstance(info, dict):
            raise TypeError("LIBERO step returned malformed observation or info")
        terminal = bool(terminated or truncated or info.get("is_success", False))
        self._sink.capture(observation, terminal=terminal)
        return observation, reward, terminated, truncated, info


def _source_rows(root: Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    report_path = root / REPORT_PATH
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = {
        str(row["condition"]): cast(dict[str, Any], row)
        for row in report["episodes"]
        if int(row["episode_index"]) == EPISODE_INDEX and str(row["condition"]) in CONDITIONS
    }
    if set(rows) != set(CONDITIONS):
        raise ValueError("Published report does not contain the selected paired example")
    if bool(rows["conflict"]["success"]) or not bool(rows["learned_monitor"]["success"]):
        raise ValueError("Selected example no longer represents failure followed by recovery")
    return report_path, rows


def _video_payload(evidence: RolloutVideoEvidence, filename: str) -> dict[str, object]:
    payload = evidence.to_dict()
    payload["path"] = filename
    return payload


def render_disclosed_example(
    repository: str | Path,
    *,
    checkpoint: str | Path,
    action_atlas_root: str | Path,
    state_dir: str | Path,
    output: str | Path,
    device: str,
    frame_stride: int = 2,
    fps: int = 20,
) -> Path:
    """Rerun and encode the fixed, outcome-conditional paper example."""

    root = Path(repository).resolve(strict=True)
    report_path, expected = _source_rows(root)
    protocol_path = root / "configs/monitor_steering_confirmatory_v3.toml"
    protocol = tomllib.loads(protocol_path.read_text(encoding="utf-8"))
    task = next(item for item in protocol["tasks"] if item["name"] == TASK_NAME)
    output_path = Path(output).resolve()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing to replace rollout example {output_path}")
    temporary = output_path.with_name(f".{output_path.name}.partial")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Partial rollout directory already exists: {temporary}")
    temporary.mkdir(parents=True)

    sinks: list[_FrameSink] = []
    activate_action_atlas(action_atlas_root, state_dir)
    adapters = importlib.import_module("experiments.model_adapters")
    adapter_type = adapters.SmolVLAAdapter
    original_create_env = adapter_type.create_env

    def recording_create_env(adapter: Any, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        created = original_create_env(adapter, *args, **kwargs)
        if not isinstance(created, tuple) or len(created) != 3:
            raise TypeError("SmolVLAAdapter.create_env must return a three-tuple")
        environment, prompt, metadata = created
        sink = _FrameSink(frame_stride)
        sinks.append(sink)
        return _RecordingEnvironment(environment, sink), prompt, metadata

    adapter_type.create_env = recording_create_env
    try:
        evaluation = run_confirmatory_monitor_evaluation(
            checkpoint,
            root / task["steering_bank"],
            root / task["monitor_bank"],
            release_monitor_bank_path=root / task["release_monitor_bank"],
            task_id=int(task["task_id"]),
            episode_indices=(EPISODE_INDEX,),
            conditions=CONDITIONS,
            suite=str(task["suite"]),
            action_atlas_root=action_atlas_root,
            state_dir=state_dir,
            device=device,
            noise_seed=NOISE_SEED,
            simulator_seed=int(protocol["runtime"]["simulator_seed"]),
            random_seed=int(protocol["runtime"]["random_seed"]),
            max_steps=int(protocol["runtime"]["max_steps"]),
            grasp_close_fraction=float(task["grasp_close_fraction"]),
            grasp_reopen_fraction=float(task["grasp_reopen_fraction"]),
            grasp_lift_threshold=float(task["grasp_lift_threshold"]),
            grasp_release_patience=int(task["grasp_release_patience"]),
            release_reactivation_patience=int(task["release_reactivation_patience"]),
            release_mode=str(task["release_mode"]),
            release_monitor_interval=int(task["release_monitor_interval"]),
            monitor_arm_timeout=int(task["monitor_arm_timeout"]),
        )
    finally:
        adapter_type.create_env = original_create_env

    if len(sinks) != len(CONDITIONS) or len(evaluation.episodes) != len(CONDITIONS):
        raise RuntimeError("Recording wrapper did not observe exactly one rollout per condition")

    videos: dict[str, dict[str, object]] = {}
    outcomes: dict[str, dict[str, object]] = {}
    encoding = RolloutVideoEncoding(fps=fps)
    for condition, sink, outcome in zip(CONDITIONS, sinks, evaluation.episodes, strict=True):
        source = expected[condition]
        if (
            outcome.condition != condition
            or outcome.success is not bool(source["success"])
            or outcome.steps != int(source["steps"])
        ):
            raise RuntimeError(f"Rerun differs from the published {condition} outcome")
        if not sink.frames:
            raise RuntimeError(f"No video frames captured for {condition}")
        filename = f"{condition}.mp4"
        evidence = write_rollout_mp4(sink.frames, temporary / filename, encoding=encoding)
        videos[condition] = _video_payload(evidence, filename)
        outcomes[condition] = {
            "success": outcome.success,
            "steps": outcome.steps,
            "steering_calls": outcome.steering_calls,
            "trigger_source": outcome.trigger_source,
            "trigger_step": outcome.trigger_step,
            "release_step": outcome.release_step,
        }

    manifest = {
        "schema_version": 1,
        "artifact_kind": "disclosed_cherry_picked_rollout_pair",
        "selection_policy": (
            "Outcome-conditional visualization selected from the frozen matrix: the spatial "
            "stale-premise task, with unsteered failure and learned-monitor recovery. This is "
            "an illustrative pair, not a random sample or an additional evaluation."
        ),
        "task": TASK_NAME,
        "episode_index": EPISODE_INDEX,
        "noise_seed": NOISE_SEED,
        "simulator_seed": int(protocol["runtime"]["simulator_seed"]),
        "conditions": list(CONDITIONS),
        "source_report": REPORT_PATH,
        "source_report_sha256": sha256_file(report_path),
        "protocol": "configs/monitor_steering_confirmatory_v3.toml",
        "protocol_sha256": sha256_file(protocol_path),
        "frame_stride": frame_stride,
        "videos": videos,
        "outcomes": outcomes,
    }
    manifest_path = temporary / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.rename(output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--action-atlas-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rendered = render_disclosed_example(
        args.repository,
        checkpoint=args.checkpoint,
        action_atlas_root=args.action_atlas_root,
        state_dir=args.state_dir,
        output=args.output,
        device=args.device,
        frame_stride=args.frame_stride,
        fps=args.fps,
    )
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
