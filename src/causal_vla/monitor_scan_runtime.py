"""Multi-task development scans for the frozen-VLM conflict monitor."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from causal_vla.evidence import language_label_evidence, vlm_candidate_evidence
from causal_vla.integrations.action_atlas import activate_action_atlas
from causal_vla.monitor_calibration import (
    LabeledMonitorExample,
    MonitorCalibration,
    calibrate_conflict_monitor,
    evaluate_monitor_operating_point,
)

_QUESTION = "Where is the black bowl right now? Answer using exactly one candidate."
_PROPOSITION = "initial_black_bowl_support"


@dataclass(frozen=True)
class SupportConflictSpec:
    """One factual source-location substitution in LIBERO Spatial."""

    task_id: int
    true_label: str
    distractor_label: str
    correct_prompt: str
    conflict_prompt: str

    @property
    def conflict_type(self) -> str:
        return f"support:{self.distractor_label}->{self.true_label}"


def support_conflict_specs() -> tuple[SupportConflictSpec, ...]:
    """Return development task pairs without any held-out episode selection."""

    return (
        SupportConflictSpec(
            3,
            "cookie_box",
            "cabinet",
            "pick up the black bowl on the cookie box and place it on the plate",
            "pick up the black bowl on the wooden cabinet and place it on the plate",
        ),
        SupportConflictSpec(
            5,
            "ramekin",
            "cookie_box",
            "pick up the black bowl on the ramekin and place it on the plate",
            "pick up the black bowl on the cookie box and place it on the plate",
        ),
        SupportConflictSpec(
            7,
            "stove",
            "cabinet",
            "pick up the black bowl on the stove and place it on the plate",
            "pick up the black bowl on the wooden cabinet and place it on the plate",
        ),
        SupportConflictSpec(
            9,
            "cabinet",
            "stove",
            "pick up the black bowl on the wooden cabinet and place it on the plate",
            "pick up the black bowl on the stove and place it on the plate",
        ),
    )


def _answer(label: str) -> str:
    rendered = {
        "cabinet": "on the wooden cabinet",
        "cookie_box": "on the cookie box",
        "ramekin": "on the ramekin",
        "stove": "on the stove",
    }
    try:
        return rendered[label]
    except KeyError as error:
        raise ValueError(f"Unsupported support label {label!r}") from error


def _aliases(labels: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    aliases: dict[str, tuple[str, ...]] = {
        label: (label.replace("_", " "),) for label in labels
    }
    if "cabinet" in labels:
        aliases["cabinet"] = ("wooden cabinet",)
    return aliases


@dataclass(frozen=True)
class MonitorScanReport:
    """Raw frozen evidence plus either selected or externally fixed thresholds."""

    schema_version: int
    checkpoint: str
    suite: str
    task_ids: tuple[int, ...]
    episode_indices: tuple[int, ...]
    simulator_seed: int
    question: str
    proposition: str
    threshold_source: str
    examples: tuple[LabeledMonitorExample, ...]
    calibration: MonitorCalibration

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "checkpoint": self.checkpoint,
            "suite": self.suite,
            "task_ids": list(self.task_ids),
            "episode_indices": list(self.episode_indices),
            "simulator_seed": self.simulator_seed,
            "question": self.question,
            "proposition": self.proposition,
            "threshold_source": self.threshold_source,
            "examples": [
                {
                    **{
                        key: value
                        for key, value in asdict(example).items()
                        if key not in {"vision", "language"}
                    },
                    "vision": asdict(example.vision),
                    "language": asdict(example.language),
                }
                for example in self.examples
            ],
            "calibration": self.calibration.to_dict(),
        }


def _fixed_threshold_calibration(
    examples: tuple[LabeledMonitorExample, ...], threshold_path: Path
) -> MonitorCalibration:
    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    selected = payload.get("calibration", {}).get("selected", {})
    if not isinstance(selected, dict):
        raise ValueError("Threshold report has no selected calibration point")
    divergence = float(selected["divergence_threshold"])
    confidence = float(selected["confidence_floor"])
    point = evaluate_monitor_operating_point(
        examples,
        divergence_threshold=divergence,
        confidence_floor=confidence,
    )
    return MonitorCalibration(
        max_aligned_false_trigger_rate=float(
            payload.get("calibration", {}).get("max_aligned_false_trigger_rate", 0.0)
        ),
        selected=point,
        operating_points=(point,),
    )


def run_support_monitor_scan(
    checkpoint: str | Path,
    *,
    task_ids: tuple[int, ...] = (3, 5, 7, 9),
    episode_indices: tuple[int, ...] = tuple(range(10)),
    suite: str = "libero_spatial",
    resolution: int = 256,
    action_atlas_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    device: str = "mps",
    temperature: float = 1.0,
    simulator_seed: int = 42,
    max_aligned_false_trigger_rate: float = 0.0,
    thresholds_from: str | Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> MonitorScanReport:
    """Collect pairwise evidence across tasks and calibrate without rollout outcomes."""

    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("Task IDs must be nonempty and unique")
    if not episode_indices or len(set(episode_indices)) != len(episode_indices):
        raise ValueError("Episode indices must be nonempty and unique")
    if set(episode_indices) & set(range(10, 15)):
        raise ValueError("Historical held-out episodes 10--14 are sealed")
    specs_by_task = {spec.task_id: spec for spec in support_conflict_specs()}
    try:
        specs = tuple(specs_by_task[task_id] for task_id in task_ids)
    except KeyError as error:
        raise ValueError(f"No support-conflict spec for task {error.args[0]}") from error

    activate_action_atlas(action_atlas_root, state_dir)
    adapter = importlib.import_module("experiments.model_adapters").SmolVLAAdapter()
    checkpoint_path = str(Path(checkpoint).resolve())
    policy = adapter.load_model(checkpoint_path, device=device)
    wrapper = policy.model.vlm_with_expert
    model, processor = wrapper.vlm, wrapper.processor
    examples: list[LabeledMonitorExample] = []
    for spec in specs:
        labels = (spec.true_label, spec.distractor_label)
        candidates = {label: _answer(label) for label in labels}
        aliases = _aliases(labels)
        for episode_index in episode_indices:
            environment, native_prompt, _ = adapter.create_env(
                spec.task_id,
                suite=suite,
                resolution=resolution,
                episode_index=episode_index,
            )
            try:
                if native_prompt.casefold() != spec.correct_prompt.casefold():
                    raise ValueError(
                        f"Task {spec.task_id} native instruction differs from monitor spec"
                    )
                observation, _ = environment.reset(seed=simulator_seed + episode_index)
                raw_image = np.asarray(observation["pixels"]["image"])
                image = np.ascontiguousarray(raw_image[::-1, ::-1])
                vision = vlm_candidate_evidence(
                    model,
                    processor,
                    image,
                    question=_QUESTION,
                    proposition=_PROPOSITION,
                    candidates=candidates,
                    device=device,
                    temperature=temperature,
                )
                for expected_status, prompt in (
                    ("aligned", spec.correct_prompt),
                    ("conflict", spec.conflict_prompt),
                ):
                    language = language_label_evidence(
                        prompt,
                        proposition=_PROPOSITION,
                        labels=labels,
                        aliases=aliases,
                    )
                    examples.append(
                        LabeledMonitorExample(
                            task_id=spec.task_id,
                            episode_index=episode_index,
                            conflict_type=spec.conflict_type,
                            expected_status=cast(Any, expected_status),
                            true_visual_label=spec.true_label,
                            vision=vision,
                            language=language,
                        )
                    )
            finally:
                environment.close()
            if progress is not None:
                progress(spec.task_id, episode_index)
    frozen_examples = tuple(examples)
    if thresholds_from is None:
        calibration = calibrate_conflict_monitor(
            frozen_examples,
            max_aligned_false_trigger_rate=max_aligned_false_trigger_rate,
        )
        threshold_source = "selected_on_this_development_scan"
    else:
        threshold_path = Path(thresholds_from).resolve()
        calibration = _fixed_threshold_calibration(frozen_examples, threshold_path)
        threshold_source = str(threshold_path)
    return MonitorScanReport(
        schema_version=1,
        checkpoint=checkpoint_path,
        suite=suite,
        task_ids=task_ids,
        episode_indices=episode_indices,
        simulator_seed=simulator_seed,
        question=_QUESTION,
        proposition=_PROPOSITION,
        threshold_source=threshold_source,
        examples=frozen_examples,
        calibration=calibration,
    )


def _integers(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("Expected comma-separated integers")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m causal_vla.monitor_scan_runtime")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tasks", type=_integers, default=(3, 5, 7, 9))
    parser.add_argument("--episodes", type=_integers, default=tuple(range(10)))
    parser.add_argument("--thresholds-from", type=Path)
    parser.add_argument("--max-aligned-false-trigger-rate", type=float, default=0.0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_support_monitor_scan(
        args.checkpoint,
        task_ids=args.tasks,
        episode_indices=args.episodes,
        device=args.device,
        thresholds_from=args.thresholds_from,
        max_aligned_false_trigger_rate=args.max_aligned_false_trigger_rate,
        progress=lambda task, episode: print(
            f"[monitor-scan] task={task} episode={episode}", file=sys.stderr, flush=True
        ),
    )
    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
