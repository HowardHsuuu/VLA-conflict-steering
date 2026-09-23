"""Frozen-VLM monitoring for visually falsifiable source-location claims."""

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

from causal_vla.evidence import vlm_candidate_evidence
from causal_vla.integrations.action_atlas import activate_action_atlas
from causal_vla.monitor_calibration import (
    LabeledMonitorExample,
    MonitorCalibration,
    calibrate_conflict_monitor,
    evaluate_monitor_operating_point,
)
from causal_vla.routing import EvidenceDistribution

_PROPOSITION = "object_at_claimed_source"
_CANDIDATES = {"yes": "yes", "no": "no"}


@dataclass(frozen=True)
class SourceClaimSpec:
    """A unique-object goal task with true and false source claims."""

    task_id: int
    object_name: str
    false_source_label: str
    false_source: str
    destination_label: str
    correct_prompt: str
    conflict_prompt: str

    @property
    def aligned_prompt(self) -> str:
        return (
            f"pick up the {self.object_name} on the table and place it "
            f"on the {self.destination_label}"
        )

    @property
    def location_question(self) -> str:
        return f"Where is the {self.object_name} right now?"

    @property
    def aligned_question(self) -> str:
        return f"Is the {self.object_name} on the table right now? Answer yes or no."

    @property
    def conflict_question(self) -> str:
        return f"Is the {self.object_name} {self.false_source} right now? Answer yes or no."

    @property
    def conflict_type(self) -> str:
        return f"false_source:{self.object_name}:{self.false_source}"


def source_claim_specs() -> tuple[SourceClaimSpec, ...]:
    """Return visually falsifiable LIBERO Goal source corruptions."""

    return (
        SourceClaimSpec(
            1,
            "black bowl",
            "cabinet",
            "on top of the wooden cabinet",
            "stove",
            "put the bowl on the stove",
            "pick up the black bowl on top of the wooden cabinet and place it on the stove",
        ),
        SourceClaimSpec(
            2,
            "wine bottle",
            "rack",
            "on the wine rack",
            "cabinet",
            "put the wine bottle on top of the cabinet",
            "pick up the wine bottle on the wine rack and place it on top of the cabinet",
        ),
        SourceClaimSpec(
            4,
            "black bowl",
            "stove",
            "on the stove",
            "cabinet",
            "put the bowl on top of the cabinet",
            "pick up the black bowl on the stove and place it on top of the cabinet",
        ),
        SourceClaimSpec(
            9,
            "wine bottle",
            "cabinet",
            "on top of the wooden cabinet",
            "rack",
            "put the wine bottle on the rack",
            "pick up the wine bottle on top of the wooden cabinet and place it on the rack",
        ),
    )


def affirmative_language(labels: tuple[str, str] = ("yes", "no")) -> EvidenceDistribution:
    """Represent the explicit source claim made by a corrupted instruction."""

    return EvidenceDistribution("language", _PROPOSITION, labels, (0.98, 0.02))


@dataclass(frozen=True)
class ClaimMonitorReport:
    schema_version: int
    checkpoint: str
    suite: str
    task_ids: tuple[int, ...]
    episode_indices: tuple[int, ...]
    simulator_seed: int
    threshold_source: str
    examples: tuple[LabeledMonitorExample, ...]
    calibration: MonitorCalibration

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                key: value
                for key, value in asdict(self).items()
                if key not in {"examples", "calibration"}
            },
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


def _load_thresholds(examples: tuple[LabeledMonitorExample, ...], path: Path) -> MonitorCalibration:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("calibration", {}).get("selected")
    if not isinstance(selected, dict):
        raise ValueError("Threshold report has no selected operating point")
    point = evaluate_monitor_operating_point(
        examples,
        divergence_threshold=float(selected["divergence_threshold"]),
        confidence_floor=float(selected["confidence_floor"]),
    )
    return MonitorCalibration(0.0, point, (point,))


def run_claim_monitor_scan(
    checkpoint: str | Path,
    *,
    task_ids: tuple[int, ...] = (1, 2, 4, 9),
    episode_indices: tuple[int, ...] = tuple(range(10)),
    suite: str = "libero_goal",
    resolution: int = 256,
    action_atlas_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    device: str = "mps",
    temperature: float = 1.0,
    simulator_seed: int = 42,
    thresholds_from: str | Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ClaimMonitorReport:
    """Score true/false source claims and select thresholds on development only."""

    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("Task IDs must be nonempty and unique")
    if not episode_indices or len(set(episode_indices)) != len(episode_indices):
        raise ValueError("Episode indices must be nonempty and unique")
    if set(episode_indices) & set(range(10, 15)):
        raise ValueError("Historical held-out episodes 10--14 are sealed")
    specs_by_task = {spec.task_id: spec for spec in source_claim_specs()}
    try:
        specs = tuple(specs_by_task[task_id] for task_id in task_ids)
    except KeyError as error:
        raise ValueError(f"No source-claim spec for task {error.args[0]}") from error
    activate_action_atlas(action_atlas_root, state_dir)
    adapter = importlib.import_module("experiments.model_adapters").SmolVLAAdapter()
    checkpoint_path = str(Path(checkpoint).resolve())
    policy = adapter.load_model(checkpoint_path, device=device)
    wrapper = policy.model.vlm_with_expert
    model, processor = wrapper.vlm, wrapper.processor
    examples: list[LabeledMonitorExample] = []
    language = affirmative_language()
    for spec in specs:
        for episode_index in episode_indices:
            environment, native_prompt, _ = adapter.create_env(
                spec.task_id,
                suite=suite,
                resolution=resolution,
                episode_index=episode_index,
            )
            try:
                if native_prompt.casefold() != spec.correct_prompt.casefold():
                    raise ValueError("Native instruction differs from source-claim spec")
                observation, _ = environment.reset(seed=simulator_seed + episode_index)
                raw_image = np.asarray(observation["pixels"]["image"])
                image = np.ascontiguousarray(raw_image[::-1, ::-1])
                for expected, question, true_label in (
                    ("aligned", spec.aligned_question, "yes"),
                    ("conflict", spec.conflict_question, "no"),
                ):
                    vision = vlm_candidate_evidence(
                        model,
                        processor,
                        image,
                        question=question,
                        proposition=_PROPOSITION,
                        candidates=_CANDIDATES,
                        device=device,
                        temperature=temperature,
                    )
                    examples.append(
                        LabeledMonitorExample(
                            task_id=spec.task_id,
                            episode_index=episode_index,
                            conflict_type=spec.conflict_type,
                            expected_status=cast(Any, expected),
                            true_visual_label=true_label,
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
            divergence_candidates=(0.05, 0.10, 0.15, 0.20, 0.25),
            confidence_candidates=(0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80),
            max_aligned_false_trigger_rate=0.0,
        )
        threshold_source = "selected_on_this_development_scan"
    else:
        path = Path(thresholds_from).resolve()
        calibration = _load_thresholds(frozen_examples, path)
        threshold_source = str(path)
    return ClaimMonitorReport(
        schema_version=1,
        checkpoint=checkpoint_path,
        suite=suite,
        task_ids=task_ids,
        episode_indices=episode_indices,
        simulator_seed=simulator_seed,
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
    parser = argparse.ArgumentParser(prog="python -m causal_vla.claim_monitor_runtime")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tasks", type=_integers, default=(1, 2, 4, 9))
    parser.add_argument("--episodes", type=_integers, default=tuple(range(10)))
    parser.add_argument("--thresholds-from", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_claim_monitor_scan(
        args.checkpoint,
        task_ids=args.tasks,
        episode_indices=args.episodes,
        thresholds_from=args.thresholds_from,
        device=args.device,
        progress=lambda task, episode: print(
            f"[claim-monitor] task={task} episode={episode}", file=sys.stderr, flush=True
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
