"""Frozen protocol and runner for the multi-noise confirmatory replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from causal_vla.confirmatory_monitor_runtime import (
    ConfirmatoryCondition,
    ConfirmatoryEpisode,
    run_confirmatory_monitor_evaluation,
    validate_confirmatory_conditions,
)
from causal_vla.monitored_knn_runtime import ReleaseMode

Device = Literal["cpu", "cuda", "mps"]
EXPECTED_SCENES = (10, 11, 12, 13, 14)
EXPECTED_CONDITIONS = frozenset(
    {
        "conflict",
        "always_steer",
        "oracle_monitor",
        "learned_monitor",
        "aligned_no_steer",
        "aligned_monitor",
        "aligned_forced_steer",
        "wrong_sign",
        "random",
    }
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ConfirmatoryAnalysisPlan:
    primary_contrast: str
    forced_trigger_contrast: str
    release_contrast: str
    cluster_unit: str
    bootstrap_replicates: int
    bootstrap_seed: int
    ci_level: float
    minimum_positive_tasks: int
    minimum_positive_noise_seeds: int
    minimum_learned_oracle_agreement: float
    require_positive_aggregate_effect: bool
    require_learned_above_controls: bool
    rollout_counts_are_descriptive: bool
    paper_update_requires_all_gates: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConfirmatoryAnalysisPlan:
        return cls(
            primary_contrast=str(payload["primary_contrast"]),
            forced_trigger_contrast=str(payload["forced_trigger_contrast"]),
            release_contrast=str(payload["release_contrast"]),
            cluster_unit=str(payload["cluster_unit"]),
            bootstrap_replicates=int(payload["bootstrap_replicates"]),
            bootstrap_seed=int(payload["bootstrap_seed"]),
            ci_level=float(payload["ci_level"]),
            minimum_positive_tasks=int(payload["minimum_positive_tasks"]),
            minimum_positive_noise_seeds=int(payload["minimum_positive_noise_seeds"]),
            minimum_learned_oracle_agreement=float(payload["minimum_learned_oracle_agreement"]),
            require_positive_aggregate_effect=bool(payload["require_positive_aggregate_effect"]),
            require_learned_above_controls=bool(payload["require_learned_above_controls"]),
            rollout_counts_are_descriptive=bool(payload["rollout_counts_are_descriptive"]),
            paper_update_requires_all_gates=bool(payload["paper_update_requires_all_gates"]),
        )

    def validate(self, *, task_count: int, noise_count: int) -> None:
        if self.primary_contrast != "learned_monitor_minus_conflict":
            raise ValueError("Confirmatory primary contrast is not recognized")
        if self.forced_trigger_contrast != "aligned_forced_steer_minus_aligned_no_steer":
            raise ValueError("Confirmatory forced-trigger contrast is not recognized")
        if self.release_contrast != "always_steer_minus_learned_monitor":
            raise ValueError("Confirmatory release contrast is not recognized")
        if self.cluster_unit != "task_x_scene":
            raise ValueError("Confirmatory cluster unit must be task_x_scene")
        if self.bootstrap_replicates < 10_000:
            raise ValueError("Confirmatory bootstrap requires at least 10,000 replicates")
        if not 0 < self.ci_level < 1:
            raise ValueError("Confirmatory confidence level must lie in (0, 1)")
        if not 1 <= self.minimum_positive_tasks <= task_count:
            raise ValueError("Positive-task gate is incompatible with the task count")
        if not 1 <= self.minimum_positive_noise_seeds <= noise_count:
            raise ValueError("Positive-noise gate is incompatible with the seed count")
        if not 0 <= self.minimum_learned_oracle_agreement <= 1:
            raise ValueError("Oracle-agreement gate must lie in [0, 1]")
        if not self.rollout_counts_are_descriptive:
            raise ValueError("Rollout-level counts must be declared descriptive")
        if not self.paper_update_requires_all_gates:
            raise ValueError("Paper update must require every predeclared gate")


@dataclass(frozen=True)
class ConfirmatoryTask:
    name: str
    suite: str
    task_id: int
    steering_bank: str
    steering_bank_sha256: str
    monitor_bank: str
    monitor_bank_sha256: str
    release_monitor_bank: str | None
    release_monitor_bank_sha256: str | None
    release_mode: ReleaseMode
    release_monitor_interval: int
    release_reactivation_patience: int
    grasp_close_fraction: float
    grasp_reopen_fraction: float
    grasp_lift_threshold: float
    grasp_release_patience: int
    monitor_arm_timeout: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConfirmatoryTask:
        release_bank = payload.get("release_monitor_bank")
        release_hash = payload.get("release_monitor_bank_sha256")
        return cls(
            name=str(payload["name"]),
            suite=str(payload["suite"]),
            task_id=int(payload["task_id"]),
            steering_bank=str(payload["steering_bank"]),
            steering_bank_sha256=str(payload["steering_bank_sha256"]),
            monitor_bank=str(payload["monitor_bank"]),
            monitor_bank_sha256=str(payload["monitor_bank_sha256"]),
            release_monitor_bank=(str(release_bank) if release_bank is not None else None),
            release_monitor_bank_sha256=(str(release_hash) if release_hash is not None else None),
            release_mode=cast(ReleaseMode, str(payload["release_mode"])),
            release_monitor_interval=int(payload["release_monitor_interval"]),
            release_reactivation_patience=int(payload["release_reactivation_patience"]),
            grasp_close_fraction=float(payload["grasp_close_fraction"]),
            grasp_reopen_fraction=float(payload["grasp_reopen_fraction"]),
            grasp_lift_threshold=float(payload["grasp_lift_threshold"]),
            grasp_release_patience=int(payload["grasp_release_patience"]),
            monitor_arm_timeout=int(payload["monitor_arm_timeout"]),
        )

    def expected_release_bank(self) -> str:
        return self.release_monitor_bank or self.monitor_bank


@dataclass(frozen=True)
class ConfirmatoryProtocol:
    schema_version: int
    protocol_id: str
    status: str
    frozen_at_utc: str
    checkpoint_revision: str
    checkpoint_config_sha256: str
    source_sha256: dict[str, str]
    heldout_episodes: tuple[int, ...]
    noise_seeds: tuple[int, ...]
    prior_noise_seeds: tuple[int, ...]
    noise_seed_selection_rule: str
    noise_schedule_id: str
    conditions: tuple[ConfirmatoryCondition, ...]
    simulator_seed: int
    random_seed: int
    max_steps: int
    analysis: ConfirmatoryAnalysisPlan
    tasks: tuple[ConfirmatoryTask, ...]

    @classmethod
    def load(cls, path: str | Path) -> ConfirmatoryProtocol:
        with Path(path).open("rb") as handle:
            payload = tomllib.load(handle)
        checkpoint = payload["checkpoint"]
        split = payload["split"]
        runtime = payload["runtime"]
        protocol = cls(
            schema_version=int(payload["schema_version"]),
            protocol_id=str(payload["protocol_id"]),
            status=str(payload["status"]),
            frozen_at_utc=str(payload["frozen_at_utc"]),
            checkpoint_revision=str(checkpoint["revision"]),
            checkpoint_config_sha256=str(checkpoint["config_sha256"]),
            source_sha256={
                str(name): str(value) for name, value in payload["source_sha256"].items()
            },
            heldout_episodes=tuple(int(value) for value in split["heldout_episodes"]),
            noise_seeds=tuple(int(value) for value in split["noise_seeds"]),
            prior_noise_seeds=tuple(int(value) for value in split["prior_noise_seeds"]),
            noise_seed_selection_rule=str(split["noise_seed_selection_rule"]),
            noise_schedule_id=str(split["noise_schedule_id"]),
            conditions=validate_confirmatory_conditions(tuple(split["conditions"])),
            simulator_seed=int(runtime["simulator_seed"]),
            random_seed=int(runtime["random_seed"]),
            max_steps=int(runtime["max_steps"]),
            analysis=ConfirmatoryAnalysisPlan.from_dict(payload["analysis"]),
            tasks=tuple(ConfirmatoryTask.from_dict(item) for item in payload["tasks"]),
        )
        protocol.validate()
        return protocol

    def validate(self) -> None:
        if self.schema_version != 2 or self.status != "frozen":
            raise ValueError("Confirmatory protocol must be frozen schema version 2")
        if not self.protocol_id or not self.frozen_at_utc:
            raise ValueError("Confirmatory protocol metadata must be nonempty")
        if not self.source_sha256 or any(
            len(digest) != 64 for digest in self.source_sha256.values()
        ):
            raise ValueError("Confirmatory source files require SHA-256 hashes")
        if self.heldout_episodes != EXPECTED_SCENES:
            raise ValueError("Confirmatory scenes must be the sealed episodes 10--14")
        if len(self.noise_seeds) < 3 or len(set(self.noise_seeds)) != len(self.noise_seeds):
            raise ValueError("Confirmatory protocol requires at least three unique noise seeds")
        if set(self.noise_seeds) & set(self.prior_noise_seeds):
            raise ValueError("Confirmatory noise seeds must be new")
        if self.noise_seed_selection_rule != "357_plus_100k_for_k_1_through_3":
            raise ValueError("Confirmatory noise seeds require the predeclared arithmetic rule")
        if self.noise_seeds != (457, 557, 657):
            raise ValueError("Confirmatory noise seeds do not match the selection rule")
        if self.noise_schedule_id != "base_plus_episode_10000_plus_step":
            raise ValueError("Confirmatory noise schedule is not recognized")
        if set(self.conditions) != EXPECTED_CONDITIONS:
            raise ValueError("Confirmatory condition matrix is incomplete")
        if len(self.tasks) != 3 or len({task.suite for task in self.tasks}) != 2:
            raise ValueError("Confirmatory protocol requires three tasks and two suites")
        if len({task.name for task in self.tasks}) != len(self.tasks):
            raise ValueError("Confirmatory task names must be unique")
        if len({(task.suite, task.task_id) for task in self.tasks}) != len(self.tasks):
            raise ValueError("Confirmatory suite/task pairs must be unique")
        if self.max_steps <= 0:
            raise ValueError("Confirmatory horizon must be positive")
        self.analysis.validate(
            task_count=len(self.tasks),
            noise_count=len(self.noise_seeds),
        )
        for task in self.tasks:
            if task.suite not in {"libero_object", "libero_spatial"}:
                raise ValueError(f"Unsupported confirmatory suite {task.suite!r}")
            if task.release_monitor_interval <= 0 or task.grasp_release_patience <= 0:
                raise ValueError("Release intervals and patience must be positive")
            hashes = (task.steering_bank_sha256, task.monitor_bank_sha256)
            if any(len(digest) != 64 for digest in hashes):
                raise ValueError("Controller hashes must be SHA-256 digests")
            if (task.release_monitor_bank is None) != (task.release_monitor_bank_sha256 is None):
                raise ValueError("Release-monitor path and hash must be specified together")

    def task(self, name: str) -> ConfirmatoryTask:
        for task in self.tasks:
            if task.name == name:
                return task
        raise ValueError(f"Task {name!r} is not in the confirmatory protocol")

    def verify_sources(self, repository: str | Path) -> None:
        _verify_hashes(Path(repository).resolve(), self.source_sha256, label="source")

    def verify_checkpoint(self, checkpoint: str | Path) -> Path:
        path = Path(checkpoint).resolve()
        if path.name != self.checkpoint_revision:
            raise RuntimeError("Checkpoint revision does not match confirmatory protocol")
        if sha256_file(path / "config.json") != self.checkpoint_config_sha256:
            raise RuntimeError("Checkpoint config does not match confirmatory protocol")
        return path

    def verify_task_assets(
        self,
        repository: str | Path,
        task: ConfirmatoryTask,
    ) -> tuple[Path, Path, Path | None]:
        root = Path(repository).resolve()
        expected = {
            task.steering_bank: task.steering_bank_sha256,
            task.monitor_bank: task.monitor_bank_sha256,
        }
        if task.release_monitor_bank is not None:
            assert task.release_monitor_bank_sha256 is not None
            expected[task.release_monitor_bank] = task.release_monitor_bank_sha256
        _verify_hashes(root, expected, label="controller")
        return (
            root / task.steering_bank,
            root / task.monitor_bank,
            root / task.release_monitor_bank if task.release_monitor_bank else None,
        )


def _verify_hashes(root: Path, expected: dict[str, str], *, label: str) -> None:
    mismatches: list[str] = []
    for relative, digest in expected.items():
        path = root / relative
        actual = sha256_file(path) if path.is_file() else "missing"
        if actual != digest:
            mismatches.append(f"{relative}: expected {digest}, observed {actual}")
    if mismatches:
        raise RuntimeError(f"Frozen confirmatory {label} mismatch:\n" + "\n".join(mismatches))


def _episode_semantics(
    protocol: ConfirmatoryProtocol,
    report: dict[str, Any],
) -> None:
    aligned_prompt = str(report["aligned_prompt"])
    conflict_prompt = str(report["conflict_prompt"])
    spatial = report["suite"] == "libero_spatial"
    for raw in report["episodes"]:
        row = cast(dict[str, Any], raw)
        condition = str(row["condition"])
        aligned = condition.startswith("aligned_")
        expected_prompt = aligned_prompt if aligned else conflict_prompt
        if row.get("prompt") != expected_prompt:
            raise ValueError(f"Condition {condition!r} used the wrong action prompt")
        if spatial:
            expected_scene = "aligned" if aligned else "conflict"
            if row.get("scene_condition") != expected_scene:
                raise ValueError(f"Condition {condition!r} used the wrong spatial scene")
        calls = int(row["steering_calls"])
        steps = int(row["steps"])
        trigger_source = row.get("trigger_source")
        monitor_checks = int(row["monitor_checks"])
        if (
            condition in {"conflict", "aligned_no_steer", "aligned_monitor"}
            and calls != 0
            and (condition != "aligned_monitor" or row.get("trigger_step") is None)
        ):
            raise ValueError(f"Condition {condition!r} has impossible steering calls")
        if condition == "always_steer" and calls != steps:
            raise ValueError("Always-steer condition did not steer every action call")
        if condition in {"conflict", "always_steer", "aligned_no_steer"} and (
            trigger_source != "none" or monitor_checks != 0
        ):
            raise ValueError(f"Condition {condition!r} unexpectedly used the monitor gate")
        if condition == "oracle_monitor" and (
            trigger_source != "oracle" or row.get("trigger_step") is None
        ):
            raise ValueError("Oracle-monitor condition did not record an oracle trigger")
        if condition in {"learned_monitor", "wrong_sign", "random", "aligned_monitor"}:
            allowed_sources = {"none", "learned"}
            if trigger_source not in allowed_sources or monitor_checks <= 0:
                raise ValueError(f"Condition {condition!r} has invalid learned-monitor provenance")
        if condition == "aligned_forced_steer":
            if (
                trigger_source != "forced"
                or row.get("trigger_step") != 0
                or monitor_checks != 0
                or calls <= 0
            ):
                raise ValueError("Aligned forced-steer semantics are not satisfied")
        elif trigger_source == "forced":
            raise ValueError("Only aligned_forced_steer may use a forced trigger")
    if report.get("noise_schedule_id") != protocol.noise_schedule_id:
        raise ValueError("Report uses the wrong paired-noise schedule")


def verify_confirmatory_report(
    protocol: ConfirmatoryProtocol,
    task: ConfirmatoryTask,
    report: dict[str, Any],
    *,
    repository: str | Path,
    checkpoint: str | Path,
    noise_seed: int,
) -> None:
    root = Path(repository).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    expected_release = root / task.expected_release_bank()
    expected: dict[str, object] = {
        "schema_version": 2,
        "checkpoint": str(checkpoint_path),
        "steering_bank_path": str(root / task.steering_bank),
        "monitor_bank_path": str(root / task.monitor_bank),
        "release_monitor_bank_path": str(expected_release),
        "suite": task.suite,
        "task_id": task.task_id,
        "episode_indices": list(protocol.heldout_episodes),
        "conditions": list(protocol.conditions),
        "noise_seed": noise_seed,
        "noise_schedule_id": protocol.noise_schedule_id,
        "simulator_seed": protocol.simulator_seed,
        "random_seed": protocol.random_seed,
        "max_steps": protocol.max_steps,
        "release_mode": task.release_mode,
        "release_monitor_interval": task.release_monitor_interval,
        "release_reactivation_patience": task.release_reactivation_patience,
        "grasp_close_fraction": task.grasp_close_fraction,
        "grasp_reopen_fraction": task.grasp_reopen_fraction,
        "grasp_lift_threshold": task.grasp_lift_threshold,
        "grasp_release_patience": task.grasp_release_patience,
        "monitor_arm_timeout": task.monitor_arm_timeout,
    }
    for field, value in expected.items():
        observed = report.get(field)
        if isinstance(value, list) and isinstance(observed, tuple):
            observed = list(observed)
        if observed != value:
            raise ValueError(f"Confirmatory report disagrees on {field!r}")
    for counter in (
        "evaluation_target_activations_used",
        "runtime_counterfactual_prompt_forwards",
        "runtime_prompt_rewrites",
        "language_token_interventions",
    ):
        if int(report.get(counter, -1)) != 0:
            raise ValueError(f"Source-free counter {counter!r} is nonzero")
    observed_cells: set[tuple[int, str]] = set()
    for raw in report.get("episodes", []):
        row = cast(dict[str, Any], raw)
        cell = (int(row["episode_index"]), str(row["condition"]))
        if cell in observed_cells:
            raise ValueError(f"Duplicate confirmatory cell {cell}")
        observed_cells.add(cell)
    expected_cells = {
        (episode, condition)
        for episode in protocol.heldout_episodes
        for condition in protocol.conditions
    }
    if observed_cells != expected_cells:
        raise ValueError("Confirmatory report does not cover the complete condition matrix")
    _episode_semantics(protocol, report)


def run_confirmatory_task(
    protocol_path: str | Path,
    *,
    repository: str | Path,
    checkpoint: str | Path,
    task_name: str,
    noise_seed: int,
    output: str | Path,
    device: Device,
) -> dict[str, Any]:
    protocol = ConfirmatoryProtocol.load(protocol_path)
    if noise_seed not in protocol.noise_seeds:
        raise ValueError(f"Noise seed {noise_seed} is not frozen")
    task = protocol.task(task_name)
    root = Path(repository).resolve()
    protocol.verify_sources(root)
    checkpoint_path = protocol.verify_checkpoint(checkpoint)
    steering_bank, monitor_bank, release_bank = protocol.verify_task_assets(root, task)
    output_path = Path(output).resolve()
    if output_path.exists():
        loaded = json.loads(output_path.read_text())
        if not isinstance(loaded, dict):
            raise ValueError("Existing confirmatory report is not a JSON object")
        payload = cast(dict[str, Any], loaded)
        verify_confirmatory_report(
            protocol,
            task,
            payload,
            repository=root,
            checkpoint=checkpoint_path,
            noise_seed=noise_seed,
        )
        return payload
    report = run_confirmatory_monitor_evaluation(
        checkpoint_path,
        steering_bank,
        monitor_bank,
        release_monitor_bank_path=release_bank,
        task_id=task.task_id,
        episode_indices=protocol.heldout_episodes,
        conditions=protocol.conditions,
        suite=task.suite,
        device=device,
        noise_seed=noise_seed,
        simulator_seed=protocol.simulator_seed,
        random_seed=protocol.random_seed,
        max_steps=protocol.max_steps,
        grasp_close_fraction=task.grasp_close_fraction,
        grasp_reopen_fraction=task.grasp_reopen_fraction,
        grasp_lift_threshold=task.grasp_lift_threshold,
        grasp_release_patience=task.grasp_release_patience,
        release_reactivation_patience=task.release_reactivation_patience,
        release_mode=task.release_mode,
        release_monitor_interval=task.release_monitor_interval,
        monitor_arm_timeout=task.monitor_arm_timeout,
        progress=lambda outcome: _print_progress(task, noise_seed, outcome),
    )
    payload = cast(dict[str, Any], report.to_dict())
    verify_confirmatory_report(
        protocol,
        task,
        payload,
        repository=root,
        checkpoint=checkpoint_path,
        noise_seed=noise_seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_path)
    return payload


def _print_progress(
    task: ConfirmatoryTask,
    noise_seed: int,
    outcome: ConfirmatoryEpisode,
) -> None:
    print(
        f"[confirmatory] task={task.name} noise={noise_seed} "
        f"episode={outcome.episode_index} condition={outcome.condition} "
        f"success={outcome.success} steps={outcome.steps}",
        file=sys.stderr,
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m causal_vla.confirmatory_monitor")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--noise-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_confirmatory_task(
        args.protocol,
        repository=args.repository,
        checkpoint=args.checkpoint,
        task_name=args.task,
        noise_seed=args.noise_seed,
        output=args.output,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
