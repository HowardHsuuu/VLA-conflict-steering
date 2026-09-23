"""Predeclared, cluster-aware analysis for the confirmatory replication."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from causal_vla.confirmatory_monitor import (
    ConfirmatoryProtocol,
    ConfirmatoryTask,
    sha256_file,
    verify_confirmatory_report,
)

ClusterKey = tuple[str, int]
CellKey = tuple[str, int, int]


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _task_for_report(
    protocol: ConfirmatoryProtocol,
    report: dict[str, Any],
) -> ConfirmatoryTask:
    matches = tuple(
        task
        for task in protocol.tasks
        if task.suite == report.get("suite") and task.task_id == report.get("task_id")
    )
    if len(matches) != 1:
        raise ValueError("Report does not identify exactly one confirmatory task")
    return matches[0]


def _row_index(
    loaded: list[tuple[Path, ConfirmatoryTask, int, dict[str, Any]]],
) -> dict[tuple[str, int, int, str], dict[str, Any]]:
    rows: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for _, task, noise_seed, report in loaded:
        for raw in report["episodes"]:
            row = cast(dict[str, Any], raw)
            key = (
                task.name,
                int(row["episode_index"]),
                noise_seed,
                str(row["condition"]),
            )
            if key in rows:
                raise ValueError(f"Duplicate confirmatory outcome {key}")
            rows[key] = row
    return rows


def _success(row: dict[str, Any]) -> int:
    return int(bool(row["success"]))


def _numeric(payload: dict[str, object], field: str) -> float:
    value = payload[field]
    if not isinstance(value, (int, float)):
        raise TypeError(f"Analysis field {field!r} is not numeric")
    return float(value)


def _paired_counts(differences: list[int]) -> dict[str, int]:
    return {
        "wins": sum(value > 0 for value in differences),
        "losses": sum(value < 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
    }


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot take a percentile of an empty collection")
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def stratified_cluster_bootstrap_ci(
    cluster_values: dict[ClusterKey, float],
    *,
    task_names: tuple[str, ...],
    replicates: int,
    seed: int,
    ci_level: float,
) -> tuple[float, float]:
    """Bootstrap task-scene clusters while preserving equal task representation."""

    by_task = {
        task: [value for (name, _), value in cluster_values.items() if name == task]
        for task in task_names
    }
    if any(not values for values in by_task.values()):
        raise ValueError("Every task needs at least one scene cluster")
    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sample: list[float] = []
        for task in task_names:
            values = by_task[task]
            sample.extend(generator.choice(values) for _ in values)
        estimates.append(sum(sample) / len(sample))
    estimates.sort()
    tail = (1.0 - ci_level) / 2.0
    return _percentile(estimates, tail), _percentile(estimates, 1.0 - tail)


def _contrast_summary(
    rows: dict[tuple[str, int, int, str], dict[str, Any]],
    *,
    protocol: ConfirmatoryProtocol,
    treatment: str,
    baseline: str,
) -> dict[str, object]:
    cell_differences: dict[CellKey, int] = {}
    cluster_values: dict[ClusterKey, float] = {}
    for task in protocol.tasks:
        for episode in protocol.heldout_episodes:
            differences = []
            for seed in protocol.noise_seeds:
                treatment_row = rows[(task.name, episode, seed, treatment)]
                baseline_row = rows[(task.name, episode, seed, baseline)]
                difference = _success(treatment_row) - _success(baseline_row)
                cell_differences[(task.name, episode, seed)] = difference
                differences.append(difference)
            cluster_values[(task.name, episode)] = sum(differences) / len(differences)
    flat = list(cell_differences.values())
    estimate = sum(cluster_values.values()) / len(cluster_values)
    lower, upper = stratified_cluster_bootstrap_ci(
        cluster_values,
        task_names=tuple(task.name for task in protocol.tasks),
        replicates=protocol.analysis.bootstrap_replicates,
        seed=protocol.analysis.bootstrap_seed,
        ci_level=protocol.analysis.ci_level,
    )
    return {
        "treatment": treatment,
        "baseline": baseline,
        "rollout_paired_counts": _paired_counts(flat),
        "cluster_unit": protocol.analysis.cluster_unit,
        "cluster_count": len(cluster_values),
        "cluster_mean_success_difference": estimate,
        "cluster_bootstrap_ci": [lower, upper],
        "cluster_bootstrap_replicates": protocol.analysis.bootstrap_replicates,
        "cluster_bootstrap_seed": protocol.analysis.bootstrap_seed,
        "ci_level": protocol.analysis.ci_level,
    }


def _group_summary(
    rows: dict[tuple[str, int, int, str], dict[str, Any]],
    *,
    protocol: ConfirmatoryProtocol,
    group: str,
) -> list[dict[str, object]]:
    grouped: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for task in protocol.tasks:
        for episode in protocol.heldout_episodes:
            for seed in protocol.noise_seeds:
                label = task.name if group == "task" else str(seed)
                grouped[label].append((task.name, episode, seed))
    summaries: list[dict[str, object]] = []
    for label, cells in grouped.items():
        condition_successes = {
            condition: sum(
                _success(rows[(task, episode, seed, condition)]) for task, episode, seed in cells
            )
            for condition in protocol.conditions
        }
        differences = [
            _success(rows[(task, episode, seed, "learned_monitor")])
            - _success(rows[(task, episode, seed, "conflict")])
            for task, episode, seed in cells
        ]
        summaries.append(
            {
                "name" if group == "task" else "noise_seed": (
                    label if group == "task" else int(label)
                ),
                "trials_per_condition": len(cells),
                "condition_successes": condition_successes,
                "learned_minus_conflict": sum(differences) / len(differences),
                "learned_vs_conflict_paired_counts": _paired_counts(differences),
            }
        )
    return summaries


def _scene_summaries(
    rows: dict[tuple[str, int, int, str], dict[str, Any]],
    *,
    protocol: ConfirmatoryProtocol,
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for task in protocol.tasks:
        for episode in protocol.heldout_episodes:
            condition_successes = {
                condition: sum(
                    _success(rows[(task.name, episode, seed, condition)])
                    for seed in protocol.noise_seeds
                )
                for condition in protocol.conditions
            }
            differences = [
                _success(rows[(task.name, episode, seed, "learned_monitor")])
                - _success(rows[(task.name, episode, seed, "conflict")])
                for seed in protocol.noise_seeds
            ]
            summaries.append(
                {
                    "task": task.name,
                    "episode_index": episode,
                    "trials_per_condition": len(protocol.noise_seeds),
                    "condition_successes": condition_successes,
                    "learned_minus_conflict": sum(differences) / len(differences),
                    "learned_vs_conflict_paired_counts": _paired_counts(differences),
                }
            )
    return summaries


def aggregate_confirmatory_reports(
    protocol_path: str | Path,
    report_paths: tuple[str | Path, ...],
    *,
    repository: str | Path,
    checkpoint: str | Path,
) -> dict[str, object]:
    """Verify complete sealed coverage and execute only the frozen analysis plan."""

    root = Path(repository).resolve()
    protocol_source = Path(protocol_path).resolve()
    protocol = ConfirmatoryProtocol.load(protocol_source)
    protocol.verify_sources(root)
    checkpoint_path = protocol.verify_checkpoint(checkpoint)
    loaded: list[tuple[Path, ConfirmatoryTask, int, dict[str, Any]]] = []
    observed: set[tuple[str, int]] = set()
    for raw_path in report_paths:
        path = Path(raw_path).resolve()
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"Confirmatory report is not a JSON object: {path}")
        report = cast(dict[str, Any], raw)
        task = _task_for_report(protocol, report)
        noise_seed = int(report["noise_seed"])
        key = (task.name, noise_seed)
        if key in observed:
            raise ValueError(f"Duplicate confirmatory task/noise report {key}")
        observed.add(key)
        protocol.verify_task_assets(root, task)
        verify_confirmatory_report(
            protocol,
            task,
            report,
            repository=root,
            checkpoint=checkpoint_path,
            noise_seed=noise_seed,
        )
        loaded.append((path, task, noise_seed, report))
    expected = {(task.name, seed) for task in protocol.tasks for seed in protocol.noise_seeds}
    if observed != expected:
        raise ValueError("Confirmatory reports do not cover every task/noise cell")
    task_order = {task.name: index for index, task in enumerate(protocol.tasks)}
    seed_order = {seed: index for index, seed in enumerate(protocol.noise_seeds)}
    loaded.sort(key=lambda item: (task_order[item[1].name], seed_order[item[2]]))
    rows = _row_index(loaded)

    trials_per_condition = (
        len(protocol.tasks) * len(protocol.heldout_episodes) * len(protocol.noise_seeds)
    )
    condition_successes = {
        condition: sum(_success(row) for key, row in rows.items() if key[3] == condition)
        for condition in protocol.conditions
    }
    primary = _contrast_summary(
        rows,
        protocol=protocol,
        treatment="learned_monitor",
        baseline="conflict",
    )
    forced = _contrast_summary(
        rows,
        protocol=protocol,
        treatment="aligned_forced_steer",
        baseline="aligned_no_steer",
    )
    release = _contrast_summary(
        rows,
        protocol=protocol,
        treatment="always_steer",
        baseline="learned_monitor",
    )
    task_summaries = _group_summary(rows, protocol=protocol, group="task")
    scene_summaries = _scene_summaries(rows, protocol=protocol)
    noise_summaries = _group_summary(rows, protocol=protocol, group="noise")

    agreement_count = sum(
        _success(rows[(task.name, episode, seed, "learned_monitor")])
        == _success(rows[(task.name, episode, seed, "oracle_monitor")])
        for task in protocol.tasks
        for episode in protocol.heldout_episodes
        for seed in protocol.noise_seeds
    )
    learned_oracle_agreement = agreement_count / trials_per_condition
    positive_tasks = sum(_numeric(row, "learned_minus_conflict") > 0 for row in task_summaries)
    positive_noise_seeds = sum(
        _numeric(row, "learned_minus_conflict") > 0 for row in noise_summaries
    )
    aggregate_effect = _numeric(primary, "cluster_mean_success_difference")
    direction_specific = condition_successes["learned_monitor"] > max(
        condition_successes["wrong_sign"],
        condition_successes["random"],
    )
    gates = {
        "positive_aggregate_effect": aggregate_effect > 0,
        "minimum_positive_tasks": (positive_tasks >= protocol.analysis.minimum_positive_tasks),
        "minimum_positive_noise_seeds": (
            positive_noise_seeds >= protocol.analysis.minimum_positive_noise_seeds
        ),
        "minimum_learned_oracle_agreement": (
            learned_oracle_agreement >= protocol.analysis.minimum_learned_oracle_agreement
        ),
        "learned_above_wrong_sign_and_random": direction_specific,
    }
    required_gates = [
        gates["minimum_positive_tasks"],
        gates["minimum_positive_noise_seeds"],
        gates["minimum_learned_oracle_agreement"],
    ]
    if protocol.analysis.require_positive_aggregate_effect:
        required_gates.append(gates["positive_aggregate_effect"])
    if protocol.analysis.require_learned_above_controls:
        required_gates.append(gates["learned_above_wrong_sign_and_random"])

    return {
        "schema_version": 2,
        "protocol_id": protocol.protocol_id,
        "protocol_path": _display_path(protocol_source, root),
        "protocol_sha256": sha256_file(protocol_source),
        "checkpoint": _display_path(checkpoint_path, root),
        "heldout_episodes": list(protocol.heldout_episodes),
        "noise_seeds": list(protocol.noise_seeds),
        "noise_seed_selection_rule": protocol.noise_seed_selection_rule,
        "noise_schedule_id": protocol.noise_schedule_id,
        "trials_per_condition": trials_per_condition,
        "total_rollouts": trials_per_condition * len(protocol.conditions),
        "condition_successes": condition_successes,
        "primary_contrast": primary,
        "aligned_forced_steer_contrast": forced,
        "always_steer_vs_learned_release_contrast": release,
        "learned_oracle_success_agreement": learned_oracle_agreement,
        "learned_oracle_success_disagreements": trials_per_condition - agreement_count,
        "positive_task_count": positive_tasks,
        "positive_noise_seed_count": positive_noise_seeds,
        "task_summaries": task_summaries,
        "scene_summaries": scene_summaries,
        "noise_seed_summaries": noise_summaries,
        "paper_update_gates": gates,
        "confirmatory_support": all(required_gates),
        "inference_note": (
            "The confidence interval resamples task-scene clusters and keeps all noise "
            "replicates together; rollout-level paired counts are descriptive."
        ),
        "reports": [
            {
                "path": _display_path(path, root),
                "sha256": sha256_file(path),
                "task": task.name,
                "noise_seed": noise_seed,
            }
            for path, task, noise_seed, _ in loaded
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m causal_vla.confirmatory_monitor_analysis")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = aggregate_confirmatory_reports(
        args.protocol,
        tuple(args.report),
        repository=args.repository,
        checkpoint=args.checkpoint,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
