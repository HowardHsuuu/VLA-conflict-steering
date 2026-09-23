"""Validate and summarize the public aggregate result release.

The Git repository intentionally contains aggregate tables rather than fitted
weights, raw rollout payloads, or machine-specific execution receipts. This
module checks the released table, its internal arithmetic, and the declared
protocol inventory using only the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

TASK_SPECS: dict[str, tuple[str, int]] = {
    "object-ketchup-vs-absent-milk": ("libero_object", 4),
    "object-milk-vs-absent-cream-cheese": ("libero_object", 7),
    "spatial-cookie-box-vs-cabinet": ("libero_spatial", 3),
    "spatial-cookie-box-vs-stove": ("libero_spatial", 6),
    "spatial-ramekin-vs-cookie-box": ("libero_spatial", 5),
}
TASKS = tuple(TASK_SPECS)
SCENES = (46, 47, 48, 49)
NOISE_SEEDS = (10057, 10157, 10257)
CONDITIONS = (
    "conflict",
    "always_steer",
    "oracle_monitor",
    "learned_monitor",
    "aligned_no_steer",
    "aligned_monitor",
    "aligned_forced_steer",
    "wrong_sign",
    "random",
    "monitor_prompt_correction",
    "oracle_prompt_correction",
)
EXPECTED_CONDITION_SUCCESSES = {
    "conflict": 5,
    "always_steer": 46,
    "oracle_monitor": 45,
    "learned_monitor": 45,
    "aligned_no_steer": 46,
    "aligned_monitor": 46,
    "aligned_forced_steer": 9,
    "wrong_sign": 1,
    "random": 8,
    "monitor_prompt_correction": 49,
    "oracle_prompt_correction": 49,
}
EXPECTED_TASK_COUNTS = {
    "conflict": {
        "object-ketchup-vs-absent-milk": 5,
        "object-milk-vs-absent-cream-cheese": 0,
        "spatial-cookie-box-vs-cabinet": 0,
        "spatial-cookie-box-vs-stove": 0,
        "spatial-ramekin-vs-cookie-box": 0,
    },
    "learned_monitor": {
        "object-ketchup-vs-absent-milk": 11,
        "object-milk-vs-absent-cream-cheese": 10,
        "spatial-cookie-box-vs-cabinet": 6,
        "spatial-cookie-box-vs-stove": 8,
        "spatial-ramekin-vs-cookie-box": 10,
    },
    "monitor_prompt_correction": {
        "object-ketchup-vs-absent-milk": 12,
        "object-milk-vs-absent-cream-cheese": 10,
        "spatial-cookie-box-vs-cabinet": 6,
        "spatial-cookie-box-vs-stove": 10,
        "spatial-ramekin-vs-cookie-box": 11,
    },
    "oracle_prompt_correction": {
        "object-ketchup-vs-absent-milk": 12,
        "object-milk-vs-absent-cream-cheese": 10,
        "spatial-cookie-box-vs-cabinet": 6,
        "spatial-cookie-box-vs-stove": 10,
        "spatial-ramekin-vs-cookie-box": 11,
    },
}


@dataclass(frozen=True)
class PublicResultSummary:
    """Validated quantities from the public aggregate table."""

    total_rollouts: int
    trials_per_condition: int
    condition_successes: dict[str, int]
    task_condition_successes: dict[str, dict[str, int]]
    learned_vs_conflict: dict[str, int]
    prompt_vs_steering: dict[str, int]
    conflict_monitor_triggers: int
    aligned_monitor_triggers: int
    primary_pooled_absolute_gain: float
    primary_task_effects: dict[str, float]
    scene_cluster_bootstrap_95: tuple[float, float]
    preregistered_gates: dict[str, bool]
    evidence_level: str


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a regular file."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def verify_public_results(archive_root: str | Path) -> PublicResultSummary:
    """Validate the aggregate-only public release and its arithmetic."""

    root = Path(archive_root).resolve(strict=True)
    manifest = _json_object(root / "manifest.json")
    if manifest != {
        "schema_version": 1,
        "release_id": "five-suite-evaluation-v1",
        "evidence_level": "aggregate_only",
        "summary_file": "aggregate_results.json",
        "raw_rollouts_included": False,
        "controller_weights_included": False,
    }:
        raise ValueError("Unsupported or changed public release manifest")
    observed = {path.name for path in root.iterdir() if path.is_file()}
    if observed != {"manifest.json", "aggregate_results.json"}:
        raise ValueError("Aggregate release inventory mismatch")

    data = _json_object(root / "aggregate_results.json")
    design = cast(dict[str, Any], data.get("design"))
    if (
        data.get("schema_version") != 1
        or data.get("status") != "paper_evidence_gate_pass"
        or data.get("evidence_level") != "aggregate_only"
        or data.get("total_rollouts") != 660
        or design.get("tasks") != list(TASKS)
        or design.get("scenes") != list(SCENES)
        or design.get("noise_seeds") != list(NOISE_SEEDS)
        or design.get("conditions") != list(CONDITIONS)
    ):
        raise ValueError("Aggregate release design differs from the evaluated protocol")

    condition_successes = {
        str(key): int(value)
        for key, value in cast(dict[str, int], data["condition_successes_out_of_60"]).items()
    }
    if condition_successes != EXPECTED_CONDITION_SUCCESSES:
        raise ValueError("Aggregate condition counts do not match the released headline")

    task_counts = {
        str(condition): {str(task): int(value) for task, value in counts.items()}
        for condition, counts in cast(
            dict[str, dict[str, int]], data["selected_task_successes_out_of_12"]
        ).items()
    }
    if task_counts != EXPECTED_TASK_COUNTS:
        raise ValueError("Aggregate per-task counts do not match the released table")
    for condition, counts in task_counts.items():
        if sum(counts.values()) != condition_successes[condition]:
            raise ValueError(f"Per-task counts do not sum to the pooled count: {condition}")

    learned_vs_conflict = {
        str(key): int(value)
        for key, value in cast(dict[str, int], data["learned_vs_conflict"]).items()
    }
    prompt_vs_steering = {
        str(key): int(value)
        for key, value in cast(dict[str, int], data["prompt_vs_steering"]).items()
    }
    if learned_vs_conflict != {"wins": 40, "losses": 0, "ties": 20}:
        raise ValueError("Primary paired counts differ from the released analysis")
    if prompt_vs_steering != {"prompt_only": 7, "steering_only": 3, "ties": 50}:
        raise ValueError("Prompt comparison differs from the released analysis")

    monitor = cast(dict[str, int], data["monitor_triggers"])
    conflict_triggers = int(monitor["conflict_out_of_60"])
    aligned_triggers = int(monitor["aligned_out_of_60"])
    if (conflict_triggers, aligned_triggers) != (60, 0):
        raise ValueError("Monitor trigger counts differ from the released analysis")

    task_effects = {
        task: (task_counts["learned_monitor"][task] - task_counts["conflict"][task]) / 12
        for task in TASKS
    }
    pooled_gain = (condition_successes["learned_monitor"] - condition_successes["conflict"]) / 60
    interval_values = cast(list[float], data["bootstrap_95"])
    if len(interval_values) != 2:
        raise ValueError("Bootstrap interval differs from the released analysis")
    interval = (float(interval_values[0]), float(interval_values[1]))
    if interval != (0.55, 0.7833333333):
        raise ValueError("Bootstrap interval differs from the released analysis")

    gates = {
        str(key): bool(value)
        for key, value in cast(dict[str, bool], data["preregistered_gates"]).items()
    }
    if len(gates) != 10 or not all(gates.values()):
        raise ValueError("Not all ten preregistered gates pass")

    return PublicResultSummary(
        total_rollouts=660,
        trials_per_condition=60,
        condition_successes=condition_successes,
        task_condition_successes=task_counts,
        learned_vs_conflict=learned_vs_conflict,
        prompt_vs_steering=prompt_vs_steering,
        conflict_monitor_triggers=conflict_triggers,
        aligned_monitor_triggers=aligned_triggers,
        primary_pooled_absolute_gain=pooled_gain,
        primary_task_effects=task_effects,
        scene_cluster_bootstrap_95=interval,
        preregistered_gates=gates,
        evidence_level="aggregate_only",
    )


def verify_protocol(repository: str | Path) -> None:
    """Check the public protocol inventory without loading fitted weights."""

    root = Path(repository).resolve(strict=True)
    config = tomllib.loads(
        (root / "configs/cross_task_decisive_v1.toml").read_text(encoding="utf-8")
    )
    if (
        config.get("campaign_id") != "five-suite-closed-loop-v1"
        or tuple(config.get("scenes", ())) != SCENES
        or tuple(config.get("flow_noise_seeds", ())) != NOISE_SEEDS
        or tuple(config.get("conditions", ())) != CONDITIONS
        or len(cast(list[object], config.get("tasks", []))) != 5
    ):
        raise ValueError("Public protocol inventory changed")


def _percent(successes: int, trials: int) -> str:
    return f"{100.0 * successes / trials:.1f}%"


def format_summary(summary: PublicResultSummary) -> str:
    """Format a compact verification receipt."""

    order = (
        ("Conflict, unsteered", "conflict"),
        ("Action-flow, always on", "always_steer"),
        ("Action-flow, oracle trigger", "oracle_monitor"),
        ("Action-flow, learned trigger", "learned_monitor"),
        ("Prompt correction, learned", "monitor_prompt_correction"),
        ("Prompt correction, oracle", "oracle_prompt_correction"),
        ("Aligned, unsteered", "aligned_no_steer"),
        ("Aligned, learned monitor", "aligned_monitor"),
        ("Aligned, forced steering", "aligned_forced_steer"),
        ("Wrong-sign steering", "wrong_sign"),
        ("Random-direction steering", "random"),
    )
    lines = [
        "Condition                         Success",
        "-------------------------------- -------",
    ]
    for label, key in order:
        successes = summary.condition_successes[key]
        lines.append(
            f"{label:<32} {_percent(successes, summary.trials_per_condition):>6} "
            f"({successes}/{summary.trials_per_condition})"
        )
    paired = summary.learned_vs_conflict
    prompt = summary.prompt_vs_steering
    lines.extend(
        [
            "",
            f"Validated aggregate for {summary.total_rollouts} rollouts; "
            "all 10 preregistered gates pass.",
            "Learned action-flow vs conflict: "
            f"{paired['wins']} wins, {paired['losses']} losses, {paired['ties']} ties.",
            "Monitor triggers: "
            f"{summary.conflict_monitor_triggers}/60 conflict, "
            f"{summary.aligned_monitor_triggers}/60 aligned.",
            "Same-information prompt correction vs action-flow: "
            f"{prompt['prompt_only']} prompt-only, {prompt['steering_only']} steering-only, "
            f"{prompt['ties']} ties.",
            "Task-stratified scene-cluster bootstrap 95% interval: "
            f"[{summary.scene_cluster_bootstrap_95[0]:.3f}, "
            f"{summary.scene_cluster_bootstrap_95[1]:.3f}].",
            "Release scope: aggregate tables only; raw rollouts and fitted weights are excluded.",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("results/cross_task_decisive_v1"))
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--json", action="store_true", help="emit the receipt as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = verify_public_results(args.archive)
    verify_protocol(args.repository)
    output = (
        json.dumps(asdict(result), indent=2, sort_keys=True)
        if args.json
        else format_summary(result)
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
