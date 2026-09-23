"""Frozen intention-to-treat analysis for the five-task closed-loop campaign."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from causal_vla.cross_task_decisive_campaign import (
    DEFAULT_OUTPUT,
    _read_completed,
    _stem,
    _write_once,
    build_plan,
)


def analyze(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    plan = build_plan()
    with (output / "plan.json").open(encoding="utf-8") as stream:
        if json.load(stream) != plan:
            raise ValueError("Frozen plan or code/assets changed before analysis")
    with (output / "summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    if (
        summary.get("status") != "complete_awaiting_frozen_analysis"
        or summary.get("plan_sha256") != plan["plan_sha256"]
        or summary.get("completed_cells") != 660
    ):
        raise ValueError("Campaign is not fully and cleanly completed")

    outcomes: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for cell in plan["inventory"]:
        path = output / "cells" / (_stem(cell) + ".json")
        if not _read_completed(path, cell, plan["plan_sha256"]):
            raise ValueError(f"Missing planned cell: {_stem(cell)}")
        with path.open(encoding="utf-8") as stream:
            outcome = json.load(stream)["outcome"]
        key = (cell["task"], cell["scene"], cell["noise_seed"], cell["condition"])
        outcomes[key] = outcome
    if len(outcomes) != 660:
        raise ValueError("Missing or duplicate analysis row")

    tasks = sorted({key[0] for key in outcomes})
    scenes = sorted({key[1] for key in outcomes})
    seeds = sorted({key[2] for key in outcomes})
    conditions = sorted({key[3] for key in outcomes})
    counts: dict[str, dict[str, int]] = {}
    for condition in conditions:
        counts[condition] = {
            task: sum(
                outcomes[(task, scene, seed, condition)]["success"]
                for scene in scenes
                for seed in seeds
            )
            for task in tasks
        }
        counts[condition]["all"] = sum(counts[condition].values())

    differences: dict[str, float] = {}
    wins = losses = ties = 0
    for task in tasks:
        learned = counts["learned_monitor"][task]
        baseline = counts["conflict"][task]
        differences[task] = (learned - baseline) / (len(scenes) * len(seeds))
        for scene in scenes:
            for seed in seeds:
                a = outcomes[(task, scene, seed, "learned_monitor")]["success"]
                b = outcomes[(task, scene, seed, "conflict")]["success"]
                wins += int(a and not b)
                losses += int(b and not a)
                ties += int(a == b)

    # Resample physical scenes within each task; noise replicates stay clustered.
    cluster_effects = np.asarray(
        [
            [
                np.mean(
                    [
                        float(outcomes[(task, scene, seed, "learned_monitor")]["success"])
                        - float(outcomes[(task, scene, seed, "conflict")]["success"])
                        for seed in seeds
                    ]
                )
                for scene in scenes
            ]
            for task in tasks
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(20260921)
    draws = np.empty(20_000, dtype=np.float64)
    for index in range(len(draws)):
        sampled = rng.integers(0, len(scenes), size=cluster_effects.shape)
        draws[index] = np.take_along_axis(cluster_effects, sampled, axis=1).mean()
    ci_lower, ci_upper = (float(x) for x in np.quantile(draws, (0.025, 0.975)))

    conflict_triggers = sum(
        outcomes[(task, scene, seed, "learned_monitor")]["trigger_step"] is not None
        for task in tasks
        for scene in scenes
        for seed in seeds
    )
    aligned_false_triggers = sum(
        outcomes[(task, scene, seed, "aligned_monitor")]["trigger_step"] is not None
        for task in tasks
        for scene in scenes
        for seed in seeds
    )
    aligned_harm = (counts["aligned_no_steer"]["all"] - counts["aligned_monitor"]["all"]) / 60
    pooled_gain = (counts["learned_monitor"]["all"] - counts["conflict"]["all"]) / 60
    spatial_tasks = [task for task in tasks if task.startswith("spatial-")]
    gate = {
        "positive_tasks_at_least_4_of_5": sum(x > 0 for x in differences.values()) >= 4,
        "positive_spatial_tasks_at_least_2_of_3": sum(
            differences[task] > 0 for task in spatial_tasks
        )
        >= 2,
        "pooled_absolute_gain_at_least_20pp": pooled_gain >= 0.20,
        "task_stratified_scene_cluster_ci_lower_above_zero": ci_lower > 0,
        "monitor_conflict_trigger_rate_at_least_80pct": conflict_triggers >= 48,
        "aligned_false_trigger_rate_at_most_5pct": aligned_false_triggers <= 3,
        "aligned_success_harm_at_most_5pp": aligned_harm <= 0.05,
        "learned_success_above_wrong_sign": (
            counts["learned_monitor"]["all"] > counts["wrong_sign"]["all"]
        ),
        "learned_success_above_random": (
            counts["learned_monitor"]["all"] > counts["random"]["all"]
        ),
        "oracle_prompt_viability_at_least_50pct_each_task": all(
            counts["oracle_prompt_correction"][task] >= 6 for task in tasks
        ),
    }
    prompt_pairs = defaultdict(int)
    for task in tasks:
        for scene in scenes:
            for seed in seeds:
                steer = outcomes[(task, scene, seed, "learned_monitor")]["success"]
                prompt = outcomes[(task, scene, seed, "monitor_prompt_correction")]["success"]
                prompt_pairs["steering_only"] += int(steer and not prompt)
                prompt_pairs["prompt_only"] += int(prompt and not steer)
                prompt_pairs["ties"] += int(steer == prompt)
    return {
        "schema_version": 1,
        "plan_sha256": plan["plan_sha256"],
        "status": "paper_evidence_gate_pass" if all(gate.values()) else "paper_evidence_gate_fail",
        "interpretation": (
            "A pass supports drafting a configured-task full paper; it does not guarantee "
            "acceptance, safety, zero-shot transfer, or real-robot deployment."
        ),
        "cells": 660,
        "tasks": tasks,
        "scenes": scenes,
        "noise_seeds": seeds,
        "condition_success_counts_out_of_12_per_task_and_60_all": counts,
        "primary_paired": {"wins": wins, "losses": losses, "ties": ties},
        "primary_task_effects": differences,
        "primary_pooled_absolute_gain": pooled_gain,
        "scene_cluster_bootstrap_95": [ci_lower, ci_upper],
        "monitor": {
            "conflict_triggered_out_of_60": conflict_triggers,
            "aligned_false_triggers_out_of_60": aligned_false_triggers,
        },
        "aligned_success_harm": aligned_harm,
        "same_information_prompt_comparison": dict(prompt_pairs),
        "preregistered_gates": gate,
        "failure_next_plan": plan["decision"]["next_plan_if_fail"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze(args.output)
    _write_once(args.output / "analysis.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "paper_evidence_gate_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
