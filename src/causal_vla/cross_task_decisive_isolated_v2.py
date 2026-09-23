"""Outcome-blind process-isolation repair for the frozen 660-cell campaign.

This changes only execution lifetime: each task/scene/noise/phase runs in a
fresh Python process. The v1 config, scientific runtime, prompts, controller
banks, inventory, analysis, and already-completed cell bytes remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from causal_vla import cross_task_decisive_campaign as v1
from causal_vla.confirmatory_monitor_runtime import run_confirmatory_monitor_evaluation
from causal_vla.cross_task_prompt_comparator import run_prompt_comparators

OUTPUT = v1.DEFAULT_OUTPUT
RECEIPT_NAME = "infrastructure_remediation_v2.json"
EXPECTED_RETAINED_CELLS = 175


def _receipt_payload(output: Path) -> dict[str, Any]:
    plan = v1.build_plan()
    plan_path = output / "plan.json"
    with plan_path.open(encoding="utf-8") as stream:
        if json.load(stream) != plan:
            raise ValueError("The v1 scientific plan changed")
    if (output / "summary.json").exists():
        raise ValueError("Cannot remediate a completed v1 campaign")
    expected = {v1._stem(cell) + ".json": cell for cell in plan["inventory"]}
    files = sorted((output / "cells").glob("*.json"))
    if len(files) != EXPECTED_RETAINED_CELLS:
        raise ValueError("Retained cell count differs from the observed stall boundary")
    for path in files:
        if path.name not in expected or not v1._read_completed(
            path, expected[path.name], plan["plan_sha256"]
        ):
            raise ValueError(f"Invalid retained v1 cell: {path}")
    return {
        "schema_version": 1,
        "status": "outcome_blind_process_isolation_only",
        "v1_plan_sha256": plan["plan_sha256"],
        "v1_config_sha256": plan["config_sha256"],
        "v1_source_sha256": plan["source_sha256"],
        "v1_asset_sha256": plan["asset_sha256"],
        "v2_runner_sha256": v1._sha(Path(__file__)),
        "retained_completed_cells": EXPECTED_RETAINED_CELLS,
        "retained_cell_sha256": {path.name: v1._sha(path) for path in files},
        "original_stall_boundary": (
            "175/660 complete; v1 single process had no new cell for over five "
            "hours while using approximately 28 GB footprint on a 24 GB Mac"
        ),
        "change": "fresh_process_per_task_scene_noise_phase",
        "unchanged": [
            "all_660_intention_to_treat_cells",
            "task_scene_noise_condition_order",
            "frozen_scientific_source",
            "controller_and_monitor_banks",
            "prompt_comparators",
            "preregistered_gates",
            "completed_v1_cell_bytes",
        ],
        "incomplete_cell_rule": "rerun_exact_frozen_cell_only_not_replace_or_retune",
        "partial_efficacy_used_to_design_repair": False,
    }


def freeze(output: Path) -> dict[str, Any]:
    receipt_path = output / RECEIPT_NAME
    if receipt_path.exists():
        with receipt_path.open(encoding="utf-8") as stream:
            actual = json.load(stream)
        plan = v1.build_plan()
        if (
            actual.get("status") != "outcome_blind_process_isolation_only"
            or actual.get("v1_plan_sha256") != plan["plan_sha256"]
            or actual.get("v1_config_sha256") != plan["config_sha256"]
            or actual.get("v1_source_sha256") != plan["source_sha256"]
            or actual.get("v1_asset_sha256") != plan["asset_sha256"]
            or actual.get("v2_runner_sha256") != v1._sha(Path(__file__))
            or actual.get("retained_completed_cells") != EXPECTED_RETAINED_CELLS
            or len(actual.get("retained_cell_sha256", {})) != EXPECTED_RETAINED_CELLS
            or any(
                v1._sha(output / "cells" / name) != digest
                for name, digest in actual["retained_cell_sha256"].items()
            )
        ):
            raise ValueError("Process-isolation receipt changed")
        return actual
    expected = _receipt_payload(output)
    v1._write_once(receipt_path, expected)
    return expected


def _assert_frozen(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = v1.build_plan()
    config = v1._config()
    if freeze(output)["v1_plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("Frozen v1 plan changed")
    return plan, config


def _base(task: dict[str, Any], scene: int, seed: int) -> dict[str, Any]:
    return {
        "task": task["name"],
        "suite": task["suite"],
        "task_id": task["task_id"],
        "scene": scene,
        "noise_seed": seed,
    }


def _missing(
    output: Path,
    plan: dict[str, Any],
    base: dict[str, Any],
    phase: str,
) -> tuple[str, ...]:
    conditions = v1.OLD_CONDITIONS if phase == "old" else v1.PROMPT_CONDITIONS
    return tuple(
        condition
        for condition in conditions
        if not v1._read_completed(
            output / "cells" / (v1._stem({**base, "condition": condition}) + ".json"),
            {**base, "condition": condition},
            plan["plan_sha256"],
        )
    )


def child(output: Path, task_name: str, scene: int, seed: int, phase: str) -> None:
    plan, config = _assert_frozen(output)
    matching = [task for task in config["tasks"] if task["name"] == task_name]
    if (
        len(matching) != 1
        or scene not in config["scenes"]
        or seed not in config["flow_noise_seeds"]
        or phase not in ("old", "prompt")
    ):
        raise ValueError("Child request is outside the fixed inventory")
    task = matching[0]
    base = _base(task, scene, seed)
    conditions = _missing(output, plan, base, phase)
    if not conditions:
        return

    if phase == "old":

        def progress(episode: Any) -> None:
            v1._record(output, plan, {**base, "condition": episode.condition}, asdict(episode))

        run_confirmatory_monitor_evaluation(
            v1.ROOT / config["checkpoint"],
            v1.ROOT / task["steering_bank"],
            v1.ROOT / task["monitor_bank"],
            release_monitor_bank_path=(
                v1.ROOT / task["release_monitor_bank"] if "release_monitor_bank" in task else None
            ),
            task_id=task["task_id"],
            episode_indices=(scene,),
            conditions=conditions,
            suite=task["suite"],
            device=config["device"],
            noise_seed=seed,
            simulator_seed=config["simulator_seed"],
            max_steps=config["max_steps"],
            release_mode=task["release_mode"],
            release_monitor_interval=task["release_monitor_interval"],
            release_reactivation_patience=task["release_reactivation_patience"],
            grasp_close_fraction=task["grasp_close_fraction"],
            grasp_reopen_fraction=task["grasp_reopen_fraction"],
            grasp_lift_threshold=task["grasp_lift_threshold"],
            grasp_release_patience=task["grasp_release_patience"],
            monitor_arm_timeout=task["monitor_arm_timeout"],
            progress=progress,
        )
    else:

        def progress(outcome: dict[str, Any]) -> None:
            v1._record(output, plan, {**base, "condition": outcome["condition"]}, outcome)

        run_prompt_comparators(
            v1.ROOT / config["checkpoint"],
            v1.ROOT / task["monitor_bank"],
            suite=task["suite"],
            task_id=task["task_id"],
            episode_indices=(scene,),
            noise_seed=seed,
            conditions=conditions,
            device=config["device"],
            max_steps=config["max_steps"],
            simulator_seed=config["simulator_seed"],
            progress=progress,
        )
    if _missing(output, plan, base, phase):
        raise AssertionError("Child returned without completing all requested cells")


def parent(output: Path) -> dict[str, Any]:
    plan, config = _assert_frozen(output)
    started = time.monotonic()
    active: dict[str, Any] | None = None
    try:
        for task in config["tasks"]:
            for seed in config["flow_noise_seeds"]:
                for scene in config["scenes"]:
                    base = _base(task, scene, seed)
                    for phase in ("old", "prompt"):
                        if not _missing(output, plan, base, phase):
                            continue
                        active = {**base, "phase": phase}
                        print(json.dumps({"starting": active}), flush=True)
                        result = subprocess.run(
                            [
                                sys.executable,
                                "-m",
                                "causal_vla.cross_task_decisive_isolated_v2",
                                "--child",
                                "--output",
                                str(output),
                                "--task",
                                task["name"],
                                "--scene",
                                str(scene),
                                "--seed",
                                str(seed),
                                "--phase",
                                phase,
                            ],
                            cwd=v1.ROOT,
                            env=os.environ.copy(),
                            check=False,
                        )
                        if result.returncode != 0:
                            raise RuntimeError(
                                f"Isolated child exited {result.returncode}: {active}"
                            )
                        if _missing(output, plan, base, phase):
                            raise AssertionError(
                                f"Isolated child left a planned cell missing: {active}"
                            )
                        print(json.dumps({"finished": active}), flush=True)
                        active = None
        if v1.build_plan() != plan or freeze(output)["v1_plan_sha256"] != plan["plan_sha256"]:
            raise ValueError("Frozen scientific plan changed during remediation")
        completed = sum(
            v1._read_completed(
                output / "cells" / (v1._stem(cell) + ".json"),
                cell,
                plan["plan_sha256"],
            )
            for cell in plan["inventory"]
        )
        if completed != 660:
            raise AssertionError("Isolated campaign ended with missing cells")
        summary = {
            "schema_version": 1,
            "status": "complete_awaiting_frozen_analysis",
            "plan_sha256": plan["plan_sha256"],
            "completed_cells": 660,
            "expected_cells": 660,
            "infrastructure_remediation_receipt": RECEIPT_NAME,
            "elapsed_wall_seconds_v2": time.monotonic() - started,
        }
        v1._write_once(output / "summary.json", summary)
        return summary
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "status": "v2_stopped_preserve_all_evidence",
            "plan_sha256": plan["plan_sha256"],
            "active": active,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "time_ns": time.time_ns(),
        }
        v1._write_once(output / f"v2_failure_{failure['time_ns']}.json", failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--task")
    parser.add_argument("--scene", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--phase", choices=("old", "prompt"))
    args = parser.parse_args()
    output = args.output.resolve()
    if args.freeze_only:
        receipt = freeze(output)
        print(
            json.dumps(
                {
                    "v1_plan_sha256": receipt["v1_plan_sha256"],
                    "retained_completed_cells": receipt["retained_completed_cells"],
                }
            )
        )
    elif args.child:
        if any(value is None for value in (args.task, args.scene, args.seed, args.phase)):
            parser.error("Child requires task, scene, seed, and phase")
        child(output, args.task, args.scene, args.seed, args.phase)
    else:
        print(json.dumps(parent(output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
