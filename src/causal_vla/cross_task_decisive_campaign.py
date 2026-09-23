"""Write-once, cell-resumable five-task closed-loop campaign.

The complete inventory and success gate live in the frozen TOML. A completed
cell is never replayed or replaced. An interrupted cell may be run again only
with byte-identical code, assets, scene, seed, and condition, retaining its
interruption receipt. The runner does not inspect efficacy to select work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import tomllib
import traceback
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

from causal_vla.confirmatory_monitor_runtime import run_confirmatory_monitor_evaluation
from causal_vla.cross_task_prompt_comparator import run_prompt_comparators

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/cross_task_decisive_v1.toml"
DEFAULT_OUTPUT = ROOT / "artifacts/results/cross_task_decisive_v1_run"
OLD_CONDITIONS = (
    "conflict",
    "always_steer",
    "oracle_monitor",
    "learned_monitor",
    "aligned_no_steer",
    "aligned_monitor",
    "aligned_forced_steer",
    "wrong_sign",
    "random",
)
PROMPT_CONDITIONS = ("monitor_prompt_correction", "oracle_prompt_correction")
SOURCE_FILES = (
    "src/causal_vla/cross_task_decisive_campaign.py",
    "src/causal_vla/cross_task_decisive_analysis.py",
    "src/causal_vla/cross_task_prompt_comparator.py",
    "src/causal_vla/confirmatory_monitor_runtime.py",
    "src/causal_vla/causal_trace_runtime.py",
    "src/causal_vla/expert_ridge_runtime.py",
    "src/causal_vla/expert_knn_runtime.py",
    "src/causal_vla/location_monitor_runtime.py",
    "src/causal_vla/object_monitor_runtime.py",
    "src/causal_vla/scene_intervention.py",
    "src/causal_vla/object_intervention.py",
    "src/causal_vla/smoke.py",
    "third_party/action-atlas/experiments/model_adapters.py",
    "third_party/action-atlas/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py",
)


def _bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or linked campaign input: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _input_path(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Campaign input must be repository-relative")
    full = ROOT / path
    _sha(full)
    return full


def _config() -> dict[str, Any]:
    with CONFIG.open("rb") as stream:
        config = tomllib.load(stream)
    if (
        config["schema_version"] != 1
        or config["campaign_id"] != "five-suite-closed-loop-v1"
        or config["scenes"] != [46, 47, 48, 49]
        or config["flow_noise_seeds"] != [10057, 10157, 10257]
        or config["max_steps"] != 220
        or config["device"] != "mps"
        or tuple(config["conditions"]) != OLD_CONDITIONS + PROMPT_CONDITIONS
        or [(x["suite"], x["task_id"]) for x in config["tasks"]]
        != [
            ("libero_object", 4),
            ("libero_object", 7),
            ("libero_spatial", 3),
            ("libero_spatial", 5),
            ("libero_spatial", 6),
        ]
    ):
        raise ValueError("Decisive campaign inventory differs from reviewed contract")
    if len({x["name"] for x in config["tasks"]}) != 5:
        raise ValueError("Task names must be unique")
    return config


def _inventory(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "task": task["name"],
            "suite": task["suite"],
            "task_id": task["task_id"],
            "scene": scene,
            "noise_seed": seed,
            "condition": condition,
        }
        for task in config["tasks"]
        for scene in config["scenes"]
        for seed in config["flow_noise_seeds"]
        for condition in config["conditions"]
    ]


def build_plan() -> dict[str, Any]:
    config = _config()
    inventory = _inventory(config)
    if len(inventory) != 660 or len({_stem(cell) for cell in inventory}) != 660:
        raise AssertionError("Decisive campaign must contain exactly 660 unique cells")
    assets = {
        relative
        for task in config["tasks"]
        for relative in (
            task["steering_bank"],
            task["monitor_bank"],
            task.get("release_monitor_bank", task["monitor_bank"]),
        )
    }
    checkpoint = _input_path(config["checkpoint"] + "/model.safetensors")
    checkpoint_config = _input_path(config["checkpoint"] + "/config.json")
    plan: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "status": "prospectively_frozen_complete_inventory",
        "config_sha256": _sha(CONFIG),
        "source_sha256": {path: _sha(_input_path(path)) for path in SOURCE_FILES},
        "asset_sha256": {path: _sha(_input_path(path)) for path in sorted(assets)},
        "checkpoint_weights_sha256": _sha(checkpoint),
        "checkpoint_config_sha256": _sha(checkpoint_config),
        "inventory": inventory,
        "decision": config["decision"],
        "protected_scenes_excluded": [10, 11, 12, 13, 14],
        "previous_task6_confirmation_scenes_excluded": list(range(36, 46)),
    }
    plan["plan_sha256"] = hashlib.sha256(_bytes(plan)).hexdigest()
    return plan


def _stem(cell: dict[str, Any]) -> str:
    return f"{cell['task']}_scene{cell['scene']}_noise{cell['noise_seed']}_{cell['condition']}"


def _read_completed(path: Path, cell: dict[str, Any], plan_hash: str) -> bool:
    if not path.exists():
        return False
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Invalid completed-cell path: {path}")
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if (
        payload.get("plan_sha256") != plan_hash
        or payload.get("cell") != cell
        or type(payload.get("outcome", {}).get("success")) is not bool
    ):
        raise ValueError(f"Completed cell failed validation: {path}")
    return True


def _record(
    output: Path,
    plan: dict[str, Any],
    cell: dict[str, Any],
    outcome: dict[str, Any],
) -> None:
    if (
        outcome.get("episode_index") != cell["scene"]
        or outcome.get("condition") != cell["condition"]
        or type(outcome.get("success")) is not bool
    ):
        raise ValueError("Completed outcome differs from the planned cell")
    path = output / "cells" / (_stem(cell) + ".json")
    _write_once(
        path,
        {"schema_version": 1, "plan_sha256": plan["plan_sha256"], "cell": cell, "outcome": outcome},
    )
    print(json.dumps({"completed": _stem(cell), "success": outcome["success"]}), flush=True)


def _record_episode(
    output: Path,
    plan: dict[str, Any],
    base: dict[str, Any],
    episode: Any,
) -> None:
    cell = {**base, "condition": episode.condition}
    _record(output, plan, cell, asdict(episode))


def _record_prompt(
    output: Path,
    plan: dict[str, Any],
    base: dict[str, Any],
    outcome: dict[str, Any],
) -> None:
    cell = {**base, "condition": outcome["condition"]}
    _record(output, plan, cell, outcome)


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    plan = build_plan()
    config = _config()
    output = output.resolve()
    if output.is_symlink():
        raise ValueError("Campaign output may not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    if plan_path.exists():
        with plan_path.open(encoding="utf-8") as stream:
            if json.load(stream) != plan:
                raise ValueError("Campaign plan or input hashes drifted; cannot resume")
    else:
        _write_once(plan_path, plan)

    expected = {_stem(cell) + ".json" for cell in plan["inventory"]}
    actual = {path.name for path in (output / "cells").glob("*.json")}
    if actual - expected:
        raise ValueError("Campaign contains undeclared cell output")
    current: dict[str, Any] | None = None
    try:
        for task in config["tasks"]:
            for seed in config["flow_noise_seeds"]:
                for scene in config["scenes"]:
                    base = {
                        "task": task["name"],
                        "suite": task["suite"],
                        "task_id": task["task_id"],
                        "scene": scene,
                        "noise_seed": seed,
                    }
                    missing_old = tuple(
                        condition
                        for condition in OLD_CONDITIONS
                        if not _read_completed(
                            output / "cells" / (_stem({**base, "condition": condition}) + ".json"),
                            {**base, "condition": condition},
                            plan["plan_sha256"],
                        )
                    )
                    if missing_old:
                        if build_plan() != plan:
                            raise ValueError("Code or assets changed during campaign")
                        current = {**base, "conditions": list(missing_old)}

                        old_progress = partial(_record_episode, output, plan, dict(base))

                        run_confirmatory_monitor_evaluation(
                            ROOT / config["checkpoint"],
                            ROOT / task["steering_bank"],
                            ROOT / task["monitor_bank"],
                            release_monitor_bank_path=(
                                ROOT / task["release_monitor_bank"]
                                if "release_monitor_bank" in task
                                else None
                            ),
                            task_id=task["task_id"],
                            episode_indices=(scene,),
                            conditions=missing_old,
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
                            progress=old_progress,
                        )
                    missing_prompt = tuple(
                        condition
                        for condition in PROMPT_CONDITIONS
                        if not _read_completed(
                            output / "cells" / (_stem({**base, "condition": condition}) + ".json"),
                            {**base, "condition": condition},
                            plan["plan_sha256"],
                        )
                    )
                    if missing_prompt:
                        if build_plan() != plan:
                            raise ValueError("Code or assets changed during campaign")
                        current = {**base, "conditions": list(missing_prompt)}

                        prompt_progress = partial(_record_prompt, output, plan, dict(base))

                        run_prompt_comparators(
                            ROOT / config["checkpoint"],
                            ROOT / task["monitor_bank"],
                            suite=task["suite"],
                            task_id=task["task_id"],
                            episode_indices=(scene,),
                            noise_seed=seed,
                            conditions=missing_prompt,
                            device=config["device"],
                            max_steps=config["max_steps"],
                            simulator_seed=config["simulator_seed"],
                            progress=prompt_progress,
                        )
                    current = None
        if build_plan() != plan:
            raise ValueError("Campaign input changed before completion")
        completed = sum(
            _read_completed(output / "cells" / (_stem(cell) + ".json"), cell, plan["plan_sha256"])
            for cell in plan["inventory"]
        )
        if completed != 660:
            raise AssertionError("Campaign ended without all planned cells")
        summary = {
            "schema_version": 1,
            "status": "complete_awaiting_frozen_analysis",
            "plan_sha256": plan["plan_sha256"],
            "completed_cells": completed,
            "expected_cells": 660,
        }
        _write_once(output / "summary.json", summary)
        return summary
    except BaseException as error:
        failures = output / "failure_events"
        failures.mkdir(parents=True, exist_ok=True)
        event = {
            "schema_version": 1,
            "plan_sha256": plan["plan_sha256"],
            "current": current,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "time_ns": time.time_ns(),
        }
        _write_once(failures / f"failure_{event['time_ns']}.json", event)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        plan = build_plan()
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "cells": len(plan["inventory"])}))
        return 0
    print(json.dumps(run(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
