"""Same-information prompt-correction comparators for cross-task evaluation.

The learned comparator uses only the deployed claim probe and a pre-authorized
task reference prompt. Once the probe declares a conflict, correction latches
for the rest of the episode. The oracle comparator uses the same reference
prompt from the first decision and is an upper-bound viability control.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from causal_vla.causal_trace_runtime import prefix_snapshot
from causal_vla.expert_knn_runtime import _sample_natural
from causal_vla.integrations.action_atlas import activate_action_atlas
from causal_vla.location_monitor_runtime import (
    LocationProbeBank,
    assess_location_claim,
    load_location_probe_bank,
)
from causal_vla.object_intervention import apply_hidden_object_scene, object_conflict_task
from causal_vla.object_monitor_runtime import (
    PresenceProbeBank,
    assess_presence_claim,
    load_presence_probe_bank,
)
from causal_vla.residual_runtime import decode_action
from causal_vla.scene_intervention import apply_unique_bowl_scene, spatial_support_task
from causal_vla.smoke import fixed_noise, prepare_libero_batch

PromptCondition = Literal["monitor_prompt_correction", "oracle_prompt_correction"]


def run_prompt_comparators(
    checkpoint: str | Path,
    monitor_bank_path: str | Path,
    *,
    suite: Literal["libero_object", "libero_spatial"],
    task_id: int,
    episode_indices: tuple[int, ...],
    noise_seed: int,
    conditions: tuple[PromptCondition, ...] = (
        "monitor_prompt_correction",
        "oracle_prompt_correction",
    ),
    device: Literal["cpu", "mps"] = "mps",
    max_steps: int = 220,
    simulator_seed: int = 42,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run matched conflict-scene prompt comparators without outcome filtering."""

    if (
        not episode_indices
        or len(set(episode_indices)) != len(episode_indices)
        or not conditions
        or len(set(conditions)) != len(conditions)
        or not set(conditions).issubset({"monitor_prompt_correction", "oracle_prompt_correction"})
    ):
        raise ValueError("Prompt comparator inventory must contain both fixed arms")
    if max_steps < 1 or (device == "mps" and not torch.backends.mps.is_available()):
        raise ValueError("Invalid prompt-comparator horizon or device")

    activate_action_atlas()
    import importlib

    adapter = importlib.import_module("experiments.model_adapters").SmolVLAAdapter()
    policy = adapter.load_model(str(checkpoint), device=device)
    if suite == "libero_object":
        task = object_conflict_task(task_id)
        native_prompt = task.native_prompt
        conflict_prompt = task.conflict_prompt
        reference_prompt = task.native_prompt
        monitored_object = task.hidden_object
        monitor: PresenceProbeBank | LocationProbeBank = load_presence_probe_bank(monitor_bank_path)
    else:
        task = spatial_support_task(task_id)
        native_prompt = task.native_prompt
        conflict_prompt = task.native_prompt
        reference_prompt = task.alternate_prompt
        monitored_object = "black bowl"
        monitor = load_location_probe_bank(monitor_bank_path)

    outcomes: list[dict[str, Any]] = []
    for episode_index in episode_indices:
        for condition in conditions:
            environment, actual_prompt, _ = adapter.create_env(
                task_id,
                suite=suite,
                resolution=256,
                episode_index=episode_index,
            )
            try:
                if actual_prompt.casefold() != native_prompt.casefold():
                    raise ValueError("Native prompt does not match frozen task definition")
                observation, _ = environment.reset(seed=simulator_seed + episode_index)
                if suite == "libero_spatial":
                    observation, _ = apply_unique_bowl_scene(environment, condition="conflict")
                else:
                    observation, _ = apply_hidden_object_scene(
                        environment, object_name=monitored_object
                    )
                policy.reset()
                latched = condition == "oracle_prompt_correction"
                trigger_step: int | None = 0 if latched else None
                monitor_checks = 0
                monitor_status: str | None = None
                monitor_visual_label: str | None = None
                steps = 0
                success = False
                while steps < max_steps and not success:
                    if not latched and condition == "monitor_prompt_correction":
                        if suite == "libero_object":
                            assert isinstance(monitor, PresenceProbeBank)
                            assessment = assess_presence_claim(
                                policy,
                                adapter,
                                observation,
                                object_name=monitored_object,
                                bank=monitor,
                            )
                        else:
                            assert isinstance(monitor, LocationProbeBank)
                            assessment = assess_location_claim(
                                policy,
                                adapter,
                                observation,
                                object_name=monitored_object,
                                claimed_label=task.instructed_support,
                                bank=monitor,
                            )
                        monitor_checks += 1
                        monitor_status = assessment.status
                        monitor_visual_label = assessment.vision_label
                        if assessment.status == "conflict" and (
                            suite == "libero_object"
                            or assessment.vision_label == task.alternate_support
                        ):
                            latched = True
                            trigger_step = steps

                    prompt = reference_prompt if latched else conflict_prompt
                    snapshot = prefix_snapshot(
                        policy, prepare_libero_batch(adapter, observation, prompt)
                    )
                    noise = fixed_noise(
                        policy,
                        noise_seed + episode_index * 10_000 + steps,
                        torch.device(device),
                    )
                    action = decode_action(adapter, _sample_natural(policy, snapshot, noise)[:, 0])
                    observation, _, terminated, truncated, info = environment.step(action)
                    steps += 1
                    if type(info.get("is_success")) not in (bool, np.bool_):
                        raise ValueError("Simulator did not return a boolean success predicate")
                    success = bool(info["is_success"])
                    if terminated or truncated:
                        break
                result: dict[str, Any] = {
                    "suite": suite,
                    "task_id": task_id,
                    "episode_index": episode_index,
                    "condition": condition,
                    "noise_seed": noise_seed,
                    "success": success,
                    "steps": steps,
                    "trigger_step": trigger_step,
                    "monitor_checks": monitor_checks,
                    "monitor_status": monitor_status,
                    "monitor_visual_label": monitor_visual_label,
                    "reference_prompt_available_offline": True,
                    "reference_prompt_used_online": latched,
                }
                outcomes.append(result)
                if progress is not None:
                    progress(result)
            finally:
                environment.close()
    return outcomes
