"""End-to-end smoke tests for the optional SmolVLA and LIBERO stack."""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from causal_vla.integrations.action_atlas import activate_action_atlas
from causal_vla.interventions import ActivationCapture, ActivationReplace


@dataclass(frozen=True)
class IdentityPatchReport:
    """Measured integrity of capture followed by identity replacement."""

    task: str
    pathway: str
    layer: int
    capture_calls: int
    action_shape: tuple[int, ...]
    max_abs_error: float
    exact_match: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report."""

        return dict[str, object](asdict(self))


def _format_libero_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw robosuite observation to LeRobot's LIBERO wrapper schema."""

    has_pixels = "pixels" in observation
    has_robot_state = "robot_state" in observation
    if has_pixels or has_robot_state:
        if not has_pixels or not has_robot_state:
            raise ValueError("LIBERO observation must provide both pixels and robot_state")
        if not isinstance(observation["pixels"], dict) or not isinstance(
            observation["robot_state"], dict
        ):
            raise TypeError("LIBERO pixels and robot_state must be mappings")
        return dict(observation)

    camera_names = {
        "agentview_image": "image",
        "robot0_eye_in_hand_image": "image2",
    }
    required = {
        *camera_names,
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    }
    missing = sorted(required - set(observation))
    if missing:
        raise ValueError(f"Raw LIBERO observation is missing required fields: {missing}")
    eef: dict[str, Any] = {
        "pos": observation["robot0_eef_pos"],
        "quat": observation["robot0_eef_quat"],
    }
    gripper: dict[str, Any] = {"qpos": observation["robot0_gripper_qpos"]}
    joints: dict[str, Any] = {}
    for raw_key, group, key in (
        ("robot0_gripper_qvel", gripper, "qvel"),
        ("robot0_joint_pos", joints, "pos"),
        ("robot0_joint_vel", joints, "vel"),
    ):
        if raw_key in observation:
            group[key] = observation[raw_key]
    robot_state: dict[str, Any] = {"eef": eef, "gripper": gripper}
    if joints:
        robot_state["joints"] = joints
    return {
        "pixels": {
            output_name: observation[input_name] for input_name, output_name in camera_names.items()
        },
        "robot_state": robot_state,
    }


def prepare_libero_batch(
    adapter: Any,
    observation: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    """Convert one nested LeRobot LIBERO observation into a policy batch."""

    utils = importlib.import_module("lerobot.envs.utils")
    adapters = importlib.import_module("experiments.model_adapters")
    batch: dict[str, Any] = utils.preprocess_observation(_format_libero_observation(observation))

    robot_state = batch.get("observation.robot_state")
    if robot_state is not None:
        for group in robot_state.values():
            for key, value in group.items():
                if isinstance(value, Tensor) and value.ndim <= 2:
                    group[key] = value.unsqueeze(0)

    batch["task"] = [task]
    batch = adapter.env_preprocessor(batch)
    batch = adapter.preprocessor(batch)
    adapters._ensure_attended_language(batch)
    return batch


def fixed_noise(policy: Any, seed: int, device: torch.device) -> Tensor:
    """Create device-independent, reproducible flow-matching noise."""

    shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device)


def run_smolvla_identity_smoke(
    checkpoint: str | Path,
    *,
    action_atlas_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    device: str = "mps",
    suite: str = "libero_goal",
    task_id: int = 0,
    episode_index: int = 0,
    resolution: int = 256,
    pathway: str = "expert",
    layer: int = 16,
    seed: int = 7,
) -> IdentityPatchReport:
    """Run an identity activation intervention on one real SmolVLA action chunk."""

    activate_action_atlas(action_atlas_root, state_dir)
    adapters = importlib.import_module("experiments.model_adapters")
    adapter = adapters.SmolVLAAdapter()
    policy = adapter.load_model(str(Path(checkpoint).resolve()), device=device)
    layer_groups = adapter.get_layer_groups()
    if pathway not in layer_groups:
        raise ValueError(f"Unknown pathway {pathway!r}; choose from {sorted(layer_groups)}")
    if not 0 <= layer < len(layer_groups[pathway]):
        raise ValueError(f"Layer {layer} is outside [0, {len(layer_groups[pathway]) - 1}]")

    environment, task, _ = adapter.create_env(
        task_id,
        suite=suite,
        resolution=resolution,
        episode_index=episode_index,
    )
    try:
        observation, _ = environment.reset()
        batch = prepare_libero_batch(adapter, observation, task)
        selected_layer = layer_groups[pathway][layer]
        noise = fixed_noise(policy, seed, torch.device(device))

        capture = ActivationCapture()
        handle = selected_layer.register_forward_hook(capture)
        try:
            policy.reset()
            with torch.inference_mode():
                baseline = policy.predict_action_chunk(batch, noise=noise.clone())
        finally:
            handle.remove()
        if not capture.records:
            raise RuntimeError(f"The {pathway} layer {layer} hook was never called")

        replacement = ActivationReplace(capture.records)
        handle = selected_layer.register_forward_hook(replacement)
        try:
            policy.reset()
            with torch.inference_mode():
                replayed = policy.predict_action_chunk(batch, noise=noise.clone())
        finally:
            handle.remove()
        replacement.assert_consumed()

        max_abs_error = float((baseline - replayed).abs().max().item())
        return IdentityPatchReport(
            task=task,
            pathway=pathway,
            layer=layer,
            capture_calls=len(capture.records),
            action_shape=tuple(baseline.shape),
            max_abs_error=max_abs_error,
            exact_match=torch.equal(baseline, replayed),
        )
    finally:
        environment.close()
