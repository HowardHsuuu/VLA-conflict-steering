"""Simulator-level visual interventions for two-location LIBERO Spatial tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

SceneCondition = Literal["aligned", "conflict"]


@dataclass(frozen=True)
class SpatialSupportTask:
    """A LIBERO Spatial task with two identical bowls at distinct locations."""

    task_id: int
    instructed_support: str
    alternate_support: str
    native_prompt: str
    alternate_prompt: str


_SPATIAL_SUPPORT_TASKS = {
    task.task_id: task
    for task in (
        SpatialSupportTask(
            1,
            "ramekin",
            "cookie_box",
            "pick up the black bowl next to the ramekin and place it on the plate",
            "pick up the black bowl next to the cookie box and place it on the plate",
        ),
        SpatialSupportTask(
            3,
            "cookie_box",
            "cabinet",
            "pick up the black bowl on the cookie box and place it on the plate",
            "pick up the black bowl on the wooden cabinet and place it on the plate",
        ),
        SpatialSupportTask(
            5,
            "ramekin",
            "cookie_box",
            "pick up the black bowl on the ramekin and place it on the plate",
            "pick up the black bowl on the cookie box and place it on the plate",
        ),
        SpatialSupportTask(
            6,
            "cookie_box",
            "stove",
            "pick up the black bowl next to the cookie box and place it on the plate",
            "pick up the black bowl on the stove and place it on the plate",
        ),
        SpatialSupportTask(
            7,
            "stove",
            "cabinet",
            "pick up the black bowl on the stove and place it on the plate",
            "pick up the black bowl on the wooden cabinet and place it on the plate",
        ),
        SpatialSupportTask(
            9,
            "cabinet",
            "stove",
            "pick up the black bowl on the wooden cabinet and place it on the plate",
            "pick up the black bowl on the stove and place it on the plate",
        ),
    )
}


def spatial_support_task(task_id: int) -> SpatialSupportTask:
    """Return a validated visual-conflict task specification."""

    try:
        return _SPATIAL_SUPPORT_TASKS[task_id]
    except KeyError as error:
        supported = ", ".join(str(value) for value in sorted(_SPATIAL_SUPPORT_TASKS))
        raise ValueError(
            f"Task {task_id} does not have a registered two-support intervention; "
            f"supported task ids are {supported}"
        ) from error


@dataclass(frozen=True)
class SceneIntervention:
    """Free-joint poses before and after making the target object unique."""

    condition: SceneCondition
    target_pose_before: tuple[float, ...]
    distractor_pose_before: tuple[float, ...]
    target_pose_after: tuple[float, ...]
    distractor_pose_after: tuple[float, ...]


@dataclass(frozen=True)
class DestinationIntervention:
    """Free-joint poses after placing the unique target bowl on the plate."""

    target_pose_before: tuple[float, ...]
    distractor_pose_before: tuple[float, ...]
    plate_pose: tuple[float, ...]
    target_pose_after: tuple[float, ...]
    distractor_pose_after: tuple[float, ...]


def unique_target_poses(
    target_pose: NDArray[np.floating[Any]],
    distractor_pose: NDArray[np.floating[Any]],
    *,
    condition: SceneCondition,
    hidden_position: tuple[float, float, float] = (3.0, 3.0, 1.0),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return target and hidden-distractor poses for one visual condition."""

    if target_pose.shape != (7,) or distractor_pose.shape != (7,):
        raise ValueError("Free-joint object poses must have seven values")
    if condition not in {"aligned", "conflict"}:
        raise ValueError(f"Unknown scene condition {condition!r}")
    active_target = np.asarray(
        target_pose if condition == "aligned" else distractor_pose, dtype=np.float64
    ).copy()
    hidden = np.asarray(distractor_pose, dtype=np.float64).copy()
    hidden[:3] = hidden_position
    return active_target, hidden


def unique_target_destination_poses(
    target_pose: NDArray[np.floating[Any]],
    distractor_pose: NDArray[np.floating[Any]],
    plate_pose: NDArray[np.floating[Any]],
    *,
    bowl_height_offset: float = 0.015,
    hidden_position: tuple[float, float, float] = (3.0, 3.0, 1.0),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Place the unique target bowl stably on the plate and hide its duplicate."""

    if any(pose.shape != (7,) for pose in (target_pose, distractor_pose, plate_pose)):
        raise ValueError("Free-joint object poses must have seven values")
    if bowl_height_offset <= 0:
        raise ValueError("Bowl height offset must be positive")
    active_target = np.asarray(target_pose, dtype=np.float64).copy()
    active_target[:2] = plate_pose[:2]
    active_target[2] = plate_pose[2] + bowl_height_offset
    hidden = np.asarray(distractor_pose, dtype=np.float64).copy()
    hidden[:3] = hidden_position
    return active_target, hidden


def apply_unique_bowl_scene(
    environment: Any,
    *,
    condition: SceneCondition,
    settle_steps: int = 5,
) -> tuple[dict[str, Any], SceneIntervention]:
    """Keep only goal bowl_1 visible, optionally relocating it to bowl_2's support."""

    if settle_steps <= 0:
        raise ValueError("Scene intervention settle steps must be positive")
    domain = environment._env.env
    target = domain.objects_dict["akita_black_bowl_1"]
    distractor = domain.objects_dict["akita_black_bowl_2"]
    target_joint = target.joints[-1]
    distractor_joint = distractor.joints[-1]
    target_before = np.asarray(domain.sim.data.get_joint_qpos(target_joint)).copy()
    distractor_before = np.asarray(domain.sim.data.get_joint_qpos(distractor_joint)).copy()
    target_after, distractor_after = unique_target_poses(
        target_before, distractor_before, condition=condition
    )
    domain.sim.data.set_joint_qpos(target_joint, target_after)
    domain.sim.data.set_joint_qpos(distractor_joint, distractor_after)
    domain.sim.forward()
    raw_observation = None
    for _ in range(settle_steps):
        raw_observation, _, _, _ = environment._env.step([0, 0, 0, 0, 0, 0, -1])
    if raw_observation is None:
        raise AssertionError("Scene intervention did not generate an observation")
    observation = cast(dict[str, Any], environment._format_raw_obs(raw_observation))
    intervention = SceneIntervention(
        condition=condition,
        target_pose_before=tuple(float(value) for value in target_before),
        distractor_pose_before=tuple(float(value) for value in distractor_before),
        target_pose_after=tuple(float(value) for value in target_after),
        distractor_pose_after=tuple(float(value) for value in distractor_after),
    )
    return observation, intervention


def apply_unique_bowl_destination_scene(
    environment: Any,
    *,
    settle_steps: int = 5,
    bowl_height_offset: float = 0.015,
) -> tuple[dict[str, Any], DestinationIntervention]:
    """Render the unique goal bowl resting on the task's destination plate."""

    if settle_steps <= 0:
        raise ValueError("Scene intervention settle steps must be positive")
    domain = environment._env.env
    target = domain.objects_dict["akita_black_bowl_1"]
    distractor = domain.objects_dict["akita_black_bowl_2"]
    plate = domain.objects_dict["plate_1"]
    target_joint = target.joints[-1]
    distractor_joint = distractor.joints[-1]
    plate_joint = plate.joints[-1]
    target_before = np.asarray(domain.sim.data.get_joint_qpos(target_joint)).copy()
    distractor_before = np.asarray(domain.sim.data.get_joint_qpos(distractor_joint)).copy()
    plate_pose = np.asarray(domain.sim.data.get_joint_qpos(plate_joint)).copy()
    target_after, distractor_after = unique_target_destination_poses(
        target_before,
        distractor_before,
        plate_pose,
        bowl_height_offset=bowl_height_offset,
    )
    domain.sim.data.set_joint_qpos(target_joint, target_after)
    domain.sim.data.set_joint_qpos(distractor_joint, distractor_after)
    domain.sim.forward()
    raw_observation = None
    for _ in range(settle_steps):
        raw_observation, _, _, _ = environment._env.step([0, 0, 0, 0, 0, 0, -1])
    if raw_observation is None:
        raise AssertionError("Destination intervention did not generate an observation")
    observation = cast(dict[str, Any], environment._format_raw_obs(raw_observation))
    intervention = DestinationIntervention(
        target_pose_before=tuple(float(value) for value in target_before),
        distractor_pose_before=tuple(float(value) for value in distractor_before),
        plate_pose=tuple(float(value) for value in plate_pose),
        target_pose_after=tuple(float(value) for value in target_after),
        distractor_pose_after=tuple(float(value) for value in distractor_after),
    )
    return observation, intervention
