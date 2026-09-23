"""Simulator interventions for absent-object visual-language conflicts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

ObjectPresenceCondition = Literal["present", "absent"]


@dataclass(frozen=True)
class ObjectConflictTask:
    """A LIBERO Object task paired with an object that can be hidden."""

    task_id: int
    target_object: str
    hidden_object: str
    native_prompt: str
    conflict_prompt: str


_OBJECT_CONFLICT_TASKS = {
    task.task_id: task
    for task in (
        ObjectConflictTask(
            task_id=4,
            target_object="ketchup",
            hidden_object="milk",
            native_prompt="pick up the ketchup and place it in the basket",
            conflict_prompt="pick up the milk and place it in the basket",
        ),
        ObjectConflictTask(
            task_id=7,
            target_object="milk",
            hidden_object="cream_cheese",
            native_prompt="pick up the milk and place it in the basket",
            conflict_prompt="pick up the cream cheese and place it in the basket",
        ),
    )
}


def object_conflict_task(task_id: int) -> ObjectConflictTask:
    """Return a validated absent-object conflict specification."""

    try:
        return _OBJECT_CONFLICT_TASKS[task_id]
    except KeyError as error:
        supported = ", ".join(str(value) for value in sorted(_OBJECT_CONFLICT_TASKS))
        raise ValueError(
            f"Task {task_id} does not have a registered object conflict; "
            f"supported task ids are {supported}"
        ) from error


@dataclass(frozen=True)
class HiddenObjectIntervention:
    """Free-joint pose before and after hiding one named scene object."""

    object_name: str
    pose_before: tuple[float, ...]
    pose_after: tuple[float, ...]


def apply_hidden_object_scene(
    environment: Any,
    *,
    object_name: str,
    settle_steps: int = 5,
    hidden_position: tuple[float, float, float] = (3.0, 3.0, 1.0),
) -> tuple[dict[str, Any], HiddenObjectIntervention]:
    """Move one free-joint object out of view and return the settled observation."""

    return apply_object_presence_scene(
        environment,
        object_name=object_name,
        condition="absent",
        settle_steps=settle_steps,
        hidden_position=hidden_position,
    )


def apply_object_presence_scene(
    environment: Any,
    *,
    object_name: str,
    condition: ObjectPresenceCondition,
    settle_steps: int = 5,
    hidden_position: tuple[float, float, float] = (3.0, 3.0, 1.0),
) -> tuple[dict[str, Any], HiddenObjectIntervention]:
    """Return a balanced present/absent scene pair for one free-joint object.

    Both conditions execute the same number of no-op settling steps.  This prevents
    a presence monitor from exploiting robot-state or simulator-time differences
    introduced by the intervention procedure itself.
    """

    if settle_steps <= 0:
        raise ValueError("Scene intervention settle steps must be positive")
    if not object_name:
        raise ValueError("Object name must be nonempty")
    if condition not in {"present", "absent"}:
        raise ValueError(f"Unknown object-presence condition {condition!r}")
    domain = environment._env.env
    object_key = f"{object_name}_1"
    try:
        scene_object = domain.objects_dict[object_key]
    except KeyError as error:
        raise ValueError(f"Scene does not contain object {object_key!r}") from error
    object_joint = scene_object.joints[-1]
    pose_before = np.asarray(domain.sim.data.get_joint_qpos(object_joint)).copy()
    if pose_before.shape != (7,):
        raise ValueError("Free-joint object pose must have seven values")
    pose_after = np.asarray(pose_before, dtype=np.float64).copy()
    if condition == "absent":
        pose_after[:3] = hidden_position
    domain.sim.data.set_joint_qpos(object_joint, pose_after)
    domain.sim.forward()
    raw_observation = None
    for _ in range(settle_steps):
        raw_observation, _, _, _ = environment._env.step([0, 0, 0, 0, 0, 0, -1])
    if raw_observation is None:
        raise AssertionError("Scene intervention did not generate an observation")
    observation = cast(dict[str, Any], environment._format_raw_obs(raw_observation))
    return observation, HiddenObjectIntervention(
        object_name=object_name,
        pose_before=tuple(float(value) for value in pose_before),
        pose_after=tuple(float(value) for value in pose_after),
    )
