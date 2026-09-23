"""Action-expert layer, token, and flow-time causal tracing for SmolVLA."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from causal_vla.causal_trace import (
    KVCache,
    TokenPatch,
    aggregate_patch_rows,
    recovery_score,
)
from causal_vla.causal_trace_runtime import PrefixSnapshot, prefix_snapshot
from causal_vla.integrations.action_atlas import activate_action_atlas
from causal_vla.interventions import InputActivationCapture
from causal_vla.scene_intervention import (
    SceneCondition,
    apply_unique_bowl_scene,
    spatial_support_task,
)
from causal_vla.smoke import fixed_noise, prepare_libero_batch


@dataclass(frozen=True)
class ExpertTraceReport:
    """Oracle causal effects within the action expert at matched flow states."""

    schema_version: int
    checkpoint: str
    suite: str
    task_id: int
    correct_prompt: str
    conflict_prompt: str
    episode_indices: tuple[int, ...]
    expert_layers: tuple[int, ...]
    denoising_steps: tuple[int, ...]
    action_token_patches: tuple[dict[str, object], ...]
    noise_seed: int
    simulator_seed: int
    scene_condition: SceneCondition | None
    native_prompt_role: str
    rows: tuple[dict[str, object], ...]
    ranked_loci: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe report."""

        payload = cast(dict[str, object], asdict(self))
        payload["action_token_patches"] = list(self.action_token_patches)
        payload["rows"] = list(self.rows)
        payload["ranked_loci"] = list(self.ranked_loci)
        return payload


class BatchedPostResidualPatch:
    """Patch a distinct post-layer token set in each batch row."""

    def __init__(self, source: Tensor, patches: Sequence[TokenPatch]):
        if source.ndim != 3 or source.shape[0] != 1:
            raise ValueError("Source residual must have shape [1, action_tokens, hidden]")
        if not patches:
            raise ValueError("At least one action-token patch is required")
        self.source = source.detach()
        self.patches = tuple(patches)
        self.residual_input: Tensor | None = None
        self.calls = 0

    def capture_residual(
        self, _module: nn.Module, inputs: tuple[object, ...]
    ) -> None:
        if not inputs or not isinstance(inputs[0], Tensor):
            raise TypeError("Expected a tensor residual input")
        if self.residual_input is not None:
            raise RuntimeError("Residual input was captured more than once")
        self.residual_input = inputs[0].detach().clone()

    def patch_mlp(
        self,
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: object,
    ) -> Tensor:
        if not isinstance(output, Tensor):
            raise TypeError("Expected a tensor MLP output")
        if self.residual_input is None:
            raise RuntimeError("MLP ran before its residual input was captured")
        if self.calls:
            raise RuntimeError("Expert MLP was called more than once")
        self.calls += 1
        base_post = self.residual_input + output
        if base_post.shape[0] != len(self.patches):
            raise ValueError("Patch count must match the intervention batch size")
        if base_post.shape[1:] != self.source.shape[1:]:
            raise ValueError("Source and base residual shapes do not align")
        updated = base_post.clone()
        source = self.source.to(device=updated.device, dtype=updated.dtype)
        sequence_length = int(updated.shape[1])
        for row, patch in enumerate(self.patches):
            if any(position < 0 or position >= sequence_length for position in patch.positions):
                raise ValueError("Action-token patch position is outside the sequence")
            positions = list(patch.positions)
            updated[row, positions] = source[0, positions]
        return updated - self.residual_input

    def assert_applied(self) -> None:
        """Require exactly one residual and MLP hook call."""

        if self.residual_input is None or self.calls != 1:
            raise RuntimeError("Expert post-residual patch was not applied exactly once")


def expert_layers(policy: Any) -> tuple[list[Any], nn.Module]:
    """Return SmolVLA action-expert decoder layers and final normalization."""

    expert = policy.model.vlm_with_expert.lm_expert
    return list(expert.layers), cast(nn.Module, expert.norm)


def _post_layer_capture_module(
    layers: list[Any], final_norm: nn.Module, layer_index: int
) -> nn.Module:
    if not 0 <= layer_index < len(layers):
        raise ValueError("Expert layer is outside the decoder")
    if layer_index + 1 < len(layers):
        return cast(nn.Module, layers[layer_index + 1].input_layernorm)
    return final_norm


def capture_expert_post_residuals(
    model: Any,
    snapshot: PrefixSnapshot,
    x_t: Tensor,
    timestep: Tensor,
    *,
    layers: list[Any],
    final_norm: nn.Module,
    layer_indices: tuple[int, ...],
) -> tuple[Tensor, dict[int, Tensor]]:
    captures = {layer: InputActivationCapture() for layer in layer_indices}
    handles = [
        _post_layer_capture_module(layers, final_norm, layer).register_forward_pre_hook(
            captures[layer]
        )
        for layer in layer_indices
    ]
    try:
        with torch.inference_mode():
            velocity = model.denoise_step(
                prefix_pad_masks=snapshot.pad_masks,
                past_key_values=snapshot.cache,
                x_t=x_t,
                timestep=timestep,
            ).detach()
    finally:
        for handle in handles:
            handle.remove()
    residuals: dict[int, Tensor] = {}
    for layer, capture in captures.items():
        if len(capture.records) != 1:
            raise RuntimeError("Expected one expert residual per denoising call")
        residuals[layer] = capture.records[0]
    return velocity, residuals


def _expand_cache(cache: KVCache, batch_size: int) -> KVCache:
    return {
        layer: {
            component: tensor.expand(batch_size, *tensor.shape[1:])
            for component, tensor in entry.items()
        }
        for layer, entry in cache.items()
    }


def patched_expert_velocity(
    model: Any,
    snapshot: PrefixSnapshot,
    x_t: Tensor,
    timestep: Tensor,
    *,
    layers: list[Any],
    expert_layer: int,
    source: Tensor,
    patches: tuple[TokenPatch, ...],
) -> Tensor:
    intervention = BatchedPostResidualPatch(source, patches)
    selected = layers[expert_layer]
    residual_handle = selected.post_attention_layernorm.register_forward_pre_hook(
        intervention.capture_residual
    )
    mlp_handle = selected.mlp.register_forward_hook(intervention.patch_mlp)
    batch_size = len(patches)
    try:
        with torch.inference_mode():
            velocity = model.denoise_step(
                prefix_pad_masks=snapshot.pad_masks.expand(batch_size, -1),
                past_key_values=_expand_cache(snapshot.cache, batch_size),
                x_t=x_t.expand(batch_size, -1, -1),
                timestep=timestep.expand(batch_size),
            ).detach()
    finally:
        residual_handle.remove()
        mlp_handle.remove()
    intervention.assert_applied()
    return cast(Tensor, velocity)


def expert_flow_states(
    model: Any, snapshot: PrefixSnapshot, noise: Tensor
) -> tuple[Tensor, ...]:
    states: list[Tensor] = [noise.clone()]
    x_t = noise.clone()
    dt = -1.0 / int(model.config.num_steps)
    with torch.inference_mode():
        for step in range(int(model.config.num_steps)):
            timestep = torch.full(
                (x_t.shape[0],),
                1.0 + step * dt,
                dtype=torch.float32,
                device=x_t.device,
            )
            velocity = model.denoise_step(
                prefix_pad_masks=snapshot.pad_masks,
                past_key_values=snapshot.cache,
                x_t=x_t,
                timestep=timestep,
            )
            x_t = x_t + dt * velocity
            states.append(x_t.detach())
    return tuple(states)


def run_expert_causal_trace(
    checkpoint: str | Path,
    *,
    correct_prompt: str,
    conflict_prompt: str,
    episode_indices: tuple[int, ...],
    expert_layer_indices: tuple[int, ...],
    denoising_steps: tuple[int, ...],
    suite: str = "libero_spatial",
    task_id: int = 7,
    resolution: int = 256,
    action_atlas_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    device: str = "mps",
    noise_seed: int = 107,
    simulator_seed: int = 42,
    scene_condition: SceneCondition | None = None,
    progress: Callable[[int, int, int], None] | None = None,
) -> ExpertTraceReport:
    """Patch correct expert residuals into conflict forwards at matched flow states."""

    if not episode_indices or len(set(episode_indices)) != len(episode_indices):
        raise ValueError("Episode indices must be nonempty and unique")
    if not expert_layer_indices or tuple(sorted(set(expert_layer_indices))) != expert_layer_indices:
        raise ValueError("Expert layers must be nonempty, unique, and sorted")
    if not denoising_steps or tuple(sorted(set(denoising_steps))) != denoising_steps:
        raise ValueError("Denoising steps must be nonempty, unique, and sorted")
    if set(episode_indices) & set(range(10, 15)):
        raise ValueError("Historical held-out episodes 10--14 are sealed")
    if scene_condition is not None:
        spatial_support_task(task_id)
    activate_action_atlas(action_atlas_root, state_dir)
    adapter = importlib.import_module("experiments.model_adapters").SmolVLAAdapter()
    checkpoint_path = str(Path(checkpoint).resolve())
    policy = adapter.load_model(checkpoint_path, device=device)
    model = policy.model
    layers, final_norm = expert_layers(policy)
    if any(layer < 0 or layer >= len(layers) for layer in expert_layer_indices):
        raise ValueError("An expert layer is outside the decoder")
    if any(step < 0 or step >= model.config.num_steps for step in denoising_steps):
        raise ValueError("A denoising step is outside the flow schedule")
    chunk_size = int(model.config.chunk_size)
    patches = (
        TokenPatch("executed_action", (0,)),
        TokenPatch("first_five_actions", tuple(range(min(5, chunk_size)))),
        TokenPatch("all_actions", tuple(range(chunk_size))),
    )
    rows: list[dict[str, object]] = []
    action_dim = int(policy.config.action_feature.shape[0])
    torch_device = torch.device(device)
    dt = -1.0 / int(model.config.num_steps)
    for episode_index in episode_indices:
        environment, native_prompt, _ = adapter.create_env(
            task_id,
            suite=suite,
            resolution=resolution,
            episode_index=episode_index,
        )
        try:
            expected_native_prompt = (
                conflict_prompt if scene_condition == "conflict" else correct_prompt
            )
            if native_prompt != expected_native_prompt:
                role = "conflict" if scene_condition == "conflict" else "correct"
                raise ValueError(f"The {role} prompt must exactly match the native instruction")
            observation, _ = environment.reset(seed=simulator_seed + episode_index)
            if scene_condition is not None:
                observation, _ = apply_unique_bowl_scene(
                    environment, condition=scene_condition
                )
            conflict_snapshot = prefix_snapshot(
                policy, prepare_libero_batch(adapter, observation, conflict_prompt)
            )
            correct_snapshot = prefix_snapshot(
                policy, prepare_libero_batch(adapter, observation, correct_prompt)
            )
        finally:
            environment.close()
        noise = fixed_noise(policy, noise_seed + episode_index, torch_device)
        states = expert_flow_states(model, conflict_snapshot, noise)
        for step in denoising_steps:
            x_t = states[step]
            timestep = torch.full(
                (1,),
                1.0 + step * dt,
                dtype=torch.float32,
                device=x_t.device,
            )
            with torch.inference_mode():
                conflict_velocity = model.denoise_step(
                    prefix_pad_masks=conflict_snapshot.pad_masks,
                    past_key_values=conflict_snapshot.cache,
                    x_t=x_t,
                    timestep=timestep,
                ).detach()
            correct_velocity, correct_residuals = capture_expert_post_residuals(
                model,
                correct_snapshot,
                x_t,
                timestep,
                layers=layers,
                final_norm=final_norm,
                layer_indices=expert_layer_indices,
            )
            for layer in expert_layer_indices:
                patched = patched_expert_velocity(
                    model,
                    conflict_snapshot,
                    x_t,
                    timestep,
                    layers=layers,
                    expert_layer=layer,
                    source=correct_residuals[layer],
                    patches=patches,
                )
                for patch_index, token_patch in enumerate(patches):
                    executed = recovery_score(
                        conflict_velocity[:, :1, :action_dim],
                        correct_velocity[:, :1, :action_dim],
                        patched[patch_index : patch_index + 1, :1, :action_dim],
                    )
                    chunk = recovery_score(
                        conflict_velocity[..., :action_dim],
                        correct_velocity[..., :action_dim],
                        patched[patch_index : patch_index + 1, ..., :action_dim],
                    )
                    rows.append(
                        {
                            "episode_index": episode_index,
                            "layer": layer,
                            "denoising_step": step,
                            "token_patch": token_patch.name,
                            "positions": list(token_patch.positions),
                            "components": "expert_post_residual",
                            "executed_directional_recovery": executed.directional_recovery,
                            "executed_mse_recovery": executed.mse_recovery,
                            "executed_effect_l2": executed.effect_l2,
                            "executed_conflict_correct_l2": executed.conflict_correct_l2,
                            "chunk_directional_recovery": chunk.directional_recovery,
                            "chunk_mse_recovery": chunk.mse_recovery,
                            "chunk_effect_l2": chunk.effect_l2,
                            "chunk_conflict_correct_l2": chunk.conflict_correct_l2,
                        }
                    )
                if progress is not None:
                    progress(episode_index, step, layer)
    return ExpertTraceReport(
        schema_version=1,
        checkpoint=checkpoint_path,
        suite=suite,
        task_id=task_id,
        correct_prompt=correct_prompt,
        conflict_prompt=conflict_prompt,
        episode_indices=episode_indices,
        expert_layers=expert_layer_indices,
        denoising_steps=denoising_steps,
        action_token_patches=tuple(
            {"name": patch.name, "positions": list(patch.positions)} for patch in patches
        ),
        noise_seed=noise_seed,
        simulator_seed=simulator_seed,
        scene_condition=scene_condition,
        native_prompt_role=("conflict" if scene_condition == "conflict" else "correct"),
        rows=tuple(rows),
        ranked_loci=tuple(aggregate_patch_rows(rows)),
    )
