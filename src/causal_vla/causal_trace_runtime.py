"""SmolVLA runtime for causal tracing across cache layer, token, and flow time."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from causal_vla.causal_trace import (
    KVCache,
    TokenPatch,
    aggregate_patch_rows,
    batched_token_patch,
    recovery_score,
    validate_kv_pair,
)
from causal_vla.integrations.action_atlas import activate_action_atlas
from causal_vla.smoke import fixed_noise, prepare_libero_batch
from causal_vla.steering import attended_language_token_ids


@dataclass(frozen=True)
class PrefixSnapshot:
    """One prompt's prefix cache and auditable token layout."""

    pad_masks: Tensor
    cache: KVCache
    image_lengths: tuple[int, ...]
    language_token_ids: tuple[int, ...]
    language_tokens: tuple[str, ...]
    language_slots: int
    state_positions: tuple[int, ...]

    @property
    def sequence_length(self) -> int:
        """Return the full image-language-state prefix length."""

        return int(self.pad_masks.shape[1])


@dataclass(frozen=True)
class CausalTraceReport:
    """Layer-token-timestep oracle localization on untouched prompt inputs."""

    schema_version: int
    checkpoint: str
    suite: str
    task_id: int
    correct_prompt: str
    conflict_prompt: str
    episode_indices: tuple[int, ...]
    noise_seed: int
    simulator_seed: int
    layers: tuple[int, ...]
    denoising_steps: tuple[int, ...]
    components: tuple[str, ...]
    patch_granularity: str
    patch_source: str
    prefix_layout: dict[str, object]
    rows: tuple[dict[str, object], ...]
    ranked_loci: tuple[dict[str, object], ...]
    correct_prompt_runtime_forwards: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report."""

        payload = cast(dict[str, object], asdict(self))
        payload["rows"] = list(self.rows)
        payload["ranked_loci"] = list(self.ranked_loci)
        return payload


def prefix_snapshot(policy: Any, batch: dict[str, Any]) -> PrefixSnapshot:
    images, image_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    language_tokens = batch["observation.language.tokens"]
    language_masks = batch["observation.language.attention_mask"]
    if not isinstance(language_tokens, Tensor) or not isinstance(language_masks, Tensor):
        raise TypeError("SmolVLA language inputs must be tensors")
    model = policy.model
    image_lengths: list[int] = []

    def capture_image_length(_module: Any, _inputs: tuple[object, ...], output: Tensor) -> None:
        if not isinstance(output, Tensor) or output.ndim < 3:
            raise TypeError("SmolVLA connector must emit an image-token tensor")
        image_lengths.append(int(output.shape[1]))

    connector = model.vlm_with_expert.get_vlm_model().connector
    connector_handle = connector.register_forward_hook(capture_image_length)
    with torch.inference_mode():
        try:
            prefix_embeds, pad_masks, attention_masks = model.embed_prefix(
                images,
                image_masks,
                language_tokens,
                language_masks,
                state=state,
            )
        finally:
            connector_handle.remove()
        attention_2d = importlib.import_module(
            "lerobot.policies.smolvla.modeling_smolvla"
        ).make_att_2d_masks(pad_masks, attention_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        _, cache = model.vlm_with_expert.forward(
            attention_mask=attention_2d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embeds, None],
            use_cache=model.config.use_cache,
            fill_kv_cache=True,
        )
    if not isinstance(cache, dict):
        raise TypeError("SmolVLA did not return a dictionary KV cache")
    typed_cache = cast(KVCache, cache)
    token_ids = attended_language_token_ids(batch)
    tokenizer = model.vlm_with_expert.processor.tokenizer
    decoded = tuple(str(tokenizer.decode([token])) for token in token_ids)
    language_slots = int(language_tokens.shape[1])
    if len(image_lengths) != len(images):
        raise RuntimeError("Image connector call count does not match camera count")
    frozen_image_lengths = tuple(image_lengths)
    state_start = sum(frozen_image_lengths) + language_slots
    state_positions = tuple(range(state_start, int(prefix_embeds.shape[1])))
    return PrefixSnapshot(
        pad_masks=pad_masks,
        cache=typed_cache,
        image_lengths=frozen_image_lengths,
        language_token_ids=token_ids,
        language_tokens=decoded,
        language_slots=language_slots,
        state_positions=state_positions,
    )


def aligned_neutral_language_batch(
    conflict_batch: dict[str, Any], neutral_batch: dict[str, Any]
) -> dict[str, Any]:
    """Keep the conflict sequence layout but attend only one task-invariant neutral token."""

    token_key = "observation.language.tokens"
    mask_key = "observation.language.attention_mask"
    conflict_tokens = conflict_batch.get(token_key)
    conflict_masks = conflict_batch.get(mask_key)
    neutral_tokens = neutral_batch.get(token_key)
    neutral_masks = neutral_batch.get(mask_key)
    if not all(
        isinstance(item, Tensor)
        for item in (conflict_tokens, conflict_masks, neutral_tokens, neutral_masks)
    ):
        raise TypeError("Aligned neutralization requires tensor language inputs")
    typed_conflict_tokens = cast(Tensor, conflict_tokens)
    typed_conflict_masks = cast(Tensor, conflict_masks)
    typed_neutral_tokens = cast(Tensor, neutral_tokens)
    typed_neutral_masks = cast(Tensor, neutral_masks)
    attended = typed_neutral_tokens[typed_neutral_masks.bool()]
    if attended.numel() != 1:
        raise ValueError("The neutral prompt must contain exactly one attended token")
    tokens = torch.zeros_like(typed_conflict_tokens)
    masks = torch.zeros_like(typed_conflict_masks)
    tokens[..., 0] = attended[0].to(tokens.device, tokens.dtype)
    masks[..., 0] = 1
    result = dict(conflict_batch)
    result[token_key] = tokens
    result[mask_key] = masks
    return result


def _token_patches(
    conflict: PrefixSnapshot,
    correct: PrefixSnapshot,
    *,
    granularity: str,
) -> tuple[TokenPatch, ...]:
    if conflict.image_lengths != correct.image_lengths:
        raise ValueError("Correct and conflict image layouts differ")
    if conflict.language_slots != correct.language_slots:
        raise ValueError("Correct and conflict language lengths differ")
    if conflict.sequence_length != correct.sequence_length:
        raise ValueError("Correct and conflict prefix lengths differ")
    if len(conflict.language_token_ids) != len(correct.language_token_ids):
        raise ValueError("Correct and conflict attended token counts differ")

    image_patches: list[TokenPatch] = []
    offset = 0
    for index, length in enumerate(conflict.image_lengths):
        image_patches.append(TokenPatch(f"image_{index}", tuple(range(offset, offset + length))))
        offset += length
    language_start = offset
    attended_count = len(conflict.language_token_ids)
    changed = tuple(
        language_start + index
        for index, (base, target) in enumerate(
            zip(conflict.language_token_ids, correct.language_token_ids, strict=True)
        )
        if base != target
    )
    unchanged = tuple(
        language_start + index
        for index, (base, target) in enumerate(
            zip(conflict.language_token_ids, correct.language_token_ids, strict=True)
        )
        if base == target
    )
    if not changed:
        raise ValueError("Correct and conflict prompts have no changed language token")

    if granularity == "groups":
        patches = [
            *image_patches,
            TokenPatch("language_changed", changed),
            TokenPatch("language_unchanged", unchanged),
            TokenPatch(
                "language_all",
                tuple(range(language_start, language_start + attended_count)),
            ),
            TokenPatch("state", conflict.state_positions),
            TokenPatch("all_prefix", tuple(range(conflict.sequence_length))),
        ]
        return tuple(patches)
    if granularity == "language_tokens":
        return tuple(
            TokenPatch(
                f"language_{index}:{conflict.language_tokens[index]!r}",
                (language_start + index,),
            )
            for index in range(attended_count)
        )
    if granularity == "all_tokens":
        labels: list[str] = []
        for image_index, length in enumerate(conflict.image_lengths):
            labels.extend(f"image_{image_index}:{index}" for index in range(length))
        labels.extend(
            f"language_{index}:{token!r}" for index, token in enumerate(conflict.language_tokens)
        )
        labels.extend(f"state:{index}" for index in range(len(conflict.state_positions)))
        if len(labels) != conflict.sequence_length:
            raise RuntimeError("Prefix labels do not cover the full sequence")
        return tuple(TokenPatch(label, (index,)) for index, label in enumerate(labels))
    raise ValueError("Patch granularity must be groups, language_tokens, or all_tokens")


def _velocity_trace(model: Any, snapshot: PrefixSnapshot, noise: Tensor) -> tuple[Tensor, ...]:
    velocities: list[Tensor] = []
    states: list[Tensor] = [noise.clone()]
    x_t = noise.clone()
    dt = -1.0 / model.config.num_steps
    with torch.inference_mode():
        for step in range(model.config.num_steps):
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
            velocities.append(velocity.detach())
            x_t = x_t + dt * velocity
            states.append(x_t.detach())
    return tuple(states + velocities)


def run_causal_trace(
    checkpoint: str | Path,
    *,
    correct_prompt: str,
    conflict_prompt: str,
    episode_indices: tuple[int, ...],
    layers: tuple[int, ...],
    denoising_steps: tuple[int, ...],
    patch_granularity: str = "groups",
    patch_source: str = "correct",
    components: tuple[str, ...] = ("key_states", "value_states"),
    suite: str = "libero_spatial",
    task_id: int = 7,
    resolution: int = 256,
    action_atlas_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    device: str = "mps",
    noise_seed: int = 107,
    simulator_seed: int = 42,
    progress: Callable[[int, int, int], None] | None = None,
) -> CausalTraceReport:
    """Run an oracle cache-patching screen on initial states from multiple scenes."""

    if not episode_indices or len(set(episode_indices)) != len(episode_indices):
        raise ValueError("Episode indices must be nonempty and unique")
    if not layers or tuple(sorted(set(layers))) != layers:
        raise ValueError("Layers must be nonempty, unique, and sorted")
    if not denoising_steps or tuple(sorted(set(denoising_steps))) != denoising_steps:
        raise ValueError("Denoising steps must be nonempty, unique, and sorted")
    if patch_source not in {"correct", "neutral"}:
        raise ValueError("Patch source must be correct or neutral")
    activate_action_atlas(action_atlas_root, state_dir)
    adapter = importlib.import_module("experiments.model_adapters").SmolVLAAdapter()
    checkpoint_path = str(Path(checkpoint).resolve())
    policy = adapter.load_model(checkpoint_path, device=device)
    model = policy.model
    layer_count = int(model.vlm_with_expert.num_vlm_layers)
    if any(layer < 0 or layer >= layer_count for layer in layers):
        raise ValueError(f"Layers must lie within [0, {layer_count - 1}]")
    if any(step < 0 or step >= model.config.num_steps for step in denoising_steps):
        raise ValueError(f"Denoising steps must lie within [0, {model.config.num_steps - 1}]")

    rows: list[dict[str, object]] = []
    reference_layout: dict[str, object] | None = None
    action_dim = int(policy.config.action_feature.shape[0])
    torch_device = torch.device(device)
    for episode_index in episode_indices:
        environment, native_prompt, _ = adapter.create_env(
            task_id,
            suite=suite,
            resolution=resolution,
            episode_index=episode_index,
        )
        try:
            if native_prompt != correct_prompt:
                raise ValueError("Correct prompt must exactly match the native task instruction")
            observation, _ = environment.reset(seed=simulator_seed + episode_index)
            conflict_batch = prepare_libero_batch(adapter, observation, conflict_prompt)
            conflict_snapshot = prefix_snapshot(policy, conflict_batch)
            correct_snapshot = prefix_snapshot(
                policy, prepare_libero_batch(adapter, observation, correct_prompt)
            )
            if patch_source == "correct":
                source_snapshot = correct_snapshot
            else:
                neutral_batch = aligned_neutral_language_batch(
                    conflict_batch,
                    prepare_libero_batch(adapter, observation, ""),
                )
                source_snapshot = prefix_snapshot(policy, neutral_batch)
        finally:
            environment.close()
        validate_kv_pair(conflict_snapshot.cache, correct_snapshot.cache)
        validate_kv_pair(conflict_snapshot.cache, source_snapshot.cache)
        patches = _token_patches(
            conflict_snapshot,
            correct_snapshot,
            granularity=patch_granularity,
        )
        layout = {
            "sequence_length": conflict_snapshot.sequence_length,
            "image_lengths": list(conflict_snapshot.image_lengths),
            "language_start": sum(conflict_snapshot.image_lengths),
            "language_token_ids": list(conflict_snapshot.language_token_ids),
            "language_tokens": list(conflict_snapshot.language_tokens),
            "state_positions": list(conflict_snapshot.state_positions),
            "patches": [
                {"name": patch.name, "positions": list(patch.positions)} for patch in patches
            ],
        }
        if reference_layout is not None and layout != reference_layout:
            raise RuntimeError("Prefix layout changed across scene initializations")
        reference_layout = layout

        noise = fixed_noise(policy, noise_seed + episode_index, torch_device)
        conflict_trace = _velocity_trace(model, conflict_snapshot, noise)
        states = conflict_trace[: model.config.num_steps + 1]
        conflict_velocities = conflict_trace[model.config.num_steps + 1 :]
        dt = -1.0 / model.config.num_steps
        for denoising_step in denoising_steps:
            x_t = states[denoising_step]
            timestep = torch.full(
                (1,),
                1.0 + denoising_step * dt,
                dtype=torch.float32,
                device=x_t.device,
            )
            with torch.inference_mode():
                correct_velocity = model.denoise_step(
                    prefix_pad_masks=correct_snapshot.pad_masks,
                    past_key_values=correct_snapshot.cache,
                    x_t=x_t,
                    timestep=timestep,
                ).detach()
            conflict_velocity = conflict_velocities[denoising_step]
            for layer in layers:
                patched_cache = batched_token_patch(
                    conflict_snapshot.cache,
                    source_snapshot.cache,
                    layer=layer,
                    patches=patches,
                    components=components,
                )
                batch_size = len(patches)
                with torch.inference_mode():
                    patched_velocity = model.denoise_step(
                        prefix_pad_masks=conflict_snapshot.pad_masks.expand(batch_size, -1),
                        past_key_values=patched_cache,
                        x_t=x_t.expand(batch_size, -1, -1),
                        timestep=timestep.expand(batch_size),
                    ).detach()
                for patch_index, patch in enumerate(patches):
                    executed = recovery_score(
                        conflict_velocity[:, :1, :action_dim],
                        correct_velocity[:, :1, :action_dim],
                        patched_velocity[patch_index : patch_index + 1, :1, :action_dim],
                    )
                    chunk = recovery_score(
                        conflict_velocity[..., :action_dim],
                        correct_velocity[..., :action_dim],
                        patched_velocity[patch_index : patch_index + 1, ..., :action_dim],
                    )
                    rows.append(
                        {
                            "episode_index": episode_index,
                            "layer": layer,
                            "denoising_step": denoising_step,
                            "time": float(timestep[0]),
                            "token_patch": patch.name,
                            "positions": list(patch.positions),
                            "components": "+".join(components),
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
                    progress(episode_index, denoising_step, layer)

    if reference_layout is None:
        raise RuntimeError("No prefix layout was recorded")
    ranked = aggregate_patch_rows(rows)
    return CausalTraceReport(
        schema_version=1,
        checkpoint=checkpoint_path,
        suite=suite,
        task_id=task_id,
        correct_prompt=correct_prompt,
        conflict_prompt=conflict_prompt,
        episode_indices=episode_indices,
        noise_seed=noise_seed,
        simulator_seed=simulator_seed,
        layers=layers,
        denoising_steps=denoising_steps,
        components=components,
        patch_granularity=patch_granularity,
        patch_source=patch_source,
        prefix_layout=reference_layout,
        rows=tuple(rows),
        ranked_loci=tuple(ranked),
        correct_prompt_runtime_forwards=len(episode_indices) * len(denoising_steps),
    )
