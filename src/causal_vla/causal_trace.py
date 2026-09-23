"""Pure utilities for layer-token-timestep causal tracing of prefix KV caches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TypeAlias, cast

import numpy as np
import torch
from torch import Tensor

KVEntry: TypeAlias = dict[str, Tensor]
KVCache: TypeAlias = dict[int, KVEntry]


@dataclass(frozen=True)
class TokenPatch:
    """One named set of absolute prefix-token positions."""

    name: str
    positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("A token patch requires a name")
        if not self.positions or len(set(self.positions)) != len(self.positions):
            raise ValueError("Token-patch positions must be nonempty and unique")


@dataclass(frozen=True)
class RecoveryScore:
    """How one intervention moves an output from conflict toward correct."""

    directional_recovery: float
    mse_recovery: float
    effect_l2: float
    conflict_correct_l2: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-safe score."""

        return cast(dict[str, float], asdict(self))


def recovery_score(conflict: Tensor, correct: Tensor, patched: Tensor) -> RecoveryScore:
    """Score a patch relative to the conflict-to-correct output difference.

    Directional recovery is the signed projection of the patch effect onto the
    correct-minus-conflict vector. MSE recovery is one minus the residual-error
    ratio. Both equal one for exact restoration and zero for an identity patch.
    """

    if conflict.shape != correct.shape or conflict.shape != patched.shape:
        raise ValueError("Conflict, correct, and patched outputs must have equal shapes")
    base = conflict.detach().cpu().double().reshape(-1)
    target = correct.detach().cpu().double().reshape(-1)
    intervention = patched.detach().cpu().double().reshape(-1)
    delta = target - base
    denominator = torch.dot(delta, delta)
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 1e-20:
        raise ValueError("Conflict and correct outputs are indistinguishable")
    effect = intervention - base
    residual = intervention - target
    directional = torch.dot(effect, delta) / denominator
    mse_recovery = 1.0 - torch.dot(residual, residual) / denominator
    return RecoveryScore(
        directional_recovery=float(directional),
        mse_recovery=float(mse_recovery),
        effect_l2=float(torch.linalg.vector_norm(effect)),
        conflict_correct_l2=float(torch.sqrt(denominator)),
    )


def validate_kv_pair(conflict: Mapping[int, KVEntry], correct: Mapping[int, KVEntry]) -> None:
    """Require aligned, finite prefix caches from equal-length prompts."""

    if not conflict or set(conflict) != set(correct):
        raise ValueError("Conflict and correct caches must contain the same nonempty layers")
    for layer in conflict:
        if set(conflict[layer]) != {"key_states", "value_states"}:
            raise ValueError(f"Layer {layer} has an unexpected KV-cache schema")
        if set(correct[layer]) != {"key_states", "value_states"}:
            raise ValueError(f"Layer {layer} has an unexpected correct-cache schema")
        for component in ("key_states", "value_states"):
            base = conflict[layer][component]
            source = correct[layer][component]
            if base.ndim != 4 or base.shape != source.shape:
                raise ValueError(
                    f"Layer {layer} {component} caches must be aligned rank-four tensors"
                )
            if base.shape[0] != 1:
                raise ValueError("Causal tracing currently requires one observation per cache")
            if not bool(torch.isfinite(base).all()) or not bool(torch.isfinite(source).all()):
                raise ValueError("KV caches must contain only finite values")


def batched_token_patch(
    conflict: Mapping[int, KVEntry],
    correct: Mapping[int, KVEntry],
    *,
    layer: int,
    patches: Sequence[TokenPatch],
    components: tuple[str, ...] = ("key_states", "value_states"),
    scale: float = 1.0,
) -> KVCache:
    """Create one batched cache whose rows contain distinct oracle token patches.

    Unmodified layers use expanded views. Only the selected layer is materialized
    per patch, so self-attention cache concatenation sees aligned batch dimensions.
    """

    validate_kv_pair(conflict, correct)
    if layer not in conflict:
        raise ValueError(f"Layer {layer} is absent from the cache")
    if not patches or len({patch.name for patch in patches}) != len(patches):
        raise ValueError("Patches must be nonempty and have unique names")
    if not components or len(set(components)) != len(components):
        raise ValueError("Components must be nonempty and unique")
    if not np.isfinite(scale):
        raise ValueError("Patch scale must be finite")
    unknown_components = set(components) - {"key_states", "value_states"}
    if unknown_components:
        raise ValueError(f"Unknown KV components: {sorted(unknown_components)}")

    sequence_length = conflict[layer]["key_states"].shape[1]
    for patch in patches:
        if any(position < 0 or position >= sequence_length for position in patch.positions):
            raise ValueError(
                f"Patch {patch.name!r} addresses positions outside [0, {sequence_length - 1}]"
            )

    batch_size = len(patches)
    result: KVCache = {
        index: {
            name: tensor.expand(batch_size, *tensor.shape[1:]) for name, tensor in entry.items()
        }
        for index, entry in conflict.items()
    }
    selected: KVEntry = {}
    for component in ("key_states", "value_states"):
        base = conflict[layer][component]
        updated = base.expand(batch_size, *base.shape[1:]).clone()
        if component in components:
            source = correct[layer][component]
            for row, patch in enumerate(patches):
                positions = list(patch.positions)
                base_tokens = base[0, positions]
                updated[row, positions] = base_tokens + scale * (source[0, positions] - base_tokens)
        selected[component] = updated
    result[layer] = selected
    return result


def aggregate_patch_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Aggregate episode-level trace rows into stable locus summaries."""

    grouped: dict[tuple[int, int, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (
            int(cast(int, row["layer"])),
            int(cast(int, row["denoising_step"])),
            str(row["token_patch"]),
            str(row["components"]),
        )
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, object]] = []
    for (layer, timestep, token_patch, components), members in grouped.items():
        executed = [float(cast(float, member["executed_mse_recovery"])) for member in members]
        chunk = [float(cast(float, member["chunk_mse_recovery"])) for member in members]
        summaries.append(
            {
                "layer": layer,
                "denoising_step": timestep,
                "token_patch": token_patch,
                "components": components,
                "episodes": len(members),
                "executed_mse_recovery_mean": sum(executed) / len(executed),
                "executed_mse_recovery_min": min(executed),
                "chunk_mse_recovery_mean": sum(chunk) / len(chunk),
                "chunk_mse_recovery_min": min(chunk),
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            -float(cast(float, item["executed_mse_recovery_min"])),
            -float(cast(float, item["executed_mse_recovery_mean"])),
            int(cast(int, item["layer"])),
            int(cast(int, item["denoising_step"])),
            str(item["token_patch"]),
        ),
    )
