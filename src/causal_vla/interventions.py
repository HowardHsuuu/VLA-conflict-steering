"""Framework-agnostic PyTorch hooks for causal activation interventions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

ModuleOutput = Tensor | tuple[Tensor, ...]


def _hidden(output: ModuleOutput) -> Tensor:
    if isinstance(output, tuple):
        if not output or not isinstance(output[0], Tensor):
            raise TypeError("Expected a tensor as the first tuple element")
        return output[0]
    if not isinstance(output, Tensor):
        raise TypeError(f"Expected Tensor or tuple[Tensor, ...], got {type(output)!r}")
    return output


def _with_hidden(output: ModuleOutput, hidden: Tensor) -> ModuleOutput:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


@dataclass
class ActivationCapture:
    """Capture module outputs in call order for paired counterfactual replay."""

    to_cpu: bool = True
    dtype: torch.dtype | None = None
    records: list[Tensor] = field(default_factory=list, init=False)

    def __call__(
        self,
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: ModuleOutput,
    ) -> ModuleOutput:
        hidden = _hidden(output).detach()
        if self.dtype is not None:
            hidden = hidden.to(dtype=self.dtype)
        if self.to_cpu:
            hidden = hidden.cpu()
        self.records.append(hidden.clone())
        return output

    def clear(self) -> None:
        """Discard captured tensors without changing hook registration."""

        self.records.clear()


@dataclass
class InputActivationCapture:
    """Capture the leading tensor passed to a module in call order."""

    to_cpu: bool = True
    dtype: torch.dtype | None = None
    records: list[Tensor] = field(default_factory=list, init=False)

    def __call__(
        self,
        _module: nn.Module,
        inputs: tuple[object, ...],
    ) -> None:
        if not inputs or not isinstance(inputs[0], Tensor):
            raise TypeError("Expected a tensor as the first module input")
        hidden = inputs[0].detach()
        if self.dtype is not None:
            hidden = hidden.to(dtype=self.dtype)
        if self.to_cpu:
            hidden = hidden.cpu()
        self.records.append(hidden.clone())

    def clear(self) -> None:
        """Discard captured tensors without changing hook registration."""

        self.records.clear()


@dataclass
class ActivationReplace:
    """Replace outputs sequentially with previously captured activations."""

    replacements: Sequence[Tensor]
    strict: bool = True
    positions: Sequence[int] | None = None
    call_index: int = field(default=0, init=False)

    def __call__(
        self,
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: ModuleOutput,
    ) -> ModuleOutput:
        if self.call_index >= len(self.replacements):
            if self.strict:
                raise RuntimeError("Module was called more times than replacement activations")
            return output

        actual = _hidden(output)
        replacement = self.replacements[self.call_index].to(
            device=actual.device,
            dtype=actual.dtype,
        )
        self.call_index += 1
        if self.positions is None:
            if replacement.shape != actual.shape:
                raise ValueError(
                    "Activation shape mismatch: "
                    f"replacement={replacement.shape}, actual={actual.shape}"
                )
            updated = replacement
        else:
            if actual.ndim < 3:
                raise ValueError("Token-local replacement requires [batch, sequence, hidden]")
            if (
                replacement.ndim != actual.ndim
                or replacement.shape[:-2] != actual.shape[:-2]
                or replacement.shape[-1] != actual.shape[-1]
            ):
                raise ValueError(
                    "Token-local replacement requires matching batch and hidden dimensions: "
                    f"replacement={replacement.shape}, actual={actual.shape}"
                )
            updated = actual.clone()
            indices = list(self.positions)
            updated[..., indices, :] = replacement[..., indices, :]
        return _with_hidden(output, updated)

    def reset(self) -> None:
        """Replay replacements from the first call."""

        self.call_index = 0

    def assert_consumed(self) -> None:
        """Fail when the forward pass used fewer replacements than expected."""

        if self.call_index != len(self.replacements):
            raise RuntimeError(
                f"Consumed {self.call_index} of {len(self.replacements)} replacement activations"
            )


class ProjectedSwap:
    """Swap only a low-rank subspace from source into a base activation.

    For a row-orthonormal projection ``P`` with shape ``[rank, hidden_dim]``:

        h_intervened = h_base + (h_source - h_base) @ P.T @ P
    """

    def __init__(
        self,
        projection: Tensor,
        source: Sequence[Tensor],
        atol: float = 1e-4,
        positions: Sequence[int] | None = None,
    ):
        if projection.ndim != 2:
            raise ValueError("projection must have shape [rank, hidden_dim]")
        gram = projection @ projection.transpose(0, 1)
        identity = torch.eye(projection.shape[0], device=gram.device, dtype=gram.dtype)
        if not torch.allclose(gram, identity, atol=atol, rtol=0):
            raise ValueError("projection rows must be orthonormal")
        self.projection = projection.detach()
        self.source = source
        self.positions = positions
        self.call_index = 0

    def __call__(
        self,
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: ModuleOutput,
    ) -> ModuleOutput:
        if self.call_index >= len(self.source):
            raise RuntimeError("Module was called more times than source activations")
        base = _hidden(output)
        source = self.source[self.call_index].to(device=base.device, dtype=base.dtype)
        projection = self.projection.to(device=base.device, dtype=base.dtype)
        self.call_index += 1
        if self.positions is None:
            if source.shape != base.shape:
                raise ValueError(
                    f"Source shape {source.shape} does not match base shape {base.shape}"
                )
            delta = source - base
            swapped = base + (delta @ projection.transpose(0, 1)) @ projection
        else:
            if base.ndim < 3:
                raise ValueError("Token-local projected swap requires [batch, sequence, hidden]")
            if (
                source.ndim != base.ndim
                or source.shape[:-2] != base.shape[:-2]
                or source.shape[-1] != base.shape[-1]
            ):
                raise ValueError(
                    "Token-local projected swap requires matching batch and hidden dimensions: "
                    f"source={source.shape}, base={base.shape}"
                )
            swapped = base.clone()
            indices = list(self.positions)
            base_tokens = base[..., indices, :]
            delta = source[..., indices, :] - base_tokens
            swapped[..., indices, :] = (
                base_tokens + (delta @ projection.transpose(0, 1)) @ projection
            )
        return _with_hidden(output, swapped)

    def reset(self) -> None:
        """Replay source activations from the first call."""

        self.call_index = 0

    def assert_consumed(self) -> None:
        """Fail when the forward pass used fewer source activations than expected."""

        if self.call_index != len(self.source):
            raise RuntimeError(
                f"Consumed {self.call_index} of {len(self.source)} source activations"
            )


class ResidualStreamSwap:
    """Replace or project a post-layer residual via its MLP residual branch.

    SmolVLA computes ``post = after_attention + mlp(layernorm(after_attention))``
    outside the decoder-layer module. Register :meth:`capture_residual` as a
    pre-hook on ``post_attention_layernorm`` and this object itself as a forward
    hook on the matching MLP. The returned MLP branch makes the resulting
    post-layer residual equal to the desired full or projected source state.
    """

    def __init__(
        self,
        source: Sequence[Tensor],
        *,
        projection: Tensor | None = None,
        positions: Sequence[int] | None = None,
        atol: float = 1e-4,
    ):
        if projection is not None:
            if projection.ndim != 2:
                raise ValueError("projection must have shape [rank, hidden_dim]")
            gram = projection @ projection.transpose(0, 1)
            identity = torch.eye(projection.shape[0], device=gram.device, dtype=gram.dtype)
            if not torch.allclose(gram, identity, atol=atol, rtol=0):
                raise ValueError("projection rows must be orthonormal")
        self.source = source
        self.projection = None if projection is None else projection.detach()
        self.positions = positions
        self.residual_inputs: list[Tensor] = []
        self.call_index = 0

    def capture_residual(
        self,
        _module: nn.Module,
        inputs: tuple[object, ...],
    ) -> None:
        """Record the live residual that will be added to the matching MLP output."""

        if not inputs or not isinstance(inputs[0], Tensor):
            raise TypeError("Expected a tensor as the first module input")
        self.residual_inputs.append(inputs[0])

    def __call__(
        self,
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: ModuleOutput,
    ) -> ModuleOutput:
        if self.call_index >= len(self.source):
            raise RuntimeError("MLP was called more times than source residuals")
        if self.call_index >= len(self.residual_inputs):
            raise RuntimeError("MLP ran before its residual input was captured")
        mlp_output = _hidden(output)
        residual_input = self.residual_inputs[self.call_index]
        source = self.source[self.call_index].to(
            device=mlp_output.device,
            dtype=mlp_output.dtype,
        )
        self.call_index += 1
        base_post = residual_input + mlp_output
        if source.shape != base_post.shape:
            raise ValueError(
                f"Source residual shape {source.shape} does not match base {base_post.shape}"
            )
        if self.positions is None:
            delta = source - base_post
            if self.projection is not None:
                projection = self.projection.to(
                    device=base_post.device,
                    dtype=base_post.dtype,
                )
                delta = (delta @ projection.transpose(0, 1)) @ projection
            updated_post = base_post + delta
        else:
            updated_post = base_post.clone()
            indices = list(self.positions)
            base_tokens = base_post[..., indices, :]
            delta = source[..., indices, :] - base_tokens
            if self.projection is not None:
                projection = self.projection.to(
                    device=base_post.device,
                    dtype=base_post.dtype,
                )
                delta = (delta @ projection.transpose(0, 1)) @ projection
            updated_post[..., indices, :] = base_tokens + delta
        return _with_hidden(output, updated_post - residual_input)

    def assert_consumed(self) -> None:
        """Fail unless every captured source and residual input was used exactly once."""

        if self.call_index != len(self.source):
            raise RuntimeError(
                f"Consumed {self.call_index} of {len(self.source)} source residuals"
            )
        if len(self.residual_inputs) != self.call_index:
            raise RuntimeError(
                f"Captured {len(self.residual_inputs)} residual inputs for "
                f"{self.call_index} MLP calls"
            )


class ResidualStreamAdd:
    """Add a cached direction to selected post-layer residual tokens.

    The hook is registered on a decoder layer's MLP output. SmolVLA adds that
    branch to the attention residual immediately afterward, so changing the MLP
    branch by ``direction`` changes the post-layer residual by exactly the same
    amount. Unlike :class:`ResidualStreamSwap`, this hook consumes no online source
    activation.
    """

    def __init__(
        self,
        directions: Sequence[Tensor],
        *,
        positions: Sequence[int],
        scale: float = 1.0,
    ):
        if not directions:
            raise ValueError("At least one cached direction is required")
        if not positions or len(set(positions)) != len(positions):
            raise ValueError("Positions must be nonempty and unique")
        if not torch.isfinite(torch.tensor(scale)):
            raise ValueError("Scale must be finite")
        self.directions = tuple(direction.detach() for direction in directions)
        self.positions = tuple(positions)
        self.scale = scale
        self.call_index = 0

    def __call__(
        self,
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: ModuleOutput,
    ) -> ModuleOutput:
        if self.call_index >= len(self.directions):
            raise RuntimeError("MLP was called more times than cached directions")
        branch = _hidden(output)
        if branch.ndim < 3:
            raise ValueError("Token-local residual addition requires [batch, sequence, hidden]")
        indices = list(self.positions)
        selected = branch[..., indices, :]
        direction = self.directions[self.call_index].to(
            device=branch.device,
            dtype=branch.dtype,
        )
        self.call_index += 1
        if (
            direction.ndim == selected.ndim
            and direction.shape[:-2] == selected.shape[:-2]
            and direction.shape[-2] == 1
            and direction.shape[-1] == selected.shape[-1]
        ):
            direction = direction.expand_as(selected)
        if direction.shape != selected.shape:
            raise ValueError(
                "Cached direction must match selected tokens or have one broadcast token: "
                f"direction={direction.shape}, selected={selected.shape}"
            )
        updated = branch.clone()
        updated[..., indices, :] = selected + self.scale * direction
        return _with_hidden(output, updated)

    def assert_consumed(self) -> None:
        """Fail unless every cached direction was applied exactly once."""

        if self.call_index != len(self.directions):
            raise RuntimeError(
                f"Consumed {self.call_index} of {len(self.directions)} cached directions"
            )
