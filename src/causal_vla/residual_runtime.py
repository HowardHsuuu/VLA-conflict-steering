"""Shared SmolVLA helpers for post-layer residual capture and action decoding."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from causal_vla.interventions import InputActivationCapture


def vlm_layers(policy: Any) -> tuple[list[Any], nn.Module]:
    """Return SmolVLA VLM decoder layers and its final normalization module."""

    wrapper = policy.model.vlm_with_expert
    text_model = wrapper.get_vlm_model().text_model
    return list(text_model.layers), cast(nn.Module, text_model.norm)


def post_layer_capture_module(
    layers: list[Any],
    final_norm: nn.Module,
    layer_index: int,
) -> nn.Module:
    """Return the next module whose input is a decoder layer's residual output."""

    if not 0 <= layer_index < len(layers):
        raise ValueError(f"Layer {layer_index} is outside [0, {len(layers) - 1}]")
    if layer_index + 1 < len(layers):
        return cast(nn.Module, layers[layer_index + 1].input_layernorm)
    return final_norm


def natural_action_chunk(policy: Any, batch: dict[str, Any], noise: Tensor) -> Tensor:
    """Predict one deterministic action chunk without intervention hooks."""

    policy.reset()
    with torch.inference_mode():
        result = policy.predict_action_chunk(batch, noise=noise.clone())
    return cast(Tensor, result)


def capture_post_layer_residuals(
    policy: Any,
    batch: dict[str, Any],
    noise: Tensor,
    *,
    layers: list[Any],
    final_norm: nn.Module,
    layer_indices: tuple[int, ...],
) -> tuple[Tensor, dict[int, tuple[Tensor, ...]]]:
    """Predict an action chunk and capture selected post-layer residual streams."""

    captures = {layer: InputActivationCapture() for layer in layer_indices}
    handles = [
        post_layer_capture_module(layers, final_norm, layer).register_forward_pre_hook(
            captures[layer]
        )
        for layer in layer_indices
    ]
    try:
        action = natural_action_chunk(policy, batch, noise)
    finally:
        for handle in handles:
            handle.remove()
    for layer, capture in captures.items():
        if not capture.records:
            raise RuntimeError(f"Post-layer residual {layer} was never captured")
    return action, {layer: tuple(capture.records) for layer, capture in captures.items()}


def capture_pre_and_post_layer_residuals(
    policy: Any,
    batch: dict[str, Any],
    noise: Tensor,
    *,
    layers: list[Any],
    final_norm: nn.Module,
    layer_indices: tuple[int, ...],
) -> tuple[
    Tensor,
    dict[int, tuple[Tensor, ...]],
    dict[int, tuple[Tensor, ...]],
]:
    """Capture each selected layer's input and output residual in one forward."""

    pre = {layer: InputActivationCapture() for layer in layer_indices}
    post = {layer: InputActivationCapture() for layer in layer_indices}
    handles = []
    for layer in layer_indices:
        handles.append(
            cast(nn.Module, layers[layer].input_layernorm).register_forward_pre_hook(pre[layer])
        )
        handles.append(
            post_layer_capture_module(layers, final_norm, layer).register_forward_pre_hook(
                post[layer]
            )
        )
    try:
        action = natural_action_chunk(policy, batch, noise)
    finally:
        for handle in handles:
            handle.remove()
    for layer in layer_indices:
        if not pre[layer].records or not post[layer].records:
            raise RuntimeError(f"Layer {layer} input or output residual was never captured")
        if len(pre[layer].records) != len(post[layer].records):
            raise RuntimeError(f"Layer {layer} residual capture call counts differ")
    return (
        action,
        {layer: tuple(capture.records) for layer, capture in pre.items()},
        {layer: tuple(capture.records) for layer, capture in post.items()},
    )


def decode_action(adapter: Any, action: Tensor) -> NDArray[np.float32]:
    """Postprocess one policy action and return the seven LIBERO controls."""

    processed = adapter.postprocessor(action)
    if isinstance(processed, Tensor):
        array = processed.detach().float().cpu().numpy()
    else:
        array = np.asarray(processed, dtype=np.float32)
    if array.ndim == 2:
        array = array[0]
    return np.asarray(array[:7], dtype=np.float32)
