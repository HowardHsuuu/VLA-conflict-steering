"""Training-only convex blending of local and global expert-output controllers."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import torch
from torch import Tensor

from causal_vla.action_flow import RidgeFlowRegressor
from causal_vla.causal_trace_runtime import PrefixSnapshot
from causal_vla.expert_knn_runtime import (
    ExpertPrototypeBank,
    _expert_prototype_from_payload,
    _expert_prototype_payload,
    _knn_prediction,
    load_expert_prototype_bank,
)
from causal_vla.expert_ridge_runtime import (
    ExpertRidgeBank,
    _expert_ridge_from_payload,
    _expert_ridge_payload,
    load_expert_ridge_bank,
)
from causal_vla.expert_trace_runtime import expert_layers


class BlendCrossValidationRow(TypedDict):
    knn_weight: float
    scale: float
    loo_target_mse_recovery: float
    loo_target_cosine: float


@dataclass(frozen=True)
class ExpertBlendBank:
    """A frozen convex blend of KNN and ridge corrections at one causal locus."""

    knn: ExpertPrototypeBank
    ridge: ExpertRidgeBank
    knn_weight: float
    scale: float

    def __post_init__(self) -> None:
        if self.knn.spec != self.ridge.spec or self.knn.num_steps != self.ridge.num_steps:
            raise ValueError("Blend components must share a steering spec and flow length")
        if not 0.0 <= self.knn_weight <= 1.0 or not math.isfinite(self.scale):
            raise ValueError("Blend weight and scale must be finite and valid")

    @property
    def spec(self):  # type: ignore[no-untyped-def]
        """Return the common causal steering specification."""

        return self.knn.spec

    @property
    def num_steps(self) -> int:
        """Return the common number of denoising steps."""

        return self.knn.num_steps


@dataclass(frozen=True)
class ExpertBlendFitReport:
    schema_version: int
    source_prototype_bank: str
    source_ridge_bank: str
    weight_candidates: tuple[float, ...]
    selected_knn_weight: float
    selected_scale: float
    cross_validation: tuple[BlendCrossValidationRow, ...]

    def to_dict(self) -> dict[str, object]:
        payload = cast(dict[str, object], asdict(self))
        payload["cross_validation"] = list(self.cross_validation)
        return payload


def save_expert_blend_bank(bank: ExpertBlendBank, path: str | Path) -> None:
    """Serialize a self-contained tensor-only blend bank."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": 1,
        "controller_type": "blend",
        "knn_weight": bank.knn_weight,
        "scale": bank.scale,
        "knn": _expert_prototype_payload(bank.knn),
        "ridge": _expert_ridge_payload(bank.ridge),
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def load_expert_blend_bank(path: str | Path) -> ExpertBlendBank:
    """Load and validate a self-contained tensor-only blend bank."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("controller_type") != "blend"
    ):
        raise ValueError("Unsupported expert blend bank schema")
    return ExpertBlendBank(
        knn=_expert_prototype_from_payload(payload.get("knn")),
        ridge=_expert_ridge_from_payload(payload.get("ridge")),
        knn_weight=float(payload["knn_weight"]),
        scale=float(payload["scale"]),
    )


def _loo_component_predictions(
    prototype: ExpertPrototypeBank,
    ridge: ExpertRidgeBank,
) -> tuple[Tensor, Tensor, Tensor]:
    knn_predictions: list[Tensor] = []
    ridge_predictions: list[Tensor] = []
    targets: list[Tensor] = []
    for step in range(prototype.num_steps):
        features = prototype.features[step]
        directions = prototype.directions[step]
        episodes = prototype.episode_indices[step]
        environment_steps = prototype.environment_steps[step]
        for heldout_episode in torch.unique(episodes):
            train = episodes != heldout_episode
            test_indices = torch.nonzero(~train, as_tuple=False).flatten()
            regressor = RidgeFlowRegressor.fit(
                features[train], directions[train], alpha=ridge.alpha
            )
            for index in test_indices:
                knn_predictions.append(
                    _knn_prediction(
                        features[index : index + 1],
                        features[train],
                        directions[train],
                        prototype.neighbors,
                        prototype_environment_steps=environment_steps[train],
                        query_environment_step=int(environment_steps[index]),
                        temporal_candidates=prototype.temporal_candidates,
                    )
                    * prototype.scale
                )
                ridge_predictions.append(
                    regressor.predict(features[index : index + 1]).reshape(1, -1) * ridge.scale
                )
                targets.append(directions[index : index + 1])
    return (
        torch.cat(knn_predictions).double(),
        torch.cat(ridge_predictions).double(),
        torch.cat(targets).double(),
    )


def fit_expert_blend_bank(
    prototype: ExpertPrototypeBank,
    ridge: ExpertRidgeBank,
    *,
    weight_candidates: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> tuple[ExpertBlendBank, tuple[BlendCrossValidationRow, ...]]:
    """Select a convex blend using leave-one-training-scene-out recovery only."""

    if (
        not weight_candidates
        or len(set(weight_candidates)) != len(weight_candidates)
        or any(not 0.0 <= weight <= 1.0 for weight in weight_candidates)
    ):
        raise ValueError("Blend weights must be unique values in [0, 1]")
    if prototype.spec != ridge.spec or prototype.num_steps != ridge.num_steps:
        raise ValueError("Blend components must share a steering spec and flow length")
    knn_prediction, ridge_prediction, target = _loo_component_predictions(prototype, ridge)
    rows: list[BlendCrossValidationRow] = []
    for weight in weight_candidates:
        prediction = weight * knn_prediction + (1.0 - weight) * ridge_prediction
        denominator = torch.sum(prediction * prediction)
        if float(denominator) <= 1e-20:
            raise ValueError("Blend cross-validation predictions are degenerate")
        scale = float(torch.sum(prediction * target) / denominator)
        residual = scale * prediction - target
        recovery = 1.0 - float(torch.sum(residual * residual) / torch.sum(target * target))
        cosine = float(
            torch.nn.functional.cosine_similarity(
                (scale * prediction).reshape(-1), target.reshape(-1), dim=0
            )
        )
        rows.append(
            {
                "knn_weight": weight,
                "scale": scale,
                "loo_target_mse_recovery": recovery,
                "loo_target_cosine": cosine,
            }
        )
    selected = max(rows, key=lambda row: row["loo_target_mse_recovery"])
    return (
        ExpertBlendBank(
            knn=prototype,
            ridge=ridge,
            knn_weight=selected["knn_weight"],
            scale=selected["scale"],
        ),
        tuple(rows),
    )


class AdaptiveExpertBlend:
    """Predict and add one blended correction from the current expert activation."""

    def __init__(
        self,
        bank: ExpertBlendBank,
        flow_step: int,
        *,
        environment_step: int | None = None,
        direction_sign: float = 1.0,
        random_seed: int | None = None,
    ):
        if direction_sign not in {-1.0, 1.0}:
            raise ValueError("Direction sign must be +1 or -1")
        self.bank = bank
        self.flow_step = flow_step
        self.environment_step = environment_step
        self.direction_sign = direction_sign
        self.random_seed = random_seed
        self.residual_input: Tensor | None = None
        self.calls = 0

    def capture_residual(self, _module: Any, inputs: tuple[object, ...]) -> None:
        if not inputs or not isinstance(inputs[0], Tensor):
            raise TypeError("Expected a tensor residual input")
        if self.residual_input is not None:
            raise RuntimeError("Expert residual was captured more than once")
        self.residual_input = inputs[0].detach().clone()

    def add_prediction(self, _module: Any, _inputs: tuple[object, ...], output: object) -> Tensor:
        if not isinstance(output, Tensor):
            raise TypeError("Expected a tensor expert MLP output")
        if self.bank.spec.interface == "post_residual" and self.residual_input is None:
            raise RuntimeError("Expert MLP ran before residual capture")
        if self.calls:
            raise RuntimeError("Expert blend hook was called more than once")
        self.calls += 1
        activation = (
            output
            if self.bank.spec.interface == "expert_output"
            else cast(Tensor, self.residual_input) + output
        )
        positions = list(self.bank.spec.token_positions)
        selected = activation[:, positions]
        query = selected.float().reshape(1, -1)
        knn_prediction = _knn_prediction(
            query,
            self.bank.knn.features[self.flow_step].to(query.device),
            self.bank.knn.directions[self.flow_step].to(query.device),
            self.bank.knn.neighbors,
            prototype_environment_steps=self.bank.knn.environment_steps[self.flow_step],
            query_environment_step=self.environment_step,
            temporal_candidates=self.bank.knn.temporal_candidates,
        ).reshape_as(selected)
        ridge_prediction = (
            self.bank.ridge.regressors[self.flow_step]
            .predict(selected)
            .reshape_as(selected)
            .to(knn_prediction)
        )
        weight = self.bank.knn_weight
        prediction = (
            weight * knn_prediction * self.bank.knn.scale
            + (1.0 - weight) * ridge_prediction * self.bank.ridge.scale
        )
        prediction = prediction * self.bank.scale * self.direction_sign
        if self.random_seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(self.random_seed)
            random_direction = torch.randn(
                prediction.shape, dtype=torch.float32, device="cpu", generator=generator
            )
            prediction_norm = torch.linalg.vector_norm(prediction.float().cpu())
            prediction = random_direction * (
                prediction_norm / torch.linalg.vector_norm(random_direction).clamp_min(1e-12)
            )
        updated = output.clone()
        updated[:, positions] = output[:, positions] + prediction.to(output)
        return updated

    def assert_applied(self) -> None:
        residual_missing = (
            self.bank.spec.interface == "post_residual" and self.residual_input is None
        )
        if residual_missing or self.calls != 1:
            raise RuntimeError("Adaptive expert blend was not applied exactly once")


def sample_with_expert_blend(
    policy: Any,
    snapshot: PrefixSnapshot,
    noise: Tensor,
    bank: ExpertBlendBank,
    *,
    environment_step: int | None = None,
    direction_sign: float = 1.0,
    random_seed: int | None = None,
) -> Tensor:
    """Integrate an action chunk with a frozen KNN-ridge blended correction."""

    model = policy.model
    layers, final_norm = expert_layers(policy)
    if bank.spec.layer >= len(layers) or bank.num_steps != int(model.config.num_steps):
        raise ValueError("Expert blend bank is incompatible with the policy")
    if bank.spec.interface == "expert_output" and bank.spec.layer != len(layers) - 1:
        raise ValueError("Expert-output steering requires the final expert layer")
    x_t = noise.clone()
    dt = -1.0 / bank.num_steps
    for flow_step in range(bank.num_steps):
        timestep = torch.full(
            (x_t.shape[0],),
            1.0 + flow_step * dt,
            dtype=torch.float32,
            device=x_t.device,
        )
        intervention = AdaptiveExpertBlend(
            bank,
            flow_step,
            environment_step=environment_step,
            direction_sign=direction_sign,
            random_seed=None if random_seed is None else random_seed + flow_step,
        )
        if bank.spec.interface == "expert_output":
            residual_handle = None
            prediction_handle = final_norm.register_forward_hook(intervention.add_prediction)
        else:
            layer = layers[bank.spec.layer]
            residual_handle = layer.post_attention_layernorm.register_forward_pre_hook(
                intervention.capture_residual
            )
            prediction_handle = layer.mlp.register_forward_hook(intervention.add_prediction)
        try:
            with torch.inference_mode():
                velocity = model.denoise_step(
                    prefix_pad_masks=snapshot.pad_masks,
                    past_key_values=snapshot.cache,
                    x_t=x_t,
                    timestep=timestep,
                )
        finally:
            if residual_handle is not None:
                residual_handle.remove()
            prediction_handle.remove()
        intervention.assert_applied()
        x_t = x_t + dt * velocity
    action_dim = int(policy.config.action_feature.shape[0])
    return x_t[..., :action_dim]


def _weights(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(not 0.0 <= item <= 1.0 for item in parsed):
        raise argparse.ArgumentTypeError("Expected comma-separated weights in [0, 1]")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m causal_vla.expert_blend_runtime")
    parser.add_argument("--prototype-bank", type=Path, required=True)
    parser.add_argument("--ridge-bank", type=Path, required=True)
    parser.add_argument("--weight-candidates", type=_weights, default=(0.0, 0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--bank-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    prototype = load_expert_prototype_bank(args.prototype_bank)
    ridge = load_expert_ridge_bank(args.ridge_bank)
    bank, rows = fit_expert_blend_bank(prototype, ridge, weight_candidates=args.weight_candidates)
    save_expert_blend_bank(bank, args.bank_output)
    report = ExpertBlendFitReport(
        schema_version=1,
        source_prototype_bank=str(args.prototype_bank.resolve()),
        source_ridge_bank=str(args.ridge_bank.resolve()),
        weight_candidates=args.weight_candidates,
        selected_knn_weight=bank.knn_weight,
        selected_scale=bank.scale,
        cross_validation=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
